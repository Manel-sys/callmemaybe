"""Constrained decoding: turn model logits into a valid function call."""

import json
from typing import Any

import numpy as np
import numpy.typing as npt

from .grammar import Grammar, State
from .llm import Qwen3LLM
from .validation_models import FunctionDefinition, ParamTypes


class DecoderError(Exception):
    """Raised when the decoder reaches a state it cannot continue from."""


# Every state a string value can be in, from before its opening quote to
# inside its content. Each of them may be followed by a token that ends
# the string.
STRING_STATES: tuple[State, ...] = (State.STRING_START,
                                    State.STRING_AFTER_SPACE,
                                    State.STRING_CONTENT,
                                    State.STRING_AFTER_BACKSLASH)
STRING_ENDS: tuple[str, ...] = ('",', '"}')


class Decoder:
    """Generate function names and argument values token by token.

    Every token the model may pick is restricted by a mask, so the
    generated text always follows the expected JSON structure.
    """

    MAX_NUMBER_TOKENS: int = 12
    MAX_STRING_TOKENS: int = 64

    NAME_EXAMPLES: int = 3

    example_calls: tuple[tuple[str, str, dict[str, Any]], ...] = (
        ("What is 7 times -3?",
         "fn_multiply", {"x": 7, "y": -3}),
        ("Calculate the cube of 27",
         "fn_cube", {"n": 27}),
        ("Translate 'good morning' into French",
         "fn_translate", {"text": "good morning", "language": "French"}),
        ("Turn off the kitchen lights",
         "fn_set_light", {"room": "kitchen", "on": False}),
        ("Count the words in ''",
         "fn_count_words", {"text": ""}),
        (r"Open the file C:\Users\ana\notes.txt",
         "fn_open_file", {"path": r"C:\Users\ana\notes.txt"}),
    )

    def __init__(self, model: Qwen3LLM) -> None:
        """Store the model and build the masks used during generation."""
        self.model: Qwen3LLM = model
        self.state_masks: dict[State, npt.NDArray[np.bool_]] = (
            self._build_state_masks())
        self.terminator_masks: dict[str, npt.NDArray[np.bool_]] = (
            self._build_terminator_masks())
        self.string_end_masks: dict[tuple[State, str],
                                    npt.NDArray[np.bool_]] = (
            self._build_string_end_masks())
        # Number of real model calls, for timing and performance reports.
        self.model_calls: int = 0

    def _build_state_masks(self) -> dict[State, npt.NDArray[np.bool_]]:
        """Return, for each grammar state, the tokens that may follow it.

        A token is allowed when every character of its text is accepted
        by the grammar from that state, and it is usable at all
        (allow_mask).
        """
        state_masks: dict[State, npt.NDArray[np.bool_]] = {}
        for state in State:
            mask: npt.NDArray[np.bool_] = np.zeros(self.model.logits_size,
                                                   dtype=bool)
            for token_id, token_text in self.model.decoded_vocab.items():
                if Grammar.transition_token(state, token_text) is not None:
                    mask[token_id] = True

            mask &= self.model.allow_mask
            state_masks[state] = mask

        return state_masks

    def _build_terminator_masks(self) -> dict[str, npt.NDArray[np.bool_]]:
        """Return a mask for each number terminator: ',' and '}'.

        Each mask allows only the token that is exactly that terminator.
        Raises DecoderError if the vocabulary has no such token, since a
        number could then never end.
        """
        terminator_masks: dict[str, npt.NDArray[np.bool_]] = {}

        for terminator in (",", "}"):
            token_ids: list[int] = self.model.text_to_ids.get(terminator, [])
            if not token_ids:
                raise DecoderError(
                    f"No token in the vocabulary is exactly {terminator!r}.")

            mask: npt.NDArray[np.bool_] = np.zeros(self.model.logits_size,
                                                   dtype=bool)
            mask[token_ids] = True
            terminator_masks[terminator] = mask

        return terminator_masks

    def _build_string_end_masks(
            self) -> dict[tuple[State, str], npt.NDArray[np.bool_]]:
        """Return, per string state and end text, the tokens that end it.

        A token ends a string when its text is valid string content
        followed by exactly the end text ('",' or '"}'), for example
        ')",'. The content part must leave the grammar in STRING_CONTENT,
        otherwise the quote would be escaped and not end the string.
        The exact tokens '",' and '"}' are the case of empty content.
        """
        masks: dict[tuple[State, str], npt.NDArray[np.bool_]] = {
            (state, end): np.zeros(self.model.logits_size, dtype=bool)
            for state in STRING_STATES for end in STRING_ENDS
        }

        for token_id, token_text in self.model.decoded_vocab.items():
            for end in STRING_ENDS:
                if not token_text.endswith(end):
                    continue
                content: str = token_text[:-len(end)]
                for state in STRING_STATES:
                    if (Grammar.transition_token(state, content)
                            == State.STRING_CONTENT):
                        masks[(state, end)][token_id] = True

        for mask in masks.values():
            mask &= self.model.allow_mask
        return masks

    # ------------------------------------------------------------------
    # Prompts
    # ------------------------------------------------------------------

    @staticmethod
    def format_call(name: str, parameters: dict[str, Any]) -> str:
        """Return a function call as JSON, with the decoder's exact spacing."""
        return json.dumps({"name": name, "parameters": parameters},
                          ensure_ascii=False)

    @staticmethod
    def format_function(func: FunctionDefinition) -> str:
        """Return one function as a compact signature line."""
        params: str = ", ".join(
            f"{param_name}: {param_def.type.value}"
            for param_name, param_def in func.parameters.items()
        )
        return f"- {func.name}({params}): {func.description}"

    @staticmethod
    def _value_type(value: Any) -> ParamTypes | None:
        """Return the parameter type of an example value.

        bool is checked first, because in Python True is also an int.
        """
        if isinstance(value, bool):
            return ParamTypes.BOOLEAN
        if isinstance(value, (int, float)):
            return ParamTypes.NUMBER
        if isinstance(value, str):
            return ParamTypes.STRING
        return None

    def _format_examples(
            self, examples: list[tuple[str, str, dict[str, Any]]]) -> str:
        """Return the given few-shot examples as Request/Answer lines."""
        return "\n".join(
            f"Request: {request}\nAnswer: {self.format_call(name, params)}"
            for request, name, params in examples
        )

    def _examples_for(self, func_def: FunctionDefinition
                      ) -> list[tuple[str, str, dict[str, Any]]]:
        """Return the examples that show the function's parameter types.

        An example is kept when at least one of its values has a type
        the function also uses. A function with no parameters (or no
        matching example) gets the first few examples, so the prompt
        still shows the answer format.
        """
        wanted: set[ParamTypes] = {
            param.type for param in func_def.parameters.values()}
        selected: list[tuple[str, str, dict[str, Any]]] = [
            example for example in self.example_calls
            if wanted & {self._value_type(v) for v in example[2].values()}
        ]
        return selected or list(self.example_calls[:self.NAME_EXAMPLES])

    @staticmethod
    def _chat(system: str, prompt: str, answer_start: str) -> str:
        """Wrap system text and a request in Qwen's chat format.

        The text ends with the beginning of the assistant's answer, so the
        model continues from there.
        """
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            f"{answer_start}"
        )

    def build_name_prompt(self, prompt: str,
                          func_defs: list[FunctionDefinition]) -> str:
        """Build the input used to choose the function.

        It lists every function, and ends right where the model writes
        the function name.
        """
        functions: str = "\n".join(self.format_function(f) for f in func_defs)
        examples: str = self._format_examples(
            list(self.example_calls[:self.NAME_EXAMPLES]))
        system: str = (
            "You translate a user request into a single function call.\n"
            "Pick the one function that matches the request.\n"
            "Answer only with JSON in this format:\n"
            '{"name": "<function name>", "parameters": {...}}\n'
            "\n"
            f"Available functions:\n{functions}\n"
            "\n"
            "Examples (these example functions are not available):\n"
            f"{examples}"
        )
        return self._chat(system, prompt, '{"name": "')

    def build_value_prompt(self, prompt: str,
                           func_def: FunctionDefinition) -> str:
        """Build the input used to fill in the arguments.

        It shows only the chosen function, and ends right after its name,
        so the caller can continue with '", "parameters": {...'.
        """
        system: str = (
            "You fill in the arguments of a function call for a user "
            "request.\n"
            "Copy text arguments exactly as they appear in the request. "
            "An empty quoted text ('' or \"\") is an empty string.\n"
            "Answer only with JSON in this format:\n"
            '{"name": "<function name>", "parameters": {...}}\n'
            "\n"
            f"Function:\n{self.format_function(func_def)}\n"
            "\n"
            "Examples (these example functions are not available):\n"
            f"{self._format_examples(self._examples_for(func_def))}"
        )
        return self._chat(system, prompt, '{"name": "' + func_def.name)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _pick(self, sequence: list[int], allowed: set[int]) -> int:
        """Return the allowed token the model scores highest.

        The model is only called when more than one token is allowed.
        """
        if not allowed:
            raise DecoderError("No possible solution found.")
        if len(allowed) == 1:
            return next(iter(allowed))

        mask: npt.NDArray[np.bool_] = np.zeros(
            self.model.logits_size, dtype=bool)
        mask[list(allowed)] = True
        logits: npt.NDArray[np.float32] = (
            self.model.get_logits_from_input_ids(sequence))
        self.model_calls += 1
        masked: npt.NDArray[np.float32] = np.where(mask, logits, -np.inf)
        return int(np.argmax(masked))

    def _next_token_position(self, name_ids: list[int],
                             generated: str) -> int | None:
        """Return the position of a name's next natural token, if aligned.

        The name is aligned when the text generated so far ends exactly
        at one of its natural token boundaries. Returns None when it is
        not aligned or when the name is already complete.
        """
        text: str = ""
        for position, token_id in enumerate(name_ids):
            if text == generated:
                return position
            text += self.model.decoded_vocab[token_id]
            if len(text) > len(generated):
                return None
        return None

    def gen_func_name(self, sequence: list[int],
                      names: list[str]) -> str:
        """Generate a function name, asking the model only where needed.

        Tokens that every remaining candidate shares in its natural
        tokenization are written directly. Once a single candidate is
        left, the rest of its name is written directly too. The model
        is only called where the candidates really diverge.
        """
        func_name_ids: dict[str, list[int]] = {
            name: self.model.encode(name) for name in names}
        generated: str = ""
        # Names still on the token path the model chose (None: no
        # natural-token choice has been made yet).
        committed: set[str] | None = None
        max_steps: int = max(len(name) for name in names) + 1

        for _ in range(max_steps):
            candidates: list[str] = [
                name for name in names
                if name.startswith(generated)
                and (committed is None or name in committed)]
            if not candidates:
                raise DecoderError(
                    f"No function name starts with {generated!r}.")

            complete: bool = generated in names
            if complete and len(candidates) == 1:
                return generated

            if len(candidates) == 1:
                # The decision is made: write the rest of the name.
                name: str = candidates[0]
                position: int | None = self._next_token_position(
                    func_name_ids[name], generated)
                if position is not None:
                    rest_ids: list[int] = func_name_ids[name][position:]
                else:
                    rest_ids = self.model.encode(name[len(generated):])
                sequence.extend(rest_ids)
                return name

            # Shared natural token: every candidate continues with it.
            next_ids: list[int | None] = []
            for name in candidates:
                pos: int | None = self._next_token_position(
                    func_name_ids[name], generated)
                next_ids.append(None if pos is None
                                else func_name_ids[name][pos])

            first: int | None = next_ids[0]
            if (not complete and first is not None
                    and all(i == first for i in next_ids)):
                sequence.append(first)
                generated += self.model.decoded_vocab[first]
                continue

            # The candidates diverge. Allow each candidate's natural next
            # token; only candidates that are not at a natural token
            # boundary fall back to every prefix of their remaining text.
            allowed: set[int] = set()
            for name, next_id in zip(candidates, next_ids):
                if next_id is not None:
                    allowed.add(next_id)
                    continue
                rest: str = name[len(generated):]
                for k in range(1, len(rest) + 1):
                    allowed.update(self.model.text_to_ids.get(rest[:k], []))
            if complete:
                allowed.update(self.model.text_to_ids.get('"', []))

            token_id: int = self._pick(sequence, allowed)
            token: str = self.model.decoded_vocab[token_id]
            if token == '"':
                return generated

            # If the model chose a natural next token, keep only the
            # names whose natural split continues with that token.
            chosen: set[str] = {
                name for name, next_id in zip(candidates, next_ids)
                if next_id == token_id}
            if chosen:
                committed = chosen

            sequence.append(token_id)
            generated += token

        raise DecoderError("Function name generation did not finish.")

    def gen_number(self, sequence: list[int], terminator: str) -> float:
        """Generate a JSON number, then the given terminator.

        The model chooses the number one token at a time, constrained by
        the number grammar. It ends the number by choosing the terminator
        (',' or '}'), which stays in the sequence. Near the length limit,
        the terminator is forced as soon as the number can end.
        """
        state: State = State.NUMBER_START
        generated: str = ""

        for step in range(self.MAX_NUMBER_TOKENS):
            steps_left: int = self.MAX_NUMBER_TOKENS - step
            can_end: bool = state in Grammar.END_CAPABLE

            if can_end and steps_left <= 2:
                # Close to the limit: finish the number now.
                mask: npt.NDArray[np.bool_] = self.terminator_masks[terminator]
            elif can_end:
                mask = (self.state_masks[state]
                        | self.terminator_masks[terminator])
            else:
                mask = self.state_masks[state]

            allowed: set[int] = set(np.flatnonzero(mask).tolist())
            token_id: int = self._pick(sequence, allowed)
            sequence.append(token_id)

            token_text: str = self.model.decoded_vocab[token_id]
            if token_text == terminator:
                return float(generated)

            next_state: State | None = Grammar.transition_token(state,
                                                                token_text)
            if next_state is None:
                raise DecoderError(f"Invalid token {token_text!r} after "
                                   f"number {generated!r}.")

            generated += token_text
            state = next_state

        raise DecoderError(
            f"Number generation did not finish: {generated!r}.")

    def gen_string(self, sequence: list[int], terminator: str) -> str:
        """Generate a JSON string value, then its closing terminator.

        The sequence ends with the key and its colon. The model writes
        the opening quote itself, so it can use its natural first token
        (for example ' "' or ' "*'). It then writes the content, and ends
        the string with a token made of optional content, the closing
        quote and the terminator (for example '",' or ')",'), which stays
        in the sequence. Near the length limit, the end is forced as soon
        as the current state allows it.
        """
        state: State = State.STRING_START
        generated: str = ""
        end_text: str = '"' + terminator

        for step in range(self.MAX_STRING_TOKENS):
            steps_left: int = self.MAX_STRING_TOKENS - step
            end_mask: npt.NDArray[np.bool_] = (
                self.string_end_masks[(state, end_text)])

            if steps_left <= 2 and end_mask.any():
                # Close to the limit: finish the string now.
                mask: npt.NDArray[np.bool_] = end_mask
            else:
                mask = self.state_masks[state] | end_mask

            allowed: set[int] = set(np.flatnonzero(mask).tolist())
            token_id: int = self._pick(sequence, allowed)
            sequence.append(token_id)
            token_text: str = self.model.decoded_vocab[token_id]

            if end_mask[token_id]:
                # Keep everything before the closing quote. That text
                # starts with an optional space and the opening quote:
                # drop them, then decode the JSON escapes.
                generated += token_text[:-len(end_text)]
                content: str = generated[generated.index('"') + 1:]
                value: str = json.loads('"' + content + '"')
                return value

            next_state: State | None = Grammar.transition_token(state,
                                                                token_text)
            if next_state is None:
                raise DecoderError(f"Invalid token {token_text!r} inside "
                                   f"string {generated!r}.")

            generated += token_text
            state = next_state

        raise DecoderError(
            f"String generation did not finish: {generated!r}.")
