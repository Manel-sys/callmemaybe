from .llm import Qwen3LLM
from .validation_models import FunctionDefinition
from typing import Any
import numpy as np
import numpy.typing as npt
from .grammar import Grammar, State
import json


class DecoderError(Exception):
    pass


class Decoder:
    MAX_NUMBER_TOKENS: int = 20

    example_calls: list[tuple[str, str, dict[str, Any]]] = [
         ("What is 7 times -3?",
          "fn_multiply", {"x": 7, "y": -3}),
         ("Translate 'good morning' into French",
          "fn_translate", {"text": "good morning", "language": "French"}),
         ("Turn off the kitchen lights",
          "fn_set_light", {"room": "kitchen", "on": False}),
         ("Count the words in ''",
          "fn_count_words", {"text": ""}),
         (r"Find files matching \d+\.log",
          "fn_find_files", {"pattern": r"\d+\.log"}),
        ]

    def __init__(self, model: Qwen3LLM) -> None:
        self.model: Qwen3LLM = model
        self.state_masks: dict[State, npt.NDArray[np.bool_]] = (
            self._build_state_masks())
        self.terminator_masks: dict[str, npt.NDArray[np.bool_]] = (
            self._build_terminator_masks()
        )

    def _build_state_masks(self) -> dict[State, npt.NDArray[np.bool_]]:
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
        terminator_masks: dict[str, npt.NDArray[np.bool_]] = {}

        for terminator in (",", "}"):
            token_ids: list[int] = self.model.text_to_ids.get(terminator, [])
            if not token_ids:
                raise DecoderError(
                    f"No token in the vocabulary is exactly '{terminator}'.")

            mask: npt.NDArray[np.bool_] = np.zeros(self.model.logits_size,
                                                   dtype=bool)

            mask[token_ids] = True

            terminator_masks[terminator] = mask

        return terminator_masks

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

    def build_prompt(self, prompt: str,
                     func_defs: list[FunctionDefinition]) -> str:
        """Build the full model input for one request.

        The text ends with the opening of the answer, so the model's next
        token is the first token of the function name.
        """
        functions: str = "\n".join(self.format_function(f) for f in func_defs)
        examples: str = "\n".join(
            f"Request: {request}\nAnswer: {self.format_call(name, params)}"
            for request, name, params in self.example_calls
        )

        system: str = (
            "You translate a user request into a single function call.\n"
            "Pick the one function that matches the request and fill in "
            "its arguments.\n"
            "Copy text arguments exactly as they appear in the request. "
            "An empty quoted text ('' or \"\") is an empty string.\n"
            "Answer only with JSON in this format:\n"
            '{"name": "<function name>", "parameters": {...}}\n'
            "\n"
            f"Available functions:\n{functions}\n"
            "\n"
            "Examples (these example functions are not available):\n"
            f"{examples}"
        )

        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            '{"name": "'
        )

    def _next_token_position(self, name_ids: list[int],
                             generated: str) -> int | None:
        text: str = ""
        for position, token_id in enumerate(name_ids):
            if text == generated:
                return position
            text += self.model.decoded_vocab[token_id]
            if len(text) > len(generated):
                return None
        return None

    def _pick(self, sequence: list[int], allowed: set[int]) -> int:
        if not allowed:
            raise DecoderError("No possible solution found.")
        if len(allowed) == 1:
            return list(allowed)[0]
        mask: npt.NDArray[np.bool_] = np.zeros(
            self.model.logits_size, dtype=bool)
        mask[list(allowed)] = True
        logits: npt.NDArray[np.float32] = (
            self.model.get_logits_from_input_ids(sequence))
        masked: npt.NDArray[np.float32] = np.where(mask, logits, -np.inf)
        return int(np.argmax(masked))

    def gen_func_name(self, sequence: list[int],
                      names: list[str]) -> str:
        func_name_ids: dict[str, list[int]] = {
            name: self.model.encode(name) for name in names}
        generated: str = ""
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
                name: str = candidates[0]
                position: int | None = self._next_token_position(
                    func_name_ids[name], generated)
                if position is not None:
                    rest_ids: list[int] = func_name_ids[name][position:]
                else:
                    rest_ids = self.model.encode(name[len(generated):])
                sequence.extend(rest_ids)
                return name

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

            chosen: set[str] = {
                name for name, next_id in zip(candidates, next_ids)
                if next_id == token_id}
            if chosen:
                committed = chosen

            sequence.append(token_id)
            generated += token

        raise DecoderError("Function name generation did not finish.")

    def gen_number(self, sequence: list[int], terminator: str) -> float:
        state: State = State.NUMBER_START
        generated: str = ""

        for step in range(self.MAX_NUMBER_TOKENS):
            steps_left: int = self.MAX_NUMBER_TOKENS - step
            can_end: bool = state in Grammar.END_CAPABLE

            if can_end and steps_left <= 2:
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
                raise DecoderError(f"Invalid token '{token_text}'"
                                   f" after number '{generated}'.")

            generated += token_text
            state = next_state

        raise DecoderError(f"Number generation did not finish: {generated}")
            
