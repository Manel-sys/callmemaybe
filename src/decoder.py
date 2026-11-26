from .llm import Qwen3LLM
from .validation_models import FunctionDefinition
from typing import Any
import numpy as np
import numpy.typing as npt
import json


class DecoderError(Exception):
    pass


class Decoder:
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

    def gen_func_name(self, sequence: list[int],
                      names: list[str]) -> str:
        generated: str = ""

        while True:
            candidates: list[str] = [name for name in names
                                     if name.startswith(generated)]

            complete: bool = generated in names
            if complete and len(candidates) == 1:
                return generated

            allowed: set[int] = set()
            for name in candidates:
                rest = name[len(generated):]
                for k in range(1, len(rest) + 1):
                    allowed.update(self.model.text_to_ids.get(rest[:k], []))

            if complete:
                allowed.update(self.model.text_to_ids.get('"', []))

            if not allowed:
                raise DecoderError("No possible solution found.")

            if len(allowed) == 1:
                token_id: int = list(allowed)[0]
            else:
                mask: npt.NDArray[np.bool_] = np.zeros(
                    self.model.logits_size, dtype=bool)

                mask[list(allowed)] = True

                logits: npt.NDArray[np.float32] = (
                    self.model.get_logits_from_input_ids(sequence)
                    )

                masked: npt.NDArray[np.float32] = np.where(
                    mask, logits, -np.inf)

                token_id = int(np.argmax(masked))

            token = self.model.decoded_vocab[token_id]

            if token == '"':
                return generated

            sequence.append(token_id)
            generated += token
