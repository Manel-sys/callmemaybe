"""Run the decoder over every prompt and collect the function calls."""

import json
import sys
import time

from .decoder import Decoder, DecoderError
from .validation_models import (FunctionDefinition, OutputEntry,
                                ParamDefinition, ParamTypes, PromptDescript)


class FunctionCaller:
    """Turn each prompt into a function call, using the decoder."""

    def __init__(self, decoder: Decoder,
                 func_defs: list[FunctionDefinition],
                 prompts: list[PromptDescript]) -> None:
        """Store the decoder, the function definitions and the prompts."""
        self.decoder: Decoder = decoder
        self.func_defs: list[FunctionDefinition] = func_defs
        self.prompts: list[str] = [item.prompt for item in prompts]
        self.func_names: list[str] = [item.name for item in func_defs]
        self.funcs_by_name: dict[str, FunctionDefinition] = {
            item.name: item for item in func_defs}

    def _append_text(self, sequence: list[int], text: str) -> None:
        """Encode a fixed piece of text and add it to the sequence."""
        sequence.extend(self.decoder.model.encode(text))

    @staticmethod
    def _key_text(key: str, param_def: ParamDefinition) -> str:
        """Return the fixed text written before a value.

        For a number: the quoted key, the colon and a space. For a
        string, only the quoted key and the colon: the model writes the
        space and the opening quote itself, so it can use its natural
        first token (for example ' "*').
        """
        if param_def.type == ParamTypes.STRING:
            return f"{json.dumps(key)}:"
        return f"{json.dumps(key)}: "

    def _gen_value(self, sequence: list[int], param_def: ParamDefinition,
                   terminator: str) -> float | str | bool:
        """Generate one parameter value of the right type."""
        if param_def.type == ParamTypes.NUMBER:
            return self.decoder.gen_number(sequence, terminator)
        if param_def.type == ParamTypes.STRING:
            return self.decoder.gen_string(sequence, terminator)
        raise DecoderError(
            f"Generating '{param_def.type.value}' values is not "
            "implemented yet.")

    def _gen_parameters(self, sequence: list[int],
                        func_def: FunctionDefinition
                        ) -> dict[str, float | str | bool]:
        """Generate every parameter of the chosen function, in order.

        The sequence ends right after the function name. Fixed text
        (keys and structure) is written directly, each piece encoded in
        one go; the model only generates the values and their
        terminators.
        """
        params: dict[str, ParamDefinition] = func_def.parameters
        keys: list[str] = list(params)
        values: dict[str, float | str | bool] = {}

        if not keys:
            self._append_text(sequence, '", "parameters": {}}')
            return values

        self._append_text(
            sequence,
            '", "parameters": {' + self._key_text(keys[0], params[keys[0]]))

        for k, key in enumerate(keys):
            is_last: bool = k == len(keys) - 1
            terminator: str = "}" if is_last else ","

            values[key] = self._gen_value(sequence, params[key], terminator)

            if is_last:
                self._append_text(sequence, "}")
            else:
                next_key: str = keys[k + 1]
                self._append_text(
                    sequence, " " + self._key_text(next_key, params[next_key]))

        return values

    def orchestrate(self) -> list[OutputEntry]:
        """Generate a function call for every prompt.

        The function is chosen from a prompt that lists every function.
        The arguments are then generated from a shorter prompt that only
        shows the chosen function, which makes each model call cheaper.
        """
        results: list[OutputEntry] = []
        total_start: float = time.perf_counter()

        for prompt in self.prompts:
            start: float = time.perf_counter()
            calls_before: int = self.decoder.model_calls

            name_sequence: list[int] = self.decoder.model.encode(
                self.decoder.build_name_prompt(prompt, self.func_defs))
            name: str = self.decoder.gen_func_name(name_sequence,
                                                   self.func_names)
            func_def: FunctionDefinition = self.funcs_by_name[name]

            # Values are generated from a shorter prompt that shows only
            # the chosen function.
            sequence: list[int] = self.decoder.model.encode(
                self.decoder.build_value_prompt(prompt, func_def))

            try:
                parameters: dict[str, float | str | bool] = (
                    self._gen_parameters(sequence, func_def))
            except DecoderError as e:
                print(f"Skipped {prompt!r}: {e}", file=sys.stderr)
                continue

            results.append(OutputEntry(prompt=prompt, name=name,
                                       parameters=parameters))

            elapsed: float = time.perf_counter() - start
            calls: int = self.decoder.model_calls - calls_before
            print(f"{elapsed:.2f}s {calls} calls "
                  f"{name} {parameters} <- {prompt}")

        total: float = time.perf_counter() - total_start
        print(f"Total: {total:.2f}s for {len(self.prompts)} prompts")

        return results
