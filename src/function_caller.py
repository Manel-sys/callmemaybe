import json
import sys
import time

from .decoder import Decoder, DecoderError
from .validation_models import (FunctionDefinition, OutputEntry,
                                ParamDefinition, ParamTypes, PromptDescript)


class FunctionCaller:

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
        sequence.extend(self.decoder.model.encode(text))

    def _gen_value(self, sequence: list[int], param_def: ParamDefinition,
                   terminator: str) -> float | str | bool:
        if param_def.type == ParamTypes.NUMBER:
            return self.decoder.gen_number(sequence, terminator)
        raise DecoderError(
            f"Generating '{param_def.type.value}' values is not "
            "implemented yet.")

    def _gen_parameters(self, sequence: list[int],
                        func_def: FunctionDefinition
                        ) -> dict[str, float | str | bool]:
        keys: list[str] = list(func_def.parameters)
        values: dict[str, float | str | bool] = {}

        if not keys:
            self._append_text(sequence, '", "parameters": {}}')
            return values

        self._append_text(sequence,
                          f'", "parameters": {{{json.dumps(keys[0])}: ')

        for k, key in enumerate(keys):
            is_last: bool = k == len(keys) - 1
            terminator: str = "}" if is_last else ","

            values[key] = self._gen_value(
                sequence, func_def.parameters[key], terminator)

            if is_last:
                self._append_text(sequence, "}")
            else:
                self._append_text(sequence, f" {json.dumps(keys[k + 1])}: ")

        return values

    def orchestrate(self) -> list[OutputEntry]:
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
