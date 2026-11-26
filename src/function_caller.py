from .decoder import Decoder
from .validation_models import FunctionDefinition, PromptDescript
import time


class FunctionCaller:
    def __init__(self, decoder: Decoder,
                 func_defs: list[FunctionDefinition],
                 prompts: list[PromptDescript]) -> None:
        self.decoder = decoder
        self.func_defs: list[FunctionDefinition] = func_defs
        self.prompts_descripts: list[PromptDescript] = prompts
        self.func_names: list[str] = [item.name for item in self.func_defs]
        self.prompts: list[str] = [item.prompt for
                                   item in self.prompts_descripts]

    def orquestrate(self) -> list[str]:
        generated_names: list[str] = []
        total_start: float = time.perf_counter()

        for prompt in self.prompts:
            start: float = time.perf_counter()

            context_prompt: str = self.decoder.build_prompt(
                prompt, self.func_defs)
            sequence: list[int] = self.decoder.model.encode(context_prompt)
            generated_func_name: str = self.decoder.gen_func_name(
                sequence, self.func_names)
            generated_names.append(generated_func_name)

            elapsed: float = time.perf_counter() - start
            print(f"{elapsed:.2f}s {generated_func_name} -< {prompt}")

        total: float = time.perf_counter() - total_start
        print(f"Total: {total:.2f}s for {len(self.prompts)} prompts")

        return generated_names
