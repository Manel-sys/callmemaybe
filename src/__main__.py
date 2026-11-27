from .parser import Parser, ParserError
from .llm import Qwen3LLM, LLMError
from .decoder import Decoder
from .function_caller import FunctionCaller
import sys
import time


def main() -> None:

    try:
        parser: Parser = Parser()
        func_defs, prompts = parser.parse()
    except ParserError as e:
        print(f"ParserError: {e}")
        sys.exit(1)

    print("Testing parser:")
    print(parser.func_defs_path)
    print(parser.func_call_tests_path)
    print(parser.func_calls_path)

    try:
        llm = Qwen3LLM()
    except LLMError as e:
        print(f"LLMError: {e}")
        sys.exit(1)

    print(len(llm.decoded_vocab))
    print(llm.logits_size)

    print(llm.encode("<|im_start|>"))
    start: float = time.perf_counter()
    decoder = Decoder(llm)
    end: float = time.perf_counter() - start
    print(f"It takes {end:.2f}s to build the state masks")
    print("State masks contents:")
    for state, mask in decoder.state_masks.items():
        print(f"{state.name:<22} {int(mask.sum())} tokens")
    func_caller: FunctionCaller = FunctionCaller(decoder, func_defs, prompts)

    print("Testing gen_func_name:\n")
    func_caller.orchestrate()


if __name__ == "__main__":
    main()
