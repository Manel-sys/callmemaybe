from .parser import Parser, ParserError
from .llm import Qwen3LLM, LLMError
from .decoder import Decoder
from .function_caller import FunctionCaller
import sys


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
    decoder = Decoder(llm)
    func_caller: FunctionCaller = FunctionCaller(decoder, func_defs, prompts)

    print("Testing gen_func_name:\n")
    func_caller.orquestrate()


if __name__ == "__main__":
    main()
