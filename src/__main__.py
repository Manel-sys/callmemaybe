from .parser import Parser, ParserError
from .llm import Qwen3LLM, LLMError
import sys


def main() -> None:
    parser: Parser = Parser()

    try:
        func_defs, prompts = parser.parse()
    except ParserError as e:
        print(f"ParserError: {e}")
        sys.exit(1)

    print("Testing parser:")
    print(parser.func_defs_path)
    print(parser.func_call_tests_path)
    print(parser.func_calls_path)

    print("Validated function definitions:")
    for func_def in func_defs:
        print(func_def)

    try:
        llm = Qwen3LLM()
    except LLMError as e:
        print(f"LLMError: {e}")
        sys.exit(1)

    print(len(llm.decoded_vocab))
    print(llm.logits_size)
    logits = llm.get_logits_from_input_ids(llm.encode("There once was"))
    print(logits)
    print(llm.decode([999999]))


if __name__ == "__main__":
    main()
