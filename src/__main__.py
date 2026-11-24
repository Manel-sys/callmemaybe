from .parser import Parser, ParserError
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


if __name__ == "__main__":
    main()
