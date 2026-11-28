import sys
import json
from .validation_models import FunctionDefinition, PromptDescript
from pydantic import ValidationError
from typing import Any


class ParserError(Exception):
    pass


class Parser:
    default_func_defs: str = "data/input/functions_definition.json"
    default_func_call_tests: str = "data/input/function_calling_tests.json"
    default_func_calls: str = "data/output/function_calls.json"
    valid_flags: list[str] = ["--functions_definition", "--input", "--output"]

    input_format: str = "[--functions_definition <function_definition_file>]" \
                        " [--input <input_file>] [--" \
                        "output <output_file>]" \
                        f"\nOptional flags are {valid_flags}."

    def __init__(self) -> None:
        self.func_defs_path: str = self.default_func_defs
        self.func_call_tests_path: str = self.default_func_call_tests
        self.func_calls_path: str = self.default_func_calls
        self.parse_input_args()

    def parse_input_args(self) -> None:
        args: list[str] = sys.argv[1:]
        seen: set[str] = set()

        i: int = 0
        while i < len(args):
            flag: str = args[i]

            if flag not in self.valid_flags:
                raise ParserError("Unrecognized input format provided."
                                  "\nThe program expects the following"
                                  " input format in no particular"
                                  f" order: {self.input_format}")

            if flag in seen:
                raise ParserError(f"{flag} provided more than once."
                                  "\nThe program expects the following"
                                  " input format in no particular"
                                  f" order: {self.input_format}")

            if i + 1 >= len(args):
                raise ParserError(f"Missing file path after {flag}."
                                  "\nThe program expects the following"
                                  " input format in no particular"
                                  f" order: {self.input_format}")

            seen.add(flag)
            value: str = args[i + 1]

            if value in self.valid_flags:
                raise ParserError(f"Invalid file path {value} "
                                  f"provided after {flag}. "
                                  "File path cannot be itself a flag from"
                                  f" {self.valid_flags}")

            if flag == "--functions_definition":
                self.func_defs_path = value
            elif flag == "--input":
                self.func_call_tests_path = value
            else:
                self.func_calls_path = value

            i += 2

    @staticmethod
    def load_json_list(path: str) -> list[Any]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except OSError as e:
            raise ParserError(f"{type(e).__name__}: {e}")

        except json.JSONDecodeError as e:
            raise ParserError(f"{path}: invalid JSON: {e}")

        except UnicodeDecodeError as e:
            raise ParserError(f"{path}: file is not valid UTF-8: {e}")

        if not isinstance(data, list):
            raise ParserError(f"{path}: expected a JSON list, "
                              f"got {type(data).__name__}")

        return data

    @staticmethod
    def format_validation_error(error: ValidationError) -> str:
        messages: list[str] = []
        for err in error.errors():
            location: str = ".".join(str(part) for part in err["loc"])
            if location:
                messages.append(f"{location}: {err['msg']}")
            else:
                messages.append(err["msg"])
        return "; ".join(messages)

    @staticmethod
    def only_unsupported_types(error: ValidationError) -> bool:
        return all(
            err["type"] == "enum" and err["loc"][-1:] == ("type",)
            for err in error.errors()
        )

    def parse(self) -> tuple[list[FunctionDefinition], list[PromptDescript]]:
        func_definitions: list[FunctionDefinition] = []
        prompts: list[PromptDescript] = []
        func_names: set[str] = set()

        for index, item in enumerate(
                self.load_json_list(self.func_defs_path)):
            try:
                func_def = FunctionDefinition.model_validate(item)
            except ValidationError as e:
                if not self.only_unsupported_types(e):
                    raise ParserError(f"{self.func_defs_path},"
                                      f" entry {index}: "
                                      f"{self.format_validation_error(e)}"
                                      )

                print(f"Warning: {self.func_defs_path}, entry {index}: "
                      "skipping function with an unsupported type: "
                      f"{self.format_validation_error(e)}\n")
                continue

            if func_def.name in func_names:
                raise ParserError(f"{self.func_defs_path}, entry {index}:"
                                  " Duplicate function name found "
                                  f"for {func_def.name}"
                                  "\nFunction names cannot be the same.")

            func_names.add(func_def.name)
            func_definitions.append(func_def)

        if not func_definitions:
            raise ParserError("No usable function definitions in file: "
                              f"{self.func_defs_path}."
                              "\nAt least one function definition with"
                              " supported types must be provided.")

        for index, item in enumerate(
                self.load_json_list(self.func_call_tests_path)):
            try:
                prompt = PromptDescript.model_validate(item)
            except ValidationError as e:
                print(f"Warning: {self.func_call_tests_path}, entry {index}: "
                      "skipping invalid prompt: "
                      f"{self.format_validation_error(e)}\n")
                continue

            prompts.append(prompt)

        if not prompts:
            raise ParserError("No prompts were provided in file: "
                              f"{self.func_call_tests_path}."
                              "\nAt least one prompt must be provided.")

        return func_definitions, prompts
