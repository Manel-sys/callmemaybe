from enum import Enum


class State(Enum):
    NUMBER_START = "number_start"
    NUMBER_AFTER_MINUS = "number_after_minus"
    NUMBER_LEADING_ZERO = "number_leading_zero"
    NUMBER_INT_DIGITS = "number_int_digits"
    NUMBER_AFTER_DOT = "number_after_dot"
    NUMBER_FRAC_DIGITS = "number_frac_digits"
    STRING_START = "string_start"
    STRING_AFTER_SPACE = "string_after_space"
    STRING_CONTENT = "string_content"
    STRING_AFTER_BACKSLASH = "string_after_backslash"


class Grammar:
    END_CAPABLE: set[State] = {State.NUMBER_LEADING_ZERO,
                               State.NUMBER_INT_DIGITS,
                               State.NUMBER_FRAC_DIGITS,
                               State.STRING_CONTENT,
                               }

    STRING_STATES: tuple[State, ...] = (State.STRING_START,
                                        State.STRING_AFTER_SPACE,
                                        State.STRING_CONTENT,
                                        State.STRING_AFTER_BACKSLASH)

    STRING_ENDS: tuple[str, ...] = ('",', '"}')

    @staticmethod
    def transition(state: State, char: str) -> State | None:
        if state == State.NUMBER_START:
            if char == "-":
                return State.NUMBER_AFTER_MINUS
            elif char in "123456789":
                return State.NUMBER_INT_DIGITS
            elif char == "0":
                return State.NUMBER_LEADING_ZERO
            else:
                return None

        if state == State.NUMBER_AFTER_MINUS:
            if char in "123456789":
                return State.NUMBER_INT_DIGITS
            elif char == "0":
                return State.NUMBER_LEADING_ZERO
            else:
                return None

        if state == State.NUMBER_LEADING_ZERO:
            if char == ".":
                return State.NUMBER_AFTER_DOT
            else:
                return None

        if state == State.NUMBER_INT_DIGITS:
            if char in "0123456789":
                return State.NUMBER_INT_DIGITS
            elif char == ".":
                return State.NUMBER_AFTER_DOT
            else:
                return None

        if state == State.NUMBER_AFTER_DOT:
            if char in "0123456789":
                return State.NUMBER_FRAC_DIGITS
            else:
                return None

        if state == State.NUMBER_FRAC_DIGITS:
            if char in "0123456789":
                return State.NUMBER_FRAC_DIGITS
            else:
                return None

        # Before a string: an optional space, then the opening quote.
        if state == State.STRING_START:
            if char == " ":
                return State.STRING_AFTER_SPACE
            elif char == '"':
                return State.STRING_CONTENT
            else:
                return None

        if state == State.STRING_AFTER_SPACE:
            if char == '"':
                return State.STRING_CONTENT
            else:
                return None

        if state == State.STRING_CONTENT:
            if char == "\\":
                return State.STRING_AFTER_BACKSLASH
            elif char == '"':
                return None
            elif ord(char) < 32:
                return None
            else:
                return State.STRING_CONTENT

        if state == State.STRING_AFTER_BACKSLASH:
            if char in '"\\/bfnrt':
                return State.STRING_CONTENT
            else:
                return None

        return None

    @staticmethod
    def transition_token(state: State, token_text: str) -> State | None:
        current: State = state
        for char in token_text:
            next_state: State | None = Grammar.transition(current, char)
            if next_state is None:
                return None
            current = next_state

        return current
