from typing import Any

from pydantic import ValidationError

from src.models import (
    PromptList,
    FunctionList,
)


def parse_prompts(data: Any) -> PromptList | None:
    """
    Validates and parses the raw input data into a PromptList object.
    Args:
        data (Any): The raw input data (usually parsed JSON).
    Returns:
        PromptList | None: A valid PromptList object if parsing succeeds,
                           otherwise None if validation errors occur.
    """
    try:
        return PromptList.model_validate(data)

    except ValidationError as e:
        for error in e.errors():
            loc = " -> ".join(str(x) for x in error["loc"])
            print(f"Error in {loc}: {error['msg']}")

        return None


def parse_functions(data: Any) -> FunctionList | None:
    """
    Validates and parses the raw function definitions into a
    FunctionList object.
    Args:
        data (Any): The raw function definition data (usually parsed JSON).
    Returns:
        FunctionList | None: A valid FunctionList object if parsing succeeds,
                             otherwise None if validation errors occur.
    """
    try:
        return FunctionList.model_validate(data)

    except ValidationError as e:
        for error in e.errors():
            loc = error["loc"]
            msg = error["msg"]

            if len(loc) >= 2 and isinstance(loc[0], int):
                item = loc[0] + 1
                field = ".".join(map(str, loc[1:]))

                print(
                    f"Function error #{item}, field '{field}': {msg}"
                )
            else:
                print(f"Error: {msg}")

        return None
