from typing import Dict, List
from pydantic import (
    BaseModel,
    RootModel,
    Field,
    field_validator
)


class PromptItem(BaseModel):
    """
    Represents an individual natural language prompt.
    Attributes:
        prompt (str): The text content of the prompt.
    """
    prompt: str = Field(..., min_length=1)

    @field_validator('prompt')
    @classmethod
    def check_not_blank(cls, v: str) -> str:
        """
        Validates that the prompt contains non-whitespace characters.
        """
        if not v.strip():
            raise ValueError(
                'The prompt cannot be empty or contain only spaces'
                )
        return v


class PromptList(RootModel[List[PromptItem]]):
    """
    Represents a collection of prompt items.
    """
    pass


class Parameter(BaseModel):
    """
    Defines the data type for a function parameter.
    Attributes:
        type (str): The expected data type
        (e.g., 'number', 'string', 'boolean').
    """
    type: str


class ReturnType(BaseModel):
    """
    Defines the return type for a function.
    Attributes:
        type (str): The return data type.
    """
    type: str


class FunctionDefinition(BaseModel):
    """
    Defines the schema for a callable function.
    Attributes:
        name (str): The unique identifier for the function.
        description (str): A brief explanation of what the function does.
        parameters (Dict[str, Parameter]): Mapping of parameter names to their
        type definitions.
        returns (ReturnType): The expected return type of the function.
    """
    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    parameters: Dict[str, Parameter]
    returns: ReturnType


class FunctionList(RootModel[List[FunctionDefinition]]):
    """
    Represents a collection of function definitions.
    """
    pass
