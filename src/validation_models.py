from enum import Enum
from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Any, Optional


class ParamTypes(str, Enum):
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"


class ParamDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: ParamTypes
    description: Optional[str] = None


class FunctionDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: dict[str, ParamDefinition]
    returns: ParamDefinition

    @field_validator("name")
    @classmethod
    def validate_name(cls, name: str) -> str:
        if not name.strip():
            raise ValueError("Function names must not be empty in"
                             " function definitions.")
        return name

    @field_validator("description")
    @classmethod
    def validate_description(cls, description: str) -> str:
        if not description.strip():
            raise ValueError("Function descriptions must not be empty in"
                             " function definitions.")
        return description

    @field_validator("parameters")
    @classmethod
    def validate_parameter_keys(
        cls, parameters: dict[str, ParamDefinition]
    ) -> dict[str, ParamDefinition]:
        for key in parameters:
            if not key.strip():
                raise ValueError("Parameter names inside function definition"
                                 " must not be empty.")

        return parameters


class PromptDescript(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1)

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("Empty prompt found.")

        return prompt


class OutputEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1)
    name: str = Field(min_length=1)
    parameters: dict[str, Any]

    @field_validator("name")
    @classmethod
    def validate_name(cls, name: str) -> str:
        if not name.strip():
            raise ValueError("Function name in output schema"
                             " must not be empty.")

        return name

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("Prompt in output schema must not be empty.")

        return prompt
