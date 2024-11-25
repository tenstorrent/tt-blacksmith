# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Annotated, Dict, Any, Type

from pydantic import BaseModel, GetJsonSchemaHandler, GetCoreSchemaHandler
from pydantic_core import core_schema
from pydantic.json_schema import JsonSchemaValue


def create_mapped_type(map_values: Dict[str, Any], map_type: Type[Any]):
    """
    Create a Pydantic type that maps a string to a specific value from a dictionary.
    Class created by following the Pydantic documentation on adding third-party types:
    https://docs.pydantic.dev/latest/concepts/types/#handling-third-party-types
    """
    reverse_map_values = {v: k for k, v in map_values.items()}

    # Create a serializer schema that converts the value to a string.
    serializer_schema = core_schema.plain_serializer_function_ser_schema(lambda v: reverse_map_values.get(v, str(v)))

    def validate(value):
        """
        Function to validate the value passed to the Pydantic type.
        """

        # If the value is a string, check if there is conversion to a value in the map_values dictionary.
        if isinstance(value, str):
            if value in map_values:
                return map_values[value]
            raise ValueError(f"Invalid value '{value}', expected one of {list(map_values.keys())}")

        # If the value is of the map_type, check if it is in the map_values dictionary.
        if isinstance(value, map_type) or (isinstance(value, type) and issubclass(value, map_type)):
            if value in map_values.values():
                return value
            raise ValueError(f"Invalid value '{value}', expected one of {list(map_values.values())}")

        # If the value is neither a string nor of the map_type, raise the error
        raise ValueError(f"Invalid value '{value}', expected a `str` or `{map_type}`")

    class _MapTypePydanticAnnotation:
        @classmethod
        def __get_pydantic_core_schema__(
            cls,
            _source_type: Any,
            _handler: GetCoreSchemaHandler,
        ) -> core_schema.CoreSchema:
            return core_schema.no_info_plain_validator_function(
                validate,
                serialization=serializer_schema,
            )

        @classmethod
        def __get_pydantic_json_schema__(
            cls, _core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
        ) -> JsonSchemaValue:
            return {"type": "string", "enum": list(map_values.keys())}

    return Annotated[Any, _MapTypePydanticAnnotation]
