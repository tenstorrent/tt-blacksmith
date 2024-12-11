# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Annotated, Any, Dict

from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema
from enum import Enum


def create_mapped_type(map_values: Dict[str, Any]):
    """
    Create a Pydantic type that maps a string to a specific value from a dictionary.
    Class created by following the Pydantic documentation on adding third-party types:
    https://docs.pydantic.dev/latest/concepts/types/#handling-third-party-types
    """
    reverse_map_values = {v: k for k, v in map_values.items()}

    # Create a serializer schema that converts the value to a string.
    serializer_schema = core_schema.plain_serializer_function_ser_schema(lambda v: reverse_map_values.get(v, str(v)))

    def validate(value: str):
        """
        Function to validate the value passed to the Pydantic type.
        """

        # If the value is a string, check if there is conversion to a value in the map_values dictionary.
        if isinstance(value, str):
            if value in map_values:
                return map_values[value]
            raise ValueError(f"Invalid value '{value}', expected one of {list(map_values.keys())}")

        # If the value is not string, raise the error
        raise ValueError(f"Invalid value '{value}', expected a `str`")

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


class Device(str, Enum):
    cpu = "cpu"
    tt = "tt"


DeviceType = create_mapped_type({k: v for k, v in Device.__members__.items()})
