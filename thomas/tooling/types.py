from typing import Annotated, Dict, Any, Type
from pydantic import GetCoreSchemaHandler
from pydantic import BaseModel, GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic_core import core_schema
from pydantic.json_schema import JsonSchemaValue


def create_mapped_type(map_dtype: Dict[str, Any], dtype_type: Type[Any]):
    reverse_map_dtype = {v: k for k, v in map_dtype.items()}

    def validate(value):
        if isinstance(value, str):
            try:
                return map_dtype[value]
            except KeyError:
                raise ValueError(
                    f"Invalid value '{value}', expected one of {list(map_dtype.keys())}"
                )
        elif value in map_dtype.values():
            return value
        else:
            raise ValueError(
                f"Invalid value '{value}', expected one of {list(map_dtype.values())}"
            )

    class _MapTypePydanticAnnotation:
        @classmethod
        def __get_pydantic_core_schema__(
            cls,
            _source_type: Any,
            _handler: GetCoreSchemaHandler,
        ) -> core_schema.CoreSchema:
            return core_schema.no_info_plain_validator_function(
                validate,
                serialization=core_schema.plain_serializer_function_ser_schema(
                    lambda v: reverse_map_dtype.get(v, str(v))
                ),
            )

        @classmethod
        def __get_pydantic_json_schema__(
            cls, _core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
        ) -> JsonSchemaValue:
            # Return a JSON schema that is a string enum of the keys in map_dtype
            return {"type": "string", "enum": list(map_dtype.keys())}
    return Annotated[Any, _MapTypePydanticAnnotation]