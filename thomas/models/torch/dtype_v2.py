from typing import Any, Dict, Type, Annotated
from pydantic import BaseModel, GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic_core import core_schema
from pydantic.json_schema import JsonSchemaValue
import torch

def create_mapped_type(map_dtype: Dict[str, Any], dtype_type: Type[Any]):
    reverse_map_dtype = {v: k for k, v in map_dtype.items()}

    class _MapTypePydanticAnnotation:
        @classmethod
        def __get_pydantic_core_schema__(
            cls,
            _source_type: Any,
            _handler: GetCoreSchemaHandler,
        ) -> core_schema.CoreSchema:
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

# Define the mapping for torch
torch_map_dtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

torch_map_loss = {
    "CrossEntropyLoss": torch.nn.CrossEntropyLoss,
    "MSELoss": torch.nn.MSELoss,
}

torch_map_optimizer = {
    "SGD": torch.optim.SGD,
    "Adam": torch.optim.Adam,
}

# Create the Annotated types
TorchDType = create_mapped_type(torch_map_dtype, torch.dtype)
TorchLoss = create_mapped_type(torch_map_loss, torch.nn.modules.loss._Loss)
TorchOptimizer = create_mapped_type(torch_map_optimizer, torch.optim.Optimizer)

# Define the Pydantic model
class Model(BaseModel):
    dtype: TorchDType
    loss: TorchLoss
    optimizer: TorchOptimizer

# Test the model with string inputs
model = Model(dtype="float32", loss="CrossEntropyLoss", optimizer="SGD")
print(model)
print(model.dtype)
print(model.loss)
print(model.optimizer)
print(model.model_dump())

# Test the model with actual torch types
model = Model(dtype=torch.float32, loss=torch.nn.CrossEntropyLoss, optimizer=torch.optim.SGD)
print(model)
print(model.model_dump())

# Test invalid input
try:
    model = Model(dtype="invalid_dtype", loss="CrossEntropyLoss", optimizer="SGD")
except Exception as e:
    print(e)

# # Output the JSON schema
# print(model.model_json_schema())
