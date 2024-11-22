from typing import Annotated, Dict, Any, Type
import pydantic
import torch

def create_map_type(map_dtype: Dict[str, Any], dtype_type: Type[Any]):
    def type_validator(dtype):
        if isinstance(dtype, str):
            try:
                return map_dtype[dtype]
            except KeyError:
                raise ValueError(f"Invalid dtype '{dtype}', expected one of {list(map_dtype.keys())}")
        if isinstance(dtype, dtype_type) or (isinstance(dtype, type) and issubclass(dtype, dtype_type)):
            if dtype in map_dtype.values():
                return dtype
            else:
                raise ValueError(f"Invalid dtype '{dtype}', expected one of {list(map_dtype.values())}")
        raise ValueError(f"Invalid dtype '{dtype}', expected `str` or `{dtype_type.__name__}`")

    def type_serializer(dtype):
        for k, v in map_dtype.items():
            if v == dtype:
                return k
        raise ValueError(f"Invalid dtype '{dtype}', expected one of {list(map_dtype.values())}")


    return Annotated[
        Any,
        pydantic.BeforeValidator(type_validator),
        pydantic.PlainSerializer(type_serializer)
    ]

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

# Create the Annotated type
TorchDType = create_map_type(torch_map_dtype, torch.dtype)
TorchLoss = create_map_type(torch_map_loss, torch.nn.modules.loss._Loss)
TorchOptimizer = create_map_type(torch_map_optimizer, torch.optim.Optimizer)

# Define the Pydantic model
class Model(pydantic.BaseModel):
    dtype: TorchDType
    loss: TorchLoss
    optimizer: TorchOptimizer

    class Config:
        arbitrary_types_allowed = True


model = Model(dtype="float32", loss="CrossEntropyLoss", optimizer="SGD")
print(model)
print(model.dtype)
print(model.loss)
print(model.optimizer)
print(model.model_dump())