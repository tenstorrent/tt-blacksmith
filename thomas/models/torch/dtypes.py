from typing import Union, Literal
import torch

DType = Union[
    Literal['float32'],
    Literal['bfloat16'],
]

map_dtype = {
    'float32': torch.float32,
    'bfloat16': torch.bfloat16,
}