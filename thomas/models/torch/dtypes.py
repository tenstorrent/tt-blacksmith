from typing import Union, Literal
import torch

dtypes = Union[
    Literal['float32'],
    Literal['bfloat16'],
]

map_dtype = {
    'float32': torch.float32,
    'bfloat16': torch.bfloat16,
}