from typing import Union, Literal

from torch import optim

Optimizer = Union[
    Literal['SGD'], 
    Literal['Adam'],
    Literal['AdamW'],
]

map_optimizer = {
    'SGD': optim.SGD,
    'Adam': optim.Adam,
    'AdamW': optim.AdamW,
}