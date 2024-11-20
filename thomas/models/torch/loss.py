from typing import Union, Literal

from torch import nn

Loss = Union[
    Literal['CrossEntropyLoss'], 
    Literal['MSELoss']
]

map_loss = {
    'CrossEntropyLoss': nn.CrossEntropyLoss,
    'MSELoss': nn.MSELoss
}