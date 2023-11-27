r"""Factory methods that create atomic networks of different kinds"""
from copy import deepcopy
from typing import Sequence, Optional

import torch
from torch.nn import Module


def standard(dims: Sequence[int],
             activation: Optional[Module] = None,
             bias: bool = True,
             classifier_out: int = 1):
    r"""Makes a standard ANI style atomic network"""
    if activation is None:
        activation = torch.nn.GELU()
    else:
        activation = activation

    dims = list(deepcopy(dims))
    layers = []
    for dim_in, dim_out in zip(dims[:-1], dims[1:]):
        layers.extend([torch.nn.Linear(dim_in, dim_out, bias=bias), activation])
    # final layer is a linear classifier that is always appended
    layers.append(torch.nn.Linear(dims[-1], classifier_out, bias=bias))

    assert len(layers) == (len(dims) - 1) * 2 + 1
    return torch.nn.Sequential(*layers)


def like_1x(atom: str = 'H', 
            aev_dim: int = 384,
            **kwargs):
    r"""Makes a sequential atomic network like the one used in the ANI-1x model"""
    dims_for_atoms = {'H': (aev_dim, 160, 128, 96),
                      'C': (aev_dim, 144, 112, 96),
                      'N': (aev_dim, 128, 112, 96),
                      'O': (aev_dim, 128, 112, 96)}
    return standard(dims_for_atoms[atom], **kwargs)


def like_1ccx(atom: str = 'H', 
            aev_dim: int = 384,
            **kwargs):
    r"""Makes a sequential atomic network like the one used in the ANI-1ccx model"""
    return like_1x(atom=atom, aev_dim=aev_dim, **kwargs)


def like_2x(atom: str = 'H', 
            aev_dim: int = 1008,
            **kwargs):
    r"""Makes a sequential atomic network like the one used in the ANI-2x model"""
    dims_for_atoms = {'H': (aev_dim, 256, 192, 160),
                      'C': (aev_dim, 224, 192, 160),
                      'N': (aev_dim, 192, 160, 128),
                      'O': (aev_dim, 192, 160, 128),
                      'S': (aev_dim, 160, 128, 96),
                      'F': (aev_dim, 160, 128, 96),
                      'Cl': (aev_dim, 160, 128, 96)}
    return standard(dims_for_atoms[atom], **kwargs)
