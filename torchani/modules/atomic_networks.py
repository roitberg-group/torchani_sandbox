"""Factory methods that create atomic networks of different kinds"""
import torch
import copy

def make_classic(dim_in, dims, activation=None, bias=True):
    """Makes a classic ANI style atomic network"""
    # automatically insert the first dimension
    if activation is None:
        activation = torch.nn.CELU(0.1)
    else:
        activation = activation
    dims_ = list(copy.deepcopy(dims))
    dims_.insert(0, dim_in)
    dimensions = range(len(dims_) - 1)
    layers = []
    for j in dimensions:
        block = [torch.nn.Linear(dims_[j], dims_[j + 1], bias=bias), activation]
        layers.extend(block)
    # final layer is always appended
    layers.append(torch.nn.Linear(dims_[-1], 1, bias=bias))
    assert len(layers) == (len(dims_)-1)*2 + 1
    return torch.nn.Sequential(*layers)

def make_like_1x(atom='H', activation=None, bias=True):
    args_for_atoms = {
        'H': {
            'dim_in': 384,
            'dims': (160, 128, 96)
        },
        'C': {
            'dim_in': 384,
            'dims': (144, 112, 96)
        },
        'N': {
            'dim_in': 384,
            'dims': (128, 112, 96)
        },
        'O': {
            'dim_in': 384,
            'dims': (128, 112, 96)
        },
    }
    return make_classic(**args_for_atoms[atom], activation=activation, bias=bias)

def make_like_2x(atom='H', activation=None, bias=True):
    args_for_atoms = {
        'H': {
            'dim_in': 1008,
            'dims': (256, 192, 160)
        },
        'C': {
            'dim_in': 1008,
            'dims': (224, 192, 160)
        },
        'N': {
            'dim_in': 1008,
            'dims': (192, 160, 128)
        },
        'O': {
            'dim_in': 1008,
            'dims': (192, 160, 128)
        },
        'S': {
            'dim_in': 1008,
            'dims': (160, 128, 96)
        },
        'F': {
            'dim_in': 1008,
            'dims': (160, 128, 96)
        },
        'Cl': {
            'dim_in': 1008,
            'dims': (160, 128, 96)
        },
    }
    return make_classic(**args_for_atoms[atom], activation=activation, bias=bias)

def make_like_1ccx(atom='H', activation=None, bias=True):
    return make_like_1x(atom)
