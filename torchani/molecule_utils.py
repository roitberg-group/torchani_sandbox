import torch
import math
from torchani.utils import ATOMIC_NUMBERS


def make_methane(device=None, eq_bond=1.09):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d = eq_bond * 2 / math.sqrt(3)
    coordinates = torch.tensor(
        [[[0.0, 0.0, 0.0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [0.5, 0.5, 0.5]]],
        device=device,
        dtype=torch.double) * d
    species = torch.tensor([[1, 1, 1, 1, 6]], device=device, dtype=torch.long)
    return species, coordinates.double()


def make_diatomic(device=None, atom1: str = 'H', atom2: str = 'H', eq_bond=1.0):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    coordinates = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.0, 0.0, 1]]],
        device=device,
        dtype=torch.double) * eq_bond
    species = torch.tensor([[ATOMIC_NUMBERS[atom1], ATOMIC_NUMBERS[atom2]]], device=device, dtype=torch.long)
    return species, coordinates.double()


def make_carbon_monoxide(device=None, eq_bond=1.13):
    return make_diatomic(device=device, atom1='C', atom2='O', eq_bond=eq_bond)


def make_angular(device=None, atom1: str = 'H', atom2: str = 'H', atom3: str = 'H', eq_bond=1.0, eq_angle=109.5):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d = eq_bond
    t = (math.pi / 180) * eq_angle  # convert to radians
    coordinates = torch.tensor(
        [[[d, 0, 0], [d * math.cos(t), d * math.sin(t), 0], [0, 0, 0]]],
        device=device).double()
    species = torch.tensor([[ATOMIC_NUMBERS[atom1], ATOMIC_NUMBERS[atom3], ATOMIC_NUMBERS[atom2]]], device=device, dtype=torch.long)
    return species, coordinates.double()


def make_ammonia(device=None, eq_bond=1.008):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d = eq_bond * 2 / math.sqrt(3)
    coordinates = torch.tensor(
        [[[0.0, 0.0, 0.0], [0, 1, 1], [1, 0, 1], [0.5, 0.5, 0.5]]],
        device=device, dtype=torch.double) * d
    species = torch.tensor([[1, 1, 1, 7]], device=device, dtype=torch.long)
    return species, coordinates.double()


def make_water(device=None, eq_bond=0.957582, eq_angle=104.485):
    return make_angular(device=device, atom1='H', atom2='O', atom3='H', eq_bond=eq_bond, eq_angle=eq_angle)
