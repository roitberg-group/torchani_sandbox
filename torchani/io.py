import torch
import warnings
from torch import Tensor
from .utils import ATOMIC_NUMBERS, PERIODIC_TABLE
from typing import Tuple, Union, Optional
from pathlib import Path


def tensor_from_xyz(path: Union[str, Path]):
    with open(path, 'r') as f:
        lines = f.readlines()
        num_atoms = int(lines[0])
        coordinates = []
        species = []
        _, _, a, b, c = lines[1].split()
        cell = torch.diag(torch.tensor([float(a), float(b), float(c)]))
        for line in lines[2:]:
            values = line.split()
            if values:
                s = values[0].strip()
                x = float(values[1])
                y = float(values[2])
                z = float(values[3])
                coordinates.append([x, y, z])
                species.append(ATOMIC_NUMBERS[s])
        coordinates = torch.tensor(coordinates)
        species = torch.tensor(species, dtype=torch.long)
        assert coordinates.shape[0] == num_atoms
        assert species.shape[0] == num_atoms
    return species, coordinates, cell


def tensor_to_xyz(path: Union[str, Path],
                  species_coordinates: Tuple[Tensor, Tensor],
                  cell: Optional[Tensor] = None,
                  no_exponent: bool = True):
    # input species must be atomic numbers
    species, coordinates = species_coordinates
    num_atoms = species.shape[1]
    assert coordinates.shape[0] == 1, "Batch printing not implemented"
    assert species.shape[0] == 1, "Batch printing not implemented"
    coordinates = coordinates.view(-1, 3)
    species = species.view(-1)

    with open(path, 'w') as f:
        f.write(f'{num_atoms}\n')
        if cell is not None:
            warnings.warn("Cell printing is not yet implemented, ignoring cell")
        f.write('\n')
        for s, c in zip(species, coordinates):
            if no_exponent:
                line = f"{c[0]:.15f} {c[1]:.15f} {c[2]:.15f}\n"
            else:
                line = f"{c[0]} {c[1]} {c[2]}\n"
            line = f"{PERIODIC_TABLE[s]} " + line
            f.write(line)
