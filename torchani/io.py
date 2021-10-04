import torch
from torch import Tensor

import warnings
from typing import Tuple, Optional
from pathlib import Path

from torchani.utils import PERIODIC_TABLE


def tensor_from_xyz(path):
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
                species.append(PERIODIC_TABLE.index(s))
        coordinates = torch.tensor(coordinates)
        species = torch.tensor(species, dtype=torch.long)
        assert coordinates.shape[0] == num_atoms
        assert species.shape[0] == num_atoms
    return species, coordinates, cell


def tensor_to_xyz(path, species_coordinates: Tuple[Tensor, Tensor],
                        cell: Optional[Tensor] = None,
                        no_exponent: bool = True,
                        comment: str = '',
                        append=False, truncate_output_file=False):
    r"""Dump a tensor as an xyz file"""
    path = Path(path).resolve()
    # input species must be atomic numbers
    species, coordinates = species_coordinates
    num_atoms = species.shape[1]

    assert coordinates.dim() == 3, "bad number of dimensions for coordinates"
    assert species.dim() == 2, "bad number of dimensions for species"
    assert coordinates.shape[0] == 1, "Batch printing not implemented"
    assert species.shape[0] == 1, "Batch printing not implemented"

    coordinates = coordinates.view(-1, 3)
    species = species.view(-1)
    if append:
        mode = 'a'
    else:
        if truncate_output_file:
            mode = 'w'
        else:
            mode = 'x'
    with open(path, mode) as f:
        f.write(f'{num_atoms}\n')
        if cell is not None:
            warnings.warn("Cell printing is not yet implemented, ignoring cell")
        f.write(f'{comment}\n')
        for s, c in zip(species, coordinates):
            if no_exponent:
                line = f"{c[0]:.15f} {c[1]:.15f} {c[2]:.15f}\n"
            else:
                line = f"{c[0]} {c[1]} {c[2]}\n"
            line = f"{PERIODIC_TABLE[s]} " + line
            f.write(line)


def tensor_to_lammpstrj(path, species_coordinates: Tuple[Tensor, Tensor],
                        cell: Tensor,
                        forces: Optional[Tensor] = None,
                        velocities: Optional[Tensor] = None,
                        charges: Optional[Tensor] = None,
                        no_exponent: bool = True,
                        append=False,
                        frame=0,
                        scale=False, truncate_output_file=False):
    r"""Dump a tensor as a lammpstrj file

    Dumps a species_coordinates tuple into a lammpstrj format file, optionally also
    dumps forces, velocities and charges. Currently the simulation cell MUST be provided
    if "append" is true then the tensor is appended to an existing file, otherwise
    it is written to a new file.
    """
    path = Path(path).resolve()
    # input species must be atomic numbers
    species, coordinates = species_coordinates
    num_atoms = species.shape[1]
    cell_diag = torch.diag(cell)

    assert coordinates.dim() == 3, "bad number of dimensions for coordinates"
    assert species.dim() == 2, "bad number of dimensions for species"
    assert coordinates.shape[0] == 1, "Batch printing not implemented"
    assert species.shape[0] == 1, "Batch printing not implemented"

    coordinates = coordinates.view(-1, 3)
    species = species.view(-1)
    if forces is not None:
        assert forces.dim() == 3
        assert forces.shape[0] == 1, "Batch printing not implemented"
        forces = forces.view(-1, 3)
    if velocities is not None:
        assert velocities.dim() == 3
        assert velocities.shape[0] == 1, "Batch printing not implemented"
        velocities = velocities.view(-1, 3)
    if charges is not None:
        assert charges.dim() == 3
        assert charges.shape[0] == 1, "Batch printing not implemented"
        charges = charges.view(-1, 3)
    if append:
        mode = 'a'
    else:
        if truncate_output_file:
            mode = 'w'
        else:
            mode = 'x'
    with open(path, mode) as f:
        f.write(f'ITEM: TIMESTEP\n')
        f.write(f'{frame}\n')
        f.write(f'ITEM: NUMBER OF ATOMS\n')
        f.write(f'{num_atoms}\n')
        f.write(f'ITEM: BOX BOUNDS xx yy zz\n')
        f.write(f'0.0 {cell_diag[0]}\n')
        f.write(f'0.0 {cell_diag[1]}\n')
        f.write(f'0.0 {cell_diag[2]}\n')
        # postfix u means the coordinates are unwrapped
        if scale:
            coordinates = torch.frac(coordinates / cell_diag)
            line = f'ITEM: ATOMS id type xs ys zs'
        else:
            line = f'ITEM: ATOMS id type xu yu zu'
        if forces is not None:
            line += ' fx fy fz'
        if velocities is not None:
            line += ' vx vy vz'
        if charges is not None:
            line += ' q'
        f.write(line + '\n')
        for j, (s, c) in enumerate(zip(species, coordinates)):
            if no_exponent:
                line = f"{c[0]:.15f} {c[1]:.15f} {c[2]:.15f}"
            else:
                line = f"{c[0]} {c[1]} {c[2]}"
            line = f"{j} {s} " + line
            if forces is not None:
                force = forces[j]
                line += f" {force[0]} {force[1]} {force[2]}"
            if velocities is not None:
                v = velocities[j]
                line += f" {v[0]} {v[1]} {v[2]}"
            if charges is not None:
                line += f" {charges[j]}"
            f.write(line + '\n')
