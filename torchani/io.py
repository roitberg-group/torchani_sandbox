r"""Simple IO module for basic parsing of common plaintext trajectory formats"""
import torch
from torch import Tensor
import itertools
from typing import Sequence

import warnings
from typing import Optional
from pathlib import Path
from torchani.utils import tqdm

from torchani.utils import PERIODIC_TABLE, ATOMIC_NUMBERS
try:
    import ase
    _ASE_AVAIL = True
except ImportError:
    _ASE_AVAIL = False


def _advance(f, num):
    if num > 0:
        for j in range(num):
            f.readline()


def tensor_from_asetraj(path, start_frame=0, end_frame=None, step=1, get_cell=True):
    traj_file = Path(path).resolve()
    traj = ase.io.Trajectory(traj_file.as_posix())
    if end_frame is None:
        end_frame = len(traj)
    coordinates = []
    species = []
    cell = []
    for j, a in enumerate(tqdm(traj[start_frame:end_frame:step])):
        coordinates.append(torch.from_numpy(a.positions))
        species.append(torch.from_numpy(a.numbers).long())
        if get_cell:
            cell.append(torch.as_tensor(a.cell))
    return torch.stack(coordinates), torch.stack(species), torch.stack(cell)


def tensor_to_xyz(path, species_coordinates, **kwargs):
    return _conformations_to_file(path, species_coordinates, dumper='xyz', **kwargs)


def tensor_to_lammpstrj(path, species_coordinates, **kwargs):
    return _conformations_to_file(path, species_coordinates, dumper='lammpstrj', **kwargs)


def _conformations_to_file(path, species_coordinates, dumper, frame_range=None, **kwargs):
    species, coordinates = species_coordinates
    tensors = {'species': species, 'coordinates': coordinates}
    num_molecules = species.shape[0] if species.dim() == 2 else 1
    if dumper == 'xyz':
        dumper = _dump_xyz
    elif dumper == 'lammpstrj':
        dumper = _dump_lammpstrj
    if frame_range is None:
        frame_range = itertools.count(0)
    else:
        assert len(frame_range) == num_molecules

    supported_tensors = ['species', 'coordinates', 'forces', 'velocities', 'charges', 'cell']
    nonbatch_dims = [1, 2, 2, 2, 1, 2]

    # unsqueeze necessary dimensions
    for key, nonbatch_dim in zip(supported_tensors, nonbatch_dims):
        tensor_ = kwargs.pop(key, None)
        if tensor_ is not None:
            if tensor_.dim() == nonbatch_dim:
                tensors[key] = tensor_.unsqueeze(0)
            else:
                tensors[key] = tensor_
            assert tensor_.dim() == nonbatch_dim + 1, f"Bad number of dimensions for {key}"
            assert tensor_.shape[0] == num_molecules, f"Bad number of molecules for {key}"
    for frame, j in zip(range(num_molecules), frame_range):
        append = kwargs.pop('append', (j != 0))
        kwargs.update({name: t[j] for name, t in tensors.items()})
        dumper(path, append=append, frame=frame, **kwargs)


def _get_index(header):
    if 'x' in header:
        return 0
    elif 'y' in header:
        return 1
    elif 'z' in header:
        return 2


def tensor_from_lammpstrj(path, start_frame=0, end_frame=None, step=1,
                          get_cell=True,
                          get_forces=False,
                          get_velocities=False,
                          coordinates_type='unscaled',
                          extract_atoms: Sequence[int] = None):
    r"""Reads a batch of conformations from an lammpstrj file

    extract_atoms is a sequence of atom indices to extract
    """
    with open(path, 'r') as f:
        _advance(f, 3)
        num_atoms = int(f.readline())
        f.seek(0)
        frame_size = num_atoms + 9  # 4 headers + 3 box bounds + ts + num_atoms
        iterable = itertools.count(start_frame * frame_size, step * frame_size)
        cell = []
        coordinates = []
        species = []
        velocities = []
        forces = []
        for line_num in iterable:
            if end_frame is not None and line_num == end_frame * frame_size:
                break
            _advance(f, 5)
            try:
                xlo, xhi = [float(v) for v in f.readline().split()]
            except ValueError:
                break
            ylo, yhi = [float(v) for v in f.readline().split()]
            zlo, zhi = [float(v) for v in f.readline().split()]
            if get_cell:
                cell_diag = [xhi - xlo, yhi - ylo, zhi - zlo]
                cell.append(torch.diag(torch.tensor(cell_diag)))
            header = f.readline().split()[2:]
            species_ = []
            coordinates_ = []
            forces_ = []
            velocities_ = []
            for j in range(num_atoms):
                atom_coords = [None, None, None]
                atom_forces = [None, None, None]
                atom_velocities = [None, None, None]
                if extract_atoms is not None and j not in extract_atoms:
                    f.readline()
                    continue
                line = f.readline().split()
                for h, v in zip(header, line):
                    if h == 'type':
                        species_.append(int(v))
                    elif 'u' in h:
                        if coordinates_type == 'unscaled':
                            atom_coords[_get_index(h)] = float(v)
                        elif coordinates_type == 'scaled':
                            atom_coords[_get_index(h)] = float(v) / cell_diag[_get_index(h)]
                    elif 's' in h:
                        if coordinates_type == 'unscaled':
                            atom_coords[_get_index(h)] = float(v) * cell_diag[_get_index(h)]
                        elif coordinates_type == 'scaled':
                            atom_coords[_get_index(h)] = float(v)
                    elif 'f' in h:
                        atom_forces[_get_index(h)] = float(v)
                    elif 'v' in h:
                        atom_velocities[_get_index(h)] = float(v)
                if all(f is not None for f in atom_forces):
                    forces_.append(atom_forces)
                if all(v is not None for v in atom_velocities):
                    velocities_.append(atom_velocities)
                if all(v is not None for v in atom_velocities):
                    velocities_.append(atom_velocities)
                coordinates_.append(atom_coords)
            species.append(species_)
            coordinates.append(coordinates_)
            velocities.append(velocities_)
            forces.append(forces_)
            if step > 1:
                _advance(f, step * frame_size)
        species = torch.tensor(species, dtype=torch.long)
        coordinates = torch.tensor(coordinates, dtype=torch.float)
        forces = torch.tensor(forces, dtype=torch.float)
        velocities = torch.tensor(velocities, dtype=torch.float)
        if extract_atoms is None:
            assert coordinates.shape[1] == num_atoms
            assert species.shape[1] == num_atoms
        else:
            assert coordinates.shape[1] == len(extract_atoms)
            assert species.shape[1] == len(extract_atoms)
        if get_cell:
            cell = torch.stack(cell)
        else:
            cell = None
        return species, coordinates, cell, forces, velocities


def tensor_from_xyz(path, start_frame=0, end_frame=None, step=1, get_cell=True, extract_atoms: Sequence[int] = None):
    r"""Reads a batch of conformations from an xyz file or trajectory

    extract_atoms is a sequence of atom indices to extract
    """
    with open(path, 'r') as f:
        num_atoms = int(f.readline())
        f.seek(0)
        frame_size = num_atoms + 2
        if end_frame is not None:
            lines_ = sum(1 for line in open(path, 'r'))
            assert frame_size * end_frame <= lines_
            iterable = range(start_frame * frame_size, end_frame * frame_size, step * frame_size)
        else:
            iterable = itertools.count(start_frame * frame_size, step * frame_size)
        cell = []
        coordinates = []
        species = []
        for line_num in iterable:
            # advance "offset" lines
            num_ = f.readline()
            try:
                int(num_)
            except ValueError:
                break
            if get_cell:
                cell_diag = [float(v) for v in f.readline().split()[-3:]]
                cell.append(torch.diag(torch.tensor(cell_diag)))
            else:
                f.readline()
            species_ = []
            coordinates_ = []
            for j in range(num_atoms):
                if extract_atoms is not None and j not in extract_atoms:
                    f.readline()
                    continue
                line = f.readline().split()
                try:
                    species_.append(ATOMIC_NUMBERS[line[0].strip()])
                except KeyError:
                    species_.append(int(line[0].strip()))
                atom_coords = [float(v) for v in line[1:]]
                coordinates_.append(atom_coords)
            species.append(species_)
            coordinates.append(coordinates_)
            if step > 1:
                _advance(f, step * frame_size)
        species = torch.tensor(species, dtype=torch.long)
        coordinates = torch.tensor(coordinates, dtype=torch.float)
        if extract_atoms is None:
            assert coordinates.shape[1] == num_atoms
            assert species.shape[1] == num_atoms
        else:
            assert coordinates.shape[1] == len(extract_atoms)
            assert species.shape[1] == len(extract_atoms)
        if get_cell:
            cell = torch.cat(cell)
            return species, coordinates, cell
        return species, coordinates, None


def _dump_xyz(path,
              species: Tensor, coordinates: Tensor,
              cell: Optional[Tensor] = None,
              no_exponent: bool = True,
              comment: str = '',
              append=False, truncate_output_file=False, frame=0):
    r"""Dump a tensor as an xyz file"""
    path = Path(path).resolve()
    # input species must be atomic numbers
    num_atoms = len(species)
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


def _dump_lammpstrj(path, species: Tensor, coordinates: Tensor,
                        cell: Tensor,
                        forces: Optional[Tensor] = None,
                        velocities: Optional[Tensor] = None,
                        charges: Optional[Tensor] = None,
                        no_exponent: bool = True,
                        append=False,
                        truncate_output_file=False, frame=0, scale=False):
    r"""Dump a tensor as a lammpstrj file

    Dumps a species_coordinates tuple into a lammpstrj format file, optionally also
    dumps forces, velocities and charges. Currently the simulation cell MUST be provided.
    If "append" is true then the tensor is appended to an existing file, otherwise
    it is written to a new file.
    """
    path = Path(path).resolve()
    # input species must be atomic numbers
    num_atoms = len(species)
    cell_diag = torch.diag(cell)
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
