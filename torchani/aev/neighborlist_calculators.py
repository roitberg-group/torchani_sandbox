import torch
import math
from torch import Tensor
import sys
from typing import Optional, Tuple
from .cutoffs import CutoffCosine, CutoffSmooth

if sys.version_info[:2] < (3, 7):
    class FakeFinal:
        def __getitem__(self, x):
            return x
    Final = FakeFinal()
else:
    from torch.jit import Final


class FullPairwise(torch.nn.Module):

    cutoff: Final[float]

    def __init__(self, cutoff : float):
        """Compute pairs of atoms that are neighbors (doesn't use PBC)
    
        Arguments:
            padding_mask (:class:`torch.Tensor`): boolean tensor of shape
                (molecules, atoms) for padding mask. 1 == is padding.
            coordinates (:class:`torch.Tensor`): tensor of shape
                (molecules, atoms, 3) for atom coordinates.
            cutoff (float): the cutoff inside which atoms are considered pairs
        """
        super().__init__()
        self.cutoff = cutoff
        # not needed by this simple implementation
        self.register_buffer('default_cell', torch.eye(3, dtype=torch.float))
    
    def forward(self, species: Tensor, coordinates: Tensor, cell: Tensor, pbc: Tensor) -> Tuple[Tensor, Tensor]:
        # cell and pbc are unused
        assert not pbc.any()
        padding_mask = species == -1
        coordinates = coordinates.detach().masked_fill(padding_mask.unsqueeze(-1), math.nan)
        current_device = coordinates.device
        num_atoms = padding_mask.shape[1]
        num_mols = padding_mask.shape[0]
        p12_all = torch.triu_indices(num_atoms, num_atoms, 1, device=current_device)
        p12_all_flattened = p12_all.view(-1)
    
        pair_coordinates = coordinates.index_select(1, p12_all_flattened).view(num_mols, 2, -1, 3)
        distances = (pair_coordinates[:, 0, ...] - pair_coordinates[:, 1, ...]).norm(2, -1)
        in_cutoff = (distances <= self.cutoff).nonzero()
        molecule_index, pair_index = in_cutoff.unbind(1)
        molecule_index *= num_atoms
        atom_index12 = p12_all[:, pair_index] + molecule_index
        return atom_index12, torch.zeros(3, dtype=coordinates.dtype, device=coordinates.device)

    @torch.jit.export
    def _compute_bounding_cell(self, coordinates : Tensor, eps : float) -> Tuple[Tensor, Tensor]:
        # this works but its not needed for this naive implementation
        # This should return a bounding cell
        # for the molecule, in all cases, also it displaces coordinates a fixed
        # value, so that they fit inside the cell completely. This should have
        # no effects on forces or energies

        # add an epsilon to pad due to floating point precision
        min_ = torch.min(coordinates.view(-1, 3), dim=0)[0] - eps
        max_ = torch.max(coordinates.view(-1, 3), dim=0)[0] + eps
        largest_dist = max_ - min_ 
        coordinates =  coordinates - min_  
        cell = self.default_cell * largest_dist
        assert (coordinates > 0.0).all()
        assert (coordinates < torch.norm(cell, dim=1)).all()
        return coordinates, cell


class FullPairwisePBC(torch.nn.Module):

    cutoff: Final[float]

    def __init__(self, cutoff : float):
        """Compute pairs of atoms that are neighbors (doesn't use PBC)
    
        Arguments:
            padding_mask (:class:`torch.Tensor`): boolean tensor of shape
                (molecules, atoms) for padding mask. 1 == is padding.
            coordinates (:class:`torch.Tensor`): tensor of shape
                (molecules, atoms, 3) for atom coordinates.
            cutoff (float): the cutoff inside which atoms are considered pairs
        """
        super().__init__()
        self.cutoff = cutoff

    def forward(self, species: Tensor, coordinates: Tensor, cell: Tensor, pbc: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute pairs of atoms that are neighbors, 

        Arguments:
            padding_mask (:class:`torch.Tensor`): boolean tensor of shape
                (molecules, atoms) for padding mask. 1 == is padding.
            coordinates (:class:`torch.Tensor`): tensor of shape
                (molecules, atoms, 3) for atom coordinates.
            cell (:class:`torch.Tensor`): tensor of shape (3, 3) of the three vectors
                defining unit cell: tensor([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])
            cutoff (float): the cutoff inside which atoms are considered pairs
            pbc (:class:`torch.Tensor`): boolean tensor of shape (3,) storing wheather pbc is required

        """
        assert pbc.any()
        shifts = self.compute_shifts(cell, pbc)
        padding_mask = (species == -1)
        coordinates = coordinates.detach().masked_fill(padding_mask.unsqueeze(-1), math.nan)
        cell = cell.detach()
        num_atoms = padding_mask.shape[1]
        num_mols = padding_mask.shape[0]
        all_atoms = torch.arange(num_atoms, device=cell.device)

        # Step 2: center cell
        # torch.triu_indices is faster than combinations
        p12_center = torch.triu_indices(num_atoms, num_atoms, 1, device=cell.device)
        shifts_center = shifts.new_zeros((p12_center.shape[1], 3))

        # Step 3: cells with shifts
        # shape convention (shift index, molecule index, atom index, 3)
        num_shifts = shifts.shape[0]
        all_shifts = torch.arange(num_shifts, device=cell.device)
        prod = torch.cartesian_prod(all_shifts, all_atoms, all_atoms).t()
        shift_index = prod[0]
        p12 = prod[1:]
        shifts_outside = shifts.index_select(0, shift_index)

        # Step 4: combine results for all cells
        shifts_all = torch.cat([shifts_center, shifts_outside])
        p12_all = torch.cat([p12_center, p12], dim=1)
        shift_values = shifts_all.to(cell.dtype) @ cell

        # step 5, compute distances, and find all pairs within cutoff
        selected_coordinates = coordinates.index_select(1, p12_all.view(-1)).view(num_mols, 2, -1, 3)
        distances = (selected_coordinates[:, 0, ...] - selected_coordinates[:, 1, ...] + shift_values).norm(2, -1)
        in_cutoff = (distances <= self.cutoff).nonzero()
        molecule_index, pair_index = in_cutoff.unbind(1)
        molecule_index *= num_atoms
        atom_index12 = p12_all[:, pair_index]
        shifts = shifts_all.index_select(0, pair_index)

        return molecule_index + atom_index12, shifts

    def compute_shifts(self, cell: Tensor, pbc: Tensor) -> Tensor:
        """Compute the shifts of unit cell along the given cell vectors to make it
        large enough to contain all pairs of neighbor atoms with PBC under
        consideration

        Arguments:
            cell (:class:`torch.Tensor`): tensor of shape (3, 3) of the three
            vectors defining unit cell:
                tensor([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])
            pbc (:class:`torch.Tensor`): boolean vector of size 3 storing
                if pbc is enabled for that direction.

        Returns:
            :class:`torch.Tensor`: long tensor of shifts. the center cell and
                symmetric cells are not included.
        """
        reciprocal_cell = cell.inverse().t()
        inv_distances = reciprocal_cell.norm(2, -1)
        num_repeats = torch.ceil(self.cutoff * inv_distances).to(torch.long)
        num_repeats = torch.where(pbc, num_repeats, num_repeats.new_zeros(()))
        r1 = torch.arange(1, num_repeats[0].item() + 1, device=cell.device)
        r2 = torch.arange(1, num_repeats[1].item() + 1, device=cell.device)
        r3 = torch.arange(1, num_repeats[2].item() + 1, device=cell.device)
        o = torch.zeros(1, dtype=torch.long, device=cell.device)
        return torch.cat([
            torch.cartesian_prod(r1, r2, r3),
            torch.cartesian_prod(r1, r2, o),
            torch.cartesian_prod(r1, r2, -r3),
            torch.cartesian_prod(r1, o, r3),
            torch.cartesian_prod(r1, o, o),
            torch.cartesian_prod(r1, o, -r3),
            torch.cartesian_prod(r1, -r2, r3),
            torch.cartesian_prod(r1, -r2, o),
            torch.cartesian_prod(r1, -r2, -r3),
            torch.cartesian_prod(o, r2, r3),
            torch.cartesian_prod(o, r2, o),
            torch.cartesian_prod(o, r2, -r3),
            torch.cartesian_prod(o, o, r3),
        ])
