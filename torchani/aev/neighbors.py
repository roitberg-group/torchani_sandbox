import torch
import math
from torch import Tensor
from typing import Tuple
from ..compat import Final


class BaseNeighborlist(torch.nn.Module):

    cutoff: Final[float]

    def __init__(self, cutoff: float):
        """Compute pairs of atoms that are neighbors, uses pbc depending on
        weather pbc.any() is True or not

        Arguments:
            coordinates (:class:`torch.Tensor`): tensor of shape
                (molecules, atoms, 3) for atom coordinates.
            cutoff (float): the cutoff inside which atoms are considered pairs
        """
        super().__init__()
        self.cutoff = cutoff
        self.register_buffer('default_cell', torch.eye(3, dtype=torch.float))

    @torch.jit.export
    def _compute_bounding_cell(self, coordinates: Tensor,
                               eps: float) -> Tuple[Tensor, Tensor]:
        # this works but its not needed for this naive implementation
        # This should return a bounding cell
        # for the molecule, in all cases, also it displaces coordinates a fixed
        # value, so that they fit inside the cell completely. This should have
        # no effects on forces or energies

        # add an epsilon to pad due to floating point precision
        min_ = torch.min(coordinates.view(-1, 3), dim=0)[0] - eps
        max_ = torch.max(coordinates.view(-1, 3), dim=0)[0] + eps
        largest_dist = max_ - min_
        coordinates = coordinates - min_
        cell = self.default_cell * largest_dist
        assert (coordinates > 0.0).all()
        assert (coordinates < torch.norm(cell, dim=1)).all()
        return coordinates, cell

    @staticmethod
    def _screen_with_cutoff(cutoff: float, coordinates: Tensor, input_neighborlist: Tensor,
            shift_values: Tensor) -> Tuple[Tensor, Tensor]:
        # screen a given neighborlist using a cutoff and return a neighborlist with
        # atoms that are within that cutoff, for all molecules in a coordinate set
        # if the initial coordinates have more than one molecule in the batch dimension
        # then the output neighborlist will correctly index flattened coordinates
        # obtained via coordinates.view(-1, 3). If the initial coordinates
        # have only one molecule then the output neighborlist will index non
        # flattened coordinates correctly
        num_mols = coordinates.shape[0]
        num_atoms = coordinates.shape[1]
        selected_coordinates = coordinates.index_select(1, input_neighborlist.view(-1))
        selected_coordinates = selected_coordinates.view(num_mols, 2, -1, 3)
        distances_sq = (selected_coordinates[:, 0, ...] - selected_coordinates[:, 1, ...] + shift_values).pow(2).sum(-1)
        in_cutoff = (distances_sq <= cutoff ** 2).nonzero()
        molecule_index, pair_index = in_cutoff.unbind(1)
        screened_neighborlist = input_neighborlist.index_select(1, pair_index) + molecule_index * num_atoms
        screened_shift_values = shift_values.index_select(0, pair_index)

        return screened_neighborlist, screened_shift_values


class FullPairwise(BaseNeighborlist):

    def __init__(self, cutoff: float):
        """Compute pairs of atoms that are neighbors, uses pbc depending on
        weather pbc.any() is True or not

        Arguments:
            cutoff (float): the cutoff inside which atoms are considered pairs
        """
        super().__init__(cutoff)
        self.register_buffer('default_shift_values', torch.tensor(0.0))

    def forward(self, species: Tensor, coordinates: Tensor, cell: Tensor,
                pbc: Tensor) -> Tuple[Tensor, Tensor]:
        """Arguments:
            coordinates (:class:`torch.Tensor`): tensor of shape
                (molecules, atoms, 3) for atom coordinates.
            cell (:class:`torch.Tensor`): tensor of shape (3, 3) of the three vectors
                defining unit cell: tensor([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])
            cutoff (float): the cutoff inside which atoms are considered pairs
            pbc (:class:`torch.Tensor`): boolean tensor of shape (3,) storing wheather pbc is required
        """
        if pbc.any():
            atom_index12, shift_indices = self._full_pairwise_pbc(species, cell, pbc)
            shift_values = shift_indices.to(cell.dtype) @ cell
        else:
            atom_index12 = torch.triu_indices(species.shape[1], species.shape[1], 1, device=species.device)
            # create dummy shift values that are zero
            shift_values = self.default_shift_values.repeat(atom_index12.shape[1], 3)

        coordinates = coordinates.detach().masked_fill((species == -1).unsqueeze(-1), math.nan)

        atom_index12, shift_values = self._screen_with_cutoff(self.cutoff, coordinates, atom_index12, shift_values)

        return atom_index12, shift_values

    def _full_pairwise_pbc(self, species: Tensor,
                           cell: Tensor, pbc: Tensor) -> Tuple[Tensor, Tensor]:
        cell = cell.detach()
        shifts = self._compute_shifts(cell, pbc)
        num_atoms = species.shape[1]
        all_atoms = torch.arange(num_atoms, device=cell.device)

        # Step 2: center cell
        # torch.triu_indices is faster than combinations
        p12_center = torch.triu_indices(num_atoms,
                                        num_atoms,
                                        1,
                                        device=cell.device)
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
        all_atom_pairs = torch.cat([p12_center, p12], dim=1)
        return all_atom_pairs, shifts_all

    def _compute_shifts(self, cell: Tensor, pbc: Tensor) -> Tensor:
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
