import math
import sys
from typing import Tuple, Optional

import torch
from torch import Tensor
from torch.nn import functional
from ..utils import map_to_central

if sys.version_info[:2] < (3, 7):

    class FakeFinal:
        def __getitem__(self, x):
            return x

    Final = FakeFinal()
else:
    from torch.jit import Final


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
    def _screen_with_cutoff(cutoff: float,
                            coordinates: Tensor,
                            input_neighborlist: Tensor,
                            shift_values: Tensor,
                            input_shifts: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
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
        distances_sq = (selected_coordinates[:, 0, ...]
                        - selected_coordinates[:, 1, ...]
                        + shift_values).pow(2).sum(-1)
        in_cutoff = (distances_sq <= cutoff ** 2).nonzero()
        molecule_index, pair_index = in_cutoff.unbind(1)
        atom_index12 = input_neighborlist.index_select(1, pair_index) + molecule_index * num_atoms

        if input_shifts is None:
            return atom_index12, torch.zeros(3, dtype=coordinates.dtype, device=coordinates.device)

        return atom_index12, input_shifts.index_select(0, pair_index)


class FullPairwise(BaseNeighborlist):
    def __init__(self, cutoff: float):
        """Compute pairs of atoms that are neighbors, uses pbc depending on
        weather pbc.any() is True or not

        Arguments:
            cutoff (float): the cutoff inside which atoms are considered pairs
        """
        super().__init__(cutoff)
        self.register_buffer('default_shift_values', torch.tensor(0.0))

    def forward(self, species: Tensor, 
                      coordinates: Tensor, 
                      cell: Tensor,
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
            atom_index12, shifts, shift_values = self._full_pairwise_pbc(
                species, cell, pbc)
            # before being screened the coordinates have to be mapped to the
            # central cell in case they are not inside it, this is not necessary
            # if there is no pbc
            coordinates = map_to_central(coordinates, cell, pbc)
        else:
            atom_index12, shifts, shift_values = self._full_pairwise(species)

        coordinates = coordinates.detach().masked_fill(
            (species == -1).unsqueeze(-1), math.nan)
        return self._screen_with_cutoff(self.cutoff, coordinates, atom_index12,
                                        shifts, shift_values)

    def _full_pairwise(self, species: Tensor) -> Tuple[Tensor, Tensor, None]:
        num_atoms = species.shape[1]
        all_atom_pairs = torch.triu_indices(num_atoms,
                                            num_atoms,
                                            1,
                                            device=species.device)
        return all_atom_pairs, self.default_shift_values, None

    def _full_pairwise_pbc(self, species: Tensor, cell: Tensor,
                           pbc: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        cell = cell.detach()
        shifts = self._compute_shifts(cell, pbc)
        num_atoms = species.shape[1]
        all_atoms = torch.arange(num_atoms, device=cell.device)

        # Step 2: center cell
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
        shift_values = shifts_all.to(cell.dtype) @ cell
        return all_atom_pairs, shift_values, shifts_all

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


class CellList(BaseNeighborlist):

    dynamic_update: Final[bool]
    constant_volume: Final[bool]

    def __init__(self,
                 cutoff,
                 buckets_per_cutoff=1,
                 dynamic_update=False,
                 skin=None,
                 constant_volume=False):
        super().__init__(cutoff)


        # right now I will only support this, and the extra neighbors are
        # hardcoded, but full support for arbitrary buckets per cutoff is possible
        assert buckets_per_cutoff == 1, "Cell list currently only supports one bucket per cutoff"
        self.constant_volume = constant_volume
        self.dynamic_update = dynamic_update
        self.register_buffer('cell_diagonal', torch.zeros(1))
        self.register_buffer('cell_inverse', torch.zeros(1))
        self.register_buffer('total_buckets', torch.zeros(1, dtype=torch.long))
        self.register_buffer('scaling_for_flat_index',
                             torch.zeros(1, dtype=torch.long))
        self.register_buffer('shape_buckets_grid',
                             torch.zeros(1, dtype=torch.long))
        self.register_buffer('vector_idx_to_flat',
                             torch.zeros(1, dtype=torch.long))
        self.register_buffer('translation_cases',
                             torch.zeros(1, dtype=torch.long))
        self.register_buffer('vector_index_displacement',
                             torch.zeros(1, dtype=torch.long))
        self.register_buffer('translation_displacement_indices',
                             torch.zeros(1, dtype=torch.long))
        self.register_buffer('translation_displacements', torch.zeros(1))
        self.register_buffer('bucket_length_lower_bound', torch.zeros(1))

        if skin is None:
            if dynamic_update:
                # default value for dynamically updated neighborlist
                skin = 1.0  
            else:
                # default value for non dynamically updated neighborlist
                skin = 0.0  
        self.register_buffer('skin', torch.tensor(skin))

        # only used for dynamic update
        self.register_buffer('old_cell_diagonal', torch.zeros(1))
        self.register_buffer('old_shift_indices', torch.zeros(1, dtype=torch.long))
        self.register_buffer('old_atom_pairs', torch.zeros(1, dtype=torch.long))
        self.register_buffer('old_coordinates', torch.zeros(1))

        # buckets_per_cutoff is also the number of buckets that is scanned in
        # each direction. It determines how fine grained the grid is, with
        # respect to the cutoff. This is 2 for amber, but 1 is useful for debug
        self.buckets_per_cutoff = buckets_per_cutoff
        # Here I get the vector index displacements for the neighbors of an
        # arbitrary vector index I think these are enough (this is different
        # from pmemd)
        # I choose all the displacements except for the zero
        # displacement that does nothing, which is the last one
        # hand written order,
        # this order is basically right-to-left, top-to-bottom
        # using the middle buckets (leftmost lower corner + rightmost lower bucket)
        # and the down buckets (all)
        # so this looks like:
        #  x--
        #  xo-
        #  xxx
        # For the middle buckets and
        #  xxx
        #  xxx
        #  xxx
        # for the down buckets
        vector_index_displacement = torch.tensor([[-1, 0, 0], [-1, -1, 0],
                                                  [0, -1, 0], [1, -1, 0],
                                                  [-1, 1, -1], [0, 1, -1],
                                                  [1, 1, -1], [-1, 0, -1],
                                                  [0, 0, -1], [1, 0, -1],
                                                  [-1, -1, -1], [0, -1, -1],
                                                  [1, -1, -1]], dtype=torch.long)
        self.vector_index_displacement = vector_index_displacement
        # these are the translation displacement indices, used to displace the
        # image atoms
        assert self.vector_index_displacement.shape == torch.Size([13, 3])

        # I need some extra positions for the translation displacements, in
        # particular, I need some positions for displacements that don't exist
        # inside individual boxes
        extra_translation_displacements = torch.tensor([
            [-1, 1, 0],  # 14
            [0, 1, 0],  # 15
            [1, 1, 0],  # 16
            [1, 0, 0],  # 17
        ], dtype=torch.long)
        translation_displacement_indices = torch.cat((torch.tensor([[0, 0, 0]], dtype=torch.long), 
                                                     self.vector_index_displacement, 
                                                     extra_translation_displacements), dim=0)
        self.translation_displacement_indices = translation_displacement_indices

        self.translation_displacements = torch.zeros_like(self.translation_displacement_indices)
        assert self.translation_displacements.shape == torch.Size([18, 3])
        assert self.translation_displacement_indices.shape == torch.Size(
            [18, 3])
        # This is 26 for 2 buckets and 17 for 1 bucket
        # This is necessary for the image - atom map and atom - image map
        self.num_neighbors = len(self.vector_index_displacement)
        # Get the lower bound of the length of a bucket in the bucket grid
        # shape 3, (Bx, By, Bz) The length is cutoff/buckets_per_cutoff +
        # epsilon
        self._register_bucket_length_lower_bound()

        # variables are not set until we have received a cell at least once
        self.register_buffer('cell_variables_are_set', torch.tensor(False, dtype=torch.bool))
        self.register_buffer('old_values_are_cached', torch.tensor(False, dtype=torch.bool))

    def forward(self, species: Tensor, 
                coordinates: Tensor, 
                cell: Tensor,
                pbc: Tensor) -> Tuple[Tensor, Tensor]:

        assert pbc.all(),\
            "Currently the cell list is only implemented for PBC in all directions"
        assert coordinates.shape[0] == 1,\
                    "Currently the cell list doesn't support batch calculations"

        coordinates = coordinates.detach()
        cell = cell.detach()
        # Cell parameters need to be set only once for constant V simulations,
        # and every time for variable V, (constant P, NPT) simulations
        if (not self.constant_volume) or (not self.cell_variables_are_set):
            shape_buckets_grid, total_buckets = self._setup_variables(cell)

        # If a new cell list is not needed just return the old values in the
        # buffers, otherwise continue
        # important: here cached values should NOT be updated
        # if they are updated the internal memory moves to the
        # new step, which is not what we want
        if self.dynamic_update and self.old_values_are_cached and (not self._need_new_list(coordinates)):
            return self.old_atom_pairs, self.old_shift_indices

        # the cell list is calculated with a skin here, since coordinates are
        # fractionalized before cell calculation, it is not needed for them to
        # be imaged to the central cell, they can lie outside the cell
        atom_pairs, shift_indices, shift_values = self._calculate_cell_list(coordinates)

        # Before the screening step we wrap the coordinates to the central cell
        coordinates = map_to_central(coordinates, cell, pbc)

        # The final screening does not use the skin, the skin is only used 
        # internally to prevent neighborlist recalculation. 
        atom_pairs, shift_indices = self._screen_with_cutoff(self.cutoff, 
                                                                           coordinates,
                                                                           atom_pairs, 
                                                                           shift_values,
                                                                           shift_indices)
        # This is done in order to prevent unnecessary rebuilds
        if self.dynamic_update:
            self._cache_values(atom_pairs, shift_indices, coordinates)
        return atom_pairs, shift_indices

    def _calculate_cell_list(self, coordinates: Tensor):
        # 2) Fractionalize coordinates
        fractional_coordinates = self._fractionalize_coordinates(coordinates)

        # 3) Get vector indices and flattened indices for atoms in unit cell
        # shape C x A x 3 this gives \vb{g}(a), the vector bucket idx
        # shape C x A this gives f(a), the flat bucket idx for atom a
        atom_vector_index, atom_flat_index =\
                                self._get_bucket_indices(fractional_coordinates)
        # 4) get image_indices -> atom_indices and inverse mapping
        # NOTE: there is not necessarily a requirement to do this here
        # both shape A, a(i) and i(a)

        # NOTE: watch out, if sorting is not stable this may scramble the atoms
        # in the same box, so that the atidx you get after applying
        # atidx_from_imidx[something] will not necessarily be the correct order
        # but since what we want is the pairs this is fine,
        # since the pairs are agnostic to species.
        imidx_from_atidx, atidx_from_imidx = self._get_imidx_converters(
            atom_flat_index)

        # FIRST WE WANT "WITHIN" IMAGE PAIRS
        # 1) Get the number of atoms in each bucket (as indexed with f idx)
        # this gives A*, A(f) , "A(f' <= f)" = Ac(f) (cumulative) f being the
        # flat bucket index, A being the number of atoms for that bucket,
        # and Ac being the cumulative number of atoms up to that bucket
        flat_bucket_count, flat_bucket_cumcount, max_in_bucket =\
            self._get_atoms_in_flat_bucket_counts(atom_flat_index)

        # 2) this are indices WITHIN the central buckets
        within_image_pairs =\
            self._get_within_image_pairs(flat_bucket_count,
                flat_bucket_cumcount, max_in_bucket)

        # NOW WE WANT "BETWEEN" IMAGE PAIRS
        # 1) Get the vector indices of all (pure) neighbors of each atom
        # this gives \vb{g}(a, n) and f(a, n)
        # shapes 1 x A x Eta x 3 and 1 x A x Eta respectively
        #
        # neighborhood count is A{n} (a), the number of atoms on the
        # neighborhood (all the neighbor buckets) of each atom,
        # A{n} (a) has shape 1 x A
        # neighbor_translation_types
        # has the type of shift for T(a, n), atom a,
        # neighbor bucket n
        neighbor_flat_indices, neighbor_translation_types =\
                self._get_neighbor_indices(atom_vector_index)
        neighbor_count = flat_bucket_count[neighbor_flat_indices]
        neighbor_cumcount = flat_bucket_cumcount[neighbor_flat_indices]
        neighborhood_count = neighbor_count.sum(-1).squeeze()

        # 2) Upper and lower part of the external pairlist this is the
        # correct "unpadded" upper
        # part of the pairlist it repeats each image
        # idx a number of times equal to the number of atoms on the
        # neighborhood of each atom
        upper = torch.repeat_interleave(imidx_from_atidx.squeeze(),
                                        neighborhood_count)
        lower, between_pairs_translations = self._get_lower_between_image_pairs(neighbor_count,
                                                                               neighbor_cumcount, 
                                                                               max_in_bucket,
                                                                               neighbor_translation_types)
        assert lower.shape == upper.shape
        between_image_pairs = torch.stack((upper, lower), dim=0)

        # concatenate within and between
        image_pairs = torch.cat((between_image_pairs, within_image_pairs), dim=1)
        atom_pairs = atidx_from_imidx[image_pairs]
        within_pairs_translations = torch.zeros(len(within_image_pairs[0]),
                                                3,
                                                device=image_pairs.device)
        # -1 is necessary to ensure correct shifts
        shift_values = -torch.cat(
            (between_pairs_translations, within_pairs_translations), dim=0)
        shift_indices = (shift_values / self.cell_diagonal).to(torch.long)
        assert shift_values.shape[0] == atom_pairs.shape[1]

        return atom_pairs, shift_indices, shift_values



    def _setup_variables(self, cell: Tensor) -> Tuple[Tensor, Tensor]:
        current_device = cell.device
        # 1) Update the cell diagonal and translation displacements
        # sizes of each side are given by norm of each basis vector of the unit cell
        self.cell_diagonal = torch.linalg.norm(cell, dim=0)
        self.cell_inverse = torch.inverse(cell)

        # I just need to index select this and add it to the coordinates to displace them
        self.translation_displacements = self.translation_displacement_indices * self.cell_diagonal

        # 2) Get max bucket index (Gx, Gy, Gz)
        # which give the size of the grid of buckets that fully covers the
        # whole volume of the unit cell U, given by "cell", and the number of
        # flat buckets (F) (equal to the total number of buckets, F
        #
        # Gx, Gy, Gz is 1 + maximum index for \vb{g} Flat bucket indices are
        # indices for the buckets written in row major order (or equivalently
        # dictionary order), the number F = Gx*Gy*Gz

        # bucket_length_lower_bound = B, unit cell U_mu = B * 3 - epsilon this means I need
        # if my unit cell es B*3 + epsilon => I can cover it with 3 buckets plus
        # some extra space that is less than a bucket, so I just stretch the buckets
        # a little bit. In this particular case shape_buckets_grid = [3, 3, 3]
        self.shape_buckets_grid = torch.floor(
            self.cell_diagonal / self.bucket_length_lower_bound).to(torch.long)

        self.total_buckets = self.shape_buckets_grid.prod()

        # 3) This is needed to scale and flatten last dimension of bucket indices
        # for row major this is (Gy * Gz, Gz, 1)
        self.scaling_for_flat_index = torch.ones(3,
                                                 dtype=torch.long,
                                                 device=current_device)
        self.scaling_for_flat_index[
            0] *= self.shape_buckets_grid[1] * self.shape_buckets_grid[2]
        self.scaling_for_flat_index[1] *= self.shape_buckets_grid[2]

        # 4) create the vector_index -> flat_index conversion tensor
        # it is not really necessary to perform circular padding,
        # since we can index the array using negative indices!
        self.vector_idx_to_flat = torch.arange(0,
                                               self.total_buckets,
                                               device=current_device)
        self.vector_idx_to_flat = self.vector_idx_to_flat.reshape(
            self.shape_buckets_grid[0], self.shape_buckets_grid[1],
            self.shape_buckets_grid[2])

        self.vector_idx_to_flat = self._pad_circular(self.vector_idx_to_flat)

        # 5) I now create a tensor that when indexed with vector indices
        # gives the shifting case for that atom/neighbor bucket
        self.translation_cases = torch.zeros_like(self.vector_idx_to_flat)
        # now I need to  fill the vector
        # in some smart way
        # this should fill the tensor in a smart way
        self.translation_cases[0, 1:-1, 1:-1] = 1
        self.translation_cases[0, 0, 1:-1] = 2
        self.translation_cases[1:-1, 0, 1:-1] = 3
        self.translation_cases[-1, 0, 1:-1] = 4
        self.translation_cases[0, -1, 0] = 5
        self.translation_cases[1:-1, -1, 0] = 6
        self.translation_cases[-1, -1, 0] = 7
        self.translation_cases[0, 1:-1, 0] = 8
        self.translation_cases[1:-1, 1:-1, 0] = 9
        self.translation_cases[-1, 1:-1, 0] = 10
        self.translation_cases[0, 0, 0] = 11
        self.translation_cases[1:-1, 0, 0] = 12
        self.translation_cases[-1, 0, 0] = 13
        # extra
        self.translation_cases[0, -1, 1:-1] = 14
        self.translation_cases[1:-1, -1, 1:-1] = 15
        self.translation_cases[-1, -1, 1:-1] = 16
        self.translation_cases[-1, 1:-1, 1:-1] = 17
        self.cell_variables_are_set = torch.tensor(True, dtype=torch.bool, device=current_device)

        return self.shape_buckets_grid, self.total_buckets

    @staticmethod
    def _pad_circular(x: Tensor) -> Tensor:
        x = x.unsqueeze(0).unsqueeze(0)
        x = functional.pad(x, (1, 1, 1, 1, 1, 1), mode='circular')
        return x.squeeze()

    def _register_bucket_length_lower_bound(self,
                                            extra_space: float = 0.00001):
        # Get the size (Bx, By, Bz) of the buckets in the grid
        # extra space by default is consistent with Amber
        # The spherical factor is different from 1 in the case of nonorthogonal
        # boxes and accounts for the "spherical protrusion", not sure exactly
        # what that is but I think it is related to the fact that the sphere of
        # radius "cutoff" around an atom needs some extra space in
        # nonorthogonal boxes.
        # note that this is not actually the bucket length used in the grid,
        # it is only a lower bound used to calculate the grid size
        spherical_factor = torch.ones(3, dtype=torch.double)
        self.bucket_length_lower_bound = (spherical_factor * self.cutoff / self.buckets_per_cutoff) + extra_space

    def _to_flat_index(self, x: Tensor) -> Tensor:
        # Converts a tensor with bucket indices in the last dimension to a
        # tensor with flat bucket indices in the last dimension if your tensor
        # is (N1, ..., Nd, 3) this transforms the tensor into (N1, ..., Nd),
        # which holds the flat bucket indices this can be done a different way,
        # same as between but this is possibly faster (?) NOTE: should
        # benchmark this or simplify the code
        assert self.scaling_for_flat_index is not None,\
            "Scaling for flat index has not been computed"
        assert x.shape[-1] == 3
        return (x * self.scaling_for_flat_index).sum(-1)

    def _expand_into_neighbors(self, x: Tensor) -> Tensor:
        # transforms a tensor of shape (... 3) with vector indices
        # in the last dimension into a tensor of shape (..., Eta, 3)
        # where Eta is the number of neighboring buckets, indexed by n
        assert self.vector_index_displacement is not None,\
            "Displacement for neighbors has not been computed"
        assert x.shape[-1] == 3
        x = x.unsqueeze(-2) + self.vector_index_displacement
        # sanity check
        assert x.shape[-1] == 3
        assert x.shape[-2] == self.vector_index_displacement.shape[0]
        return x

    def _fractionalize_coordinates(self, coordinates: Tensor) -> Tensor:
        # Scale coordinates to box size
        # I make all my coordinates relative to the box size this means for
        # instance that if the coordinate is 3.15 times the cell length, it is
        # turned into 3.15; if it is 0.15 times the cell length, it is turned
        # into 0.15, etc
        fractional_coordinates = torch.matmul(coordinates, self.cell_inverse)
        # this is done to account for possible coordinates outside the box,
        # which amber does, in order to calculate diffusion coefficients, etc
        fractional_coordinates -= fractional_coordinates.floor()
        # fractional_coordinates should be in the range [0, 1.0)
        fractional_coordinates[fractional_coordinates >= 1.0] -= 1.0
        fractional_coordinates[fractional_coordinates < 0.0] += 1.0

        assert not torch.isnan(fractional_coordinates).any(),\
                "Some fractional coordinates are NaN."
        assert not torch.isinf(fractional_coordinates).any(),\
                "Some fractional coordinates are +-Inf."
        assert (fractional_coordinates < 1.0).all(),\
            f"Some fractional coordinates are too large {fractional_coordinates[fractional_coordinates >= 1.]}"
        assert (fractional_coordinates >= 0.0).all(),\
            f"Some coordinates are too small {fractional_coordinates.masked_select(fractional_coordinates < 0.)}"
        return fractional_coordinates

    def _fractional_to_vector_bucket_indices(self,
                                             fractional: Tensor) -> Tensor:
        # transforms a tensor of fractional coordinates (shape (..., 3))
        # into a tensor of vector bucket indices (same shape)
        # Since the number of indices to iterate over is a cartesian product of 3
        # vectors it will grow with L^3 (volume) I think it is intelligent to first
        # get all indices and then apply as needed I will call the buckets "main
        # bucket" and "neighboring buckets"
        assert self.shape_buckets_grid is not None,\
            "Max bucket index not computed"
        out = torch.floor(fractional * (self.shape_buckets_grid).reshape(1, 1, -1))
        out = out.to(torch.long)
        return out

    @staticmethod
    def _get_imidx_converters(x: Tensor) -> Tuple[Tensor, Tensor]:
        # this are the "image indices", indices that sort atoms in the order of
        # the flattened bucket index.  Only occupied buckets are considered, so
        # if a bucket is unoccupied the index is not taken into account.  for
        # example if the atoms are distributed as:
        # / 1 9 8 / - / 3 2 4 / 7 /
        # where the bars delimit flat buckets, then the assoc. image indices
        # are:
        # / 0 1 2 / - / 3 4 5 / 6 /
        # atom indices can be reconstructed from the image indices, so the
        # pairlist can be built with image indices and then at the end calling
        # atom_indices_from_image_indices[pairlist] you convert to atom_indices

        # imidx_from_atidx returns tensors that convert image indices into atom
        # indices and viceversa
        # move to device necessary? not sure
        atidx_from_imidx = torch.argsort(x.squeeze()).to(x.device)
        imidx_from_atidx = torch.argsort(atidx_from_imidx).to(x.device)
        return imidx_from_atidx, atidx_from_imidx

    def _get_atoms_in_flat_bucket_counts(
            self, atom_flat_index: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # NOTE: count in flat bucket: 3 0 0 0 ... 2 0 0 0 ... 1 0 1 0 ...,
        # shape is total buckets F cumulative buckets count has the number of
        # atoms BEFORE a given bucket cumulative buckets count: 0 3 3 3 ... 3 5
        # 5 5 ... 5 6 6 7 ...
        atom_flat_index = atom_flat_index.squeeze()
        flat_bucket_count = torch.bincount(atom_flat_index,
                                           minlength=self.total_buckets).to(
                                               atom_flat_index.device)
        flat_bucket_cumcount = self.cumsum_from_zero(flat_bucket_count).to(
            atom_flat_index.device)

        # this is A*
        max_in_bucket = flat_bucket_count.max()
        return flat_bucket_count, flat_bucket_cumcount, max_in_bucket

    @staticmethod
    def cumsum_from_zero(input_: Tensor) -> Tensor:
        cumsum = torch.zeros_like(input_)
        torch.cumsum(input_[:-1], dim=0, out=cumsum[1:])
        return cumsum

    @staticmethod
    def _sort_along_row(x: Tensor,
                        max_value: Tensor,
                        row_for_sorting: int = 1) -> Tensor:
        # reorder padded pairs by ordering according to lower part instead of upper
        # based on https://discuss.pytorch.org/t/sorting-2d-tensor-by-pairs-not-columnwise/59465
        assert row_for_sorting == 1, "Due to JIT issues can only sort along row 1"
        assert x.dim() == 2, "The input must have 2 dimensions"
        assert x.shape[0] == 2, "The inut must be shape (2, ?)"
        assert row_for_sorting == 1 or row_for_sorting == 0
        # NOTE: This code fails due to some JIT error, which prevents sorting along
        # row 0
        # if row_for_sorting == 1:
        # aug = x[0, :] + x[1, :] * max_value
        # elif row_for_sorting == 0:
        #    aug = x[0, :] * max_value + x[1, :]
        aug = x[0, :] + x[1, :] * max_value
        # x = x.index_select(1, stable_sort(aug)[0]) # 0 - indices, 1 - values
        x = x.index_select(1,
                           torch.sort(aug).indices)  # 0 - indices, 1 - values
        return x

    def _get_within_image_pairs(self, flat_bucket_count: Tensor,
                                flat_bucket_cumcount: Tensor,
                                max_in_bucket: Tensor) -> Tensor:
        # max_in_bucket = maximum number of atoms contained in any bucket
        current_device = flat_bucket_count.device
        max_in_bucket = max_in_bucket

        # get all indices f that have pairs inside
        # these are A(w) and Ac(w), and withpairs_flat_index is actually f(w)
        # NOTE: workaround since nonzero(as_tuple=True) is not JITable
        withpairs_flat_index = (flat_bucket_count > 1).nonzero()
        withpairs_flat_index = withpairs_flat_index.t().unbind(0)
        withpairs_flat_bucket_count = flat_bucket_count[
            withpairs_flat_index[0]]
        withpairs_flat_bucket_cumcount = flat_bucket_cumcount[
            withpairs_flat_index[0]]

        padded_pairs = torch.triu_indices(max_in_bucket,
                                          max_in_bucket,
                                          offset=1,
                                          device=current_device)
        padded_pairs = self._sort_along_row(padded_pairs,
                                            max_in_bucket,
                                            row_for_sorting=1)
        # shape (2, pairs) + shape (withpairs, 1, 1) = shape (withpairs, 2, pairs)
        padded_pairs = padded_pairs + withpairs_flat_bucket_cumcount.reshape(
            -1, 1, 1)
        # basically this repeats the padded pairs "withpairs" times and adds to all of
        # them the cumulative counts
        # now we unravel all pairs, which remain in the correct order in the
        # second row (the order within same numbers in the first row is actually
        # not essential)
        padded_pairs = padded_pairs.permute(1, 0, 2).reshape(2, -1)

        # TODO this code is very confusing
        # NOTE: JIT bug makes it so that we have to add the "one" tensor here
        one = torch.tensor(1, device=current_device, dtype=torch.long)
        max_pairs_in_bucket = torch.div(max_in_bucket * (max_in_bucket - one),
                                        2,
                                        rounding_mode='floor')
        mask = torch.arange(0, max_pairs_in_bucket, device=current_device)
        num_buckets_with_pairs = len(withpairs_flat_index[0])
        mask = mask.repeat(num_buckets_with_pairs, 1)

        # NOTE: JIT bug makes it so that we have to add the "one" tensor here
        one = torch.tensor(1, device=current_device, dtype=torch.long)
        withpairs_count_pairs = torch.div(withpairs_flat_bucket_count * (withpairs_flat_bucket_count - one), 2,
                                          rounding_mode='floor')
        mask = (mask < withpairs_count_pairs.reshape(-1, 1)).reshape(-1)

        upper = torch.masked_select(padded_pairs[0], mask)
        lower = torch.masked_select(padded_pairs[1], mask)
        within_image_pairs = torch.stack((upper, lower), dim=0)
        return within_image_pairs

    def _get_lower_between_image_pairs(
            self, neighbor_count: Tensor, neighbor_cumcount: Tensor,
            max_in_bucket: Tensor,
            neighbor_translation_types: Tensor) -> Tuple[Tensor, Tensor]:
        # neighbor_translation_types has shape 1 x At x Eta
        # 3) now I need the LOWER part
        # this gives, for each atom, for each neighbor bucket, all the
        # unpadded, unshifted atom neighbors
        # this is basically broadcasted to the shape of fna
        # shape is 1 x A x eta x A*
        atoms = neighbor_count.shape[1]
        padded_atom_neighbors = torch.arange(0,
                                             max_in_bucket,
                                             device=neighbor_count.device)
        padded_atom_neighbors = padded_atom_neighbors.reshape(1, 1, 1, -1)
        padded_atom_neighbors = padded_atom_neighbors.repeat(
            1, atoms, self.num_neighbors, 1)

        # repeat the neighbor translation types to account for all neighboring atoms
        neighbor_translation_types = neighbor_translation_types.unsqueeze(
            -1).repeat(1, 1, 1, padded_atom_neighbors.shape[-1])

        # now I need to add A(f' < fna) shift the padded atom neighbors to get
        # image indices I need to check here that the cumcount is correct since
        # it was technically done with imidx so I need to check correctnes of
        # both counting schemes, but first I create the mask to unpad
        # and then I shift to the correct indices
        mask = (padded_atom_neighbors < neighbor_count.unsqueeze(-1))
        padded_atom_neighbors += neighbor_cumcount.unsqueeze(-1)
        # the mask should have the same shape as padded_atom_neighbors, and
        # now all that is left is to apply the mask in order to unpad
        assert padded_atom_neighbors.shape == mask.shape
        assert neighbor_translation_types.shape == mask.shape
        lower = torch.masked_select(padded_atom_neighbors, mask)
        between_pairs_translations = torch.masked_select(neighbor_translation_types, mask)
        between_pairs_translations = self.translation_displacements.index_select(
            0, between_pairs_translations)
        assert between_pairs_translations.shape[-1] == 3
        return lower, between_pairs_translations

    def _get_bucket_indices(
            self, fractional_coordinates: Tensor) -> Tuple[Tensor, Tensor]:
        atom_vector_index = self._fractional_to_vector_bucket_indices(fractional_coordinates)
        atom_flat_index = self._to_flat_index(atom_vector_index)
        return atom_vector_index, atom_flat_index

    def _get_neighbor_indices(
            self, atom_vector_index: Tensor) -> Tuple[Tensor, Tensor]:
        # This is actually pure neighbors, so it doesn't have
        # "the bucket itself"
        # These are
        # - g(a, n),  shape 1 x A x Eta x 3
        # - f(a, n),  shape 1 x A x Eta
        # These give, for each atom, the flat index or the vector index of its
        # neighbor buckets (neighbor buckets indexed by n).
        neighbor_vector_indices = self._expand_into_neighbors(
            atom_vector_index)
        # these vector indices have the information that says whether to shift
        # each pair and what amount to shift it

        atoms = neighbor_vector_indices.shape[1]
        neighbors = neighbor_vector_indices.shape[2]
        neighbor_vector_indices += torch.ones(1,
                                              dtype=torch.long,
                                              device=atom_vector_index.device)
        neighbor_vector_indices = neighbor_vector_indices.reshape(-1, 3)
        # NOTE: This is needed instead of unbind due to torchscript bug
        # neighbor_vector_indices = neighbor_vector_indices.unbind(1)
        neighbor_flat_indices = self.vector_idx_to_flat[
            neighbor_vector_indices[:, 0], neighbor_vector_indices[:, 1],
            neighbor_vector_indices[:, 2]]
        neighbor_translation_types = self.translation_cases[
            neighbor_vector_indices[:, 0], neighbor_vector_indices[:, 1],
            neighbor_vector_indices[:, 2]]
        neighbor_translation_types = neighbor_translation_types.reshape(
            1, atoms, neighbors)
        neighbor_flat_indices = neighbor_flat_indices.reshape(
            1, atoms, neighbors)
        return neighbor_flat_indices, neighbor_translation_types

    def _cache_values(self, atom_pairs: Tensor, 
                            shift_indices: Tensor, 
                            coordinates: Tensor):

        self.old_atom_pairs = atom_pairs.detach()
        self.old_shift_indices = shift_indices.detach()
        self.old_cell_diagonal = self.cell_diagonal.detach()
        self.old_coordinates = coordinates.detach()
        self.old_values_are_cached = torch.tensor(True, dtype=torch.bool, device=coordinates.device)

    def reset_cached_values(self):
        dtype = self.cell_diagonal.dtype
        device = self.cell_diagonal.device
        self._cache_values(torch.zeros(1, dtype=torch.long, device=device), 
                           torch.zeros(1, dtype=torch.long, device=device), 
                           torch.zeros(1, dtype=dtype, device=device), 
                           torch.zeros(1, dtype=dtype, device=device))
        self.old_values_are_cached = torch.tensor(False, dtype=torch.bool, device=device)

    def _need_new_list(self, coordinates: Tensor):
        # Check if any coordinate exceedes half the skin depth,
        # if a coordinate exceedes this then the cell list has to be rebuilt
        box_scaling = self.cell_diagonal / self.old_cell_diagonal
        delta = coordinates - self.old_coordinates * box_scaling
        dist_squared = delta.pow(2).sum(-1)
        need_new_list = (dist_squared > (self.skin / 2) ** 2).any()
        return need_new_list
