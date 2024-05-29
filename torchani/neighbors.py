import typing as tp
import math

import torch
from torch import Tensor
from torch.jit import Final

from torchani.utils import map_to_central, cumsum_from_zero
from torchani.tuples import NeighborData


def rescreen(
    cutoff: float,
    neighbors: NeighborData,
) -> NeighborData:
    closer_indices = (neighbors.distances <= cutoff).nonzero().flatten()
    return NeighborData(
        indices=neighbors.indices.index_select(1, closer_indices),
        distances=neighbors.distances.index_select(0, closer_indices),
        diff_vectors=neighbors.diff_vectors.index_select(0, closer_indices),
    )


class Neighborlist(torch.nn.Module):
    default_pbc: Tensor
    default_cell: Tensor

    def __init__(self):
        """Compute pairs of atoms that are neighbors, uses pbc depending on
        weather pbc.any() is True or not

        Arguments:
            coordinates (:class:`torch.Tensor`): tensor of shape
                (molecules, atoms, 3) for atom coordinates.
        """
        super().__init__()
        self.register_buffer(
            "default_cell", torch.eye(3, dtype=torch.float), persistent=False
        )
        self.register_buffer(
            "default_pbc", torch.zeros(3, dtype=torch.bool), persistent=False
        )
        self.diff_vectors = torch.empty(0)

    @torch.jit.export
    def _compute_bounding_cell(
        self, coordinates: Tensor, eps: float
    ) -> tp.Tuple[Tensor, Tensor]:
        # this works but its not needed for this naive implementation
        # This should return a bounding cell
        # for the molecule, in all cases, also it displaces coordinates a fixed
        # value, so that they fit inside the cell completely. This should have
        # no effects on forces or energies

        # add an epsilon to pad due to floating point precision
        min_ = torch.min(coordinates.view(-1, 3), dim=0).values - eps
        max_ = torch.max(coordinates.view(-1, 3), dim=0).values + eps
        largest_dist = max_ - min_
        coordinates = coordinates - min_
        cell = self.default_cell * largest_dist
        assert (coordinates > 0.0).all()
        assert (coordinates < torch.norm(cell, dim=1)).all()
        return coordinates, cell

    def _screen_with_cutoff(
        self,
        cutoff: float,
        coordinates: Tensor,
        input_neighbor_indices: Tensor,
        shift_values: tp.Optional[Tensor] = None,
        mask: tp.Optional[Tensor] = None,
    ) -> NeighborData:
        # passing an infinite cutoff will only work for non pbc conditions
        # (shift values must be None)
        #
        # Screen a given neighborlist using a cutoff and return a neighborlist with
        # atoms that are within that cutoff, for all molecules in a coordinate set.
        #
        # If the initial coordinates have more than one molecule in the batch
        # dimension then this function expects an input neighborlist that
        # correctly indexes flattened coordinates obtained via
        # coordinates.view(-1, 3).  If the initial coordinates have only one
        # molecule then the output neighborlist will index non flattened
        # coordinates correctly

        # First we check if there are any dummy atoms in species, if there are
        # we get rid of those pairs to prevent wasting resources in calculation
        # of dummy distances
        if mask is not None:
            if mask.any():
                mask = mask.view(-1)[input_neighbor_indices.view(-1)].view(2, -1)
                non_dummy_pairs = (~torch.any(mask, dim=0)).nonzero().flatten()
                input_neighbor_indices = input_neighbor_indices.index_select(
                    1, non_dummy_pairs
                )
                # shift_values can be None when there are no pbc conditions to prevent
                # torch from launching kernels with only zeros
                if shift_values is not None:
                    shift_values = shift_values.index_select(0, non_dummy_pairs)

        coordinates = coordinates.view(-1, 3)
        # Difference vector and distances could be obtained for free when
        # screening, unfortunately distances have to be recalculated twice each
        # time they are screened, since otherwise torch prepares to calculate
        # derivatives of multiple distances that will later be disregarded
        if cutoff != math.inf:
            coordinates_ = coordinates.detach()
            # detached calculation #
            coords0 = coordinates_.index_select(0, input_neighbor_indices[0])
            coords1 = coordinates_.index_select(0, input_neighbor_indices[1])
            diff_vectors = coords0 - coords1
            if shift_values is not None:
                diff_vectors += shift_values
            distances = diff_vectors.norm(2, -1)
            in_cutoff = (distances <= cutoff).nonzero().flatten()
            # ------------------- #

            screened_neighbor_indices = input_neighbor_indices.index_select(
                1, in_cutoff
            )
            if shift_values is not None:
                shift_values = shift_values.index_select(0, in_cutoff)
        else:
            assert (
                shift_values is None
            ), "PBC can't be implemented with an infinite cutoff"
            screened_neighbor_indices = input_neighbor_indices

        coords0 = coordinates.index_select(0, screened_neighbor_indices[0])
        coords1 = coordinates.index_select(0, screened_neighbor_indices[1])
        screened_diff_vectors = coords0 - coords1
        if shift_values is not None:
            screened_diff_vectors += shift_values

        # This is the very first `diff_vectors` that are used to calculate
        # various potentials: 2-body (radial), 3-body (angular), repulsion,
        # dispersion and etc. To enable stress calculation using partial_fdotr
        # approach, `diff_vectors` requires the `requires_grad` flag to be set
        # and needs to be saved for future differentiation.
        screened_diff_vectors.requires_grad_()
        self.diff_vectors = screened_diff_vectors

        screened_distances = screened_diff_vectors.norm(2, -1)
        return NeighborData(
            indices=screened_neighbor_indices,
            distances=screened_distances,
            diff_vectors=screened_diff_vectors,
        )

    def get_diff_vectors(self):
        return self.diff_vectors

    def dummy(self) -> NeighborData:
        # return dummy neighbor data
        device = self.default_cell.device
        dtype = self.default_cell.dtype
        indices = torch.tensor([[0], [1]], dtype=torch.long, device=device)
        distances = torch.tensor([1.0], dtype=dtype, device=device)
        diff_vectors = torch.tensor([[1.0, 0.0, 0.0]], dtype=dtype, device=device)
        return NeighborData(
            indices=indices,
            distances=distances,
            diff_vectors=diff_vectors,
        )

    @torch.jit.export
    def _recast_long_buffers(self) -> None:
        pass


class FullPairwise(Neighborlist):
    default_shift_values: Tensor

    def __init__(self):
        """Compute pairs of atoms that are neighbors, uses pbc depending on
        weather pbc.any() is True or not
        """
        super().__init__()
        self.register_buffer(
            "default_shift_values", torch.tensor(0.0), persistent=False
        )

    def forward(
        self,
        species: Tensor,
        coordinates: Tensor,
        cutoff: float,
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
    ) -> NeighborData:
        """
        Arguments:
            coordinates (:class:`torch.Tensor`): tensor of shape
                (molecules, atoms, 3) for atom coordinates.
            cell (:class:`torch.Tensor`): tensor of shape (3, 3) of the three vectors
                defining unit cell: tensor([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])
            cutoff (float): the cutoff inside which atoms are considered pairs
            pbc (:class:`torch.Tensor`): boolean tensor of shape (3,) storing
            wheather pbc is required
        """
        assert (cell is not None and pbc is not None) or (cell is None and pbc is None)
        cell = cell if cell is not None else self.default_cell
        pbc = pbc if pbc is not None else self.default_pbc

        mask = species == -1
        if pbc.any():
            atom_index12, shift_indices = self._full_pairwise_pbc(
                species, cutoff, cell, pbc
            )
            shift_values = shift_indices.to(cell.dtype) @ cell
            # before being screened the coordinates have to be mapped to the
            # central cell in case they are not inside it, this is not necessary
            # if there is no pbc
            coordinates = map_to_central(coordinates, cell, pbc)
            return self._screen_with_cutoff(
                cutoff, coordinates, atom_index12, shift_values, mask
            )
        else:
            num_molecules = species.shape[0]
            num_atoms = species.shape[1]
            # Create a pairwise neighborlist for all molecules and all atoms,
            # assuming that there are no atoms at all. Dummy species will be
            # screened later
            atom_index12 = torch.triu_indices(
                num_atoms, num_atoms, 1, device=species.device
            )
            if num_molecules > 1:
                atom_index12 = atom_index12.unsqueeze(1).repeat(1, num_molecules, 1)
                atom_index12 += num_atoms * torch.arange(
                    num_molecules, device=mask.device
                ).view(1, -1, 1)
                atom_index12 = atom_index12.view(-1).view(2, -1)
            return self._screen_with_cutoff(
                cutoff, coordinates, atom_index12, shift_values=None, mask=mask
            )

    def _full_pairwise_pbc(
        self,
        species: Tensor,
        cutoff: float,
        cell: Tensor,
        pbc: Tensor,
    ) -> tp.Tuple[Tensor, Tensor]:
        cell = cell.detach()
        shifts = self._compute_shifts(cutoff, cell, pbc)
        num_atoms = species.shape[1]
        all_atoms = torch.arange(num_atoms, device=cell.device)

        # Step 2: center cell
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
        all_atom_pairs = torch.cat([p12_center, p12], dim=1)
        return all_atom_pairs, shifts_all

    def _compute_shifts(self, cutoff: float, cell: Tensor, pbc: Tensor) -> Tensor:
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
        num_repeats = torch.ceil(cutoff * inv_distances).to(torch.long)
        num_repeats = torch.where(pbc, num_repeats, num_repeats.new_zeros(()))
        r1 = torch.arange(1, num_repeats[0].item() + 1, device=cell.device)
        r2 = torch.arange(1, num_repeats[1].item() + 1, device=cell.device)
        r3 = torch.arange(1, num_repeats[2].item() + 1, device=cell.device)
        o = torch.zeros(1, dtype=torch.long, device=cell.device)
        return torch.cat(
            [
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
            ]
        )


class CellList(Neighborlist):
    verlet: Final[bool]
    constant_volume: Final[bool]

    grid_numel: int
    skin: float
    cell_diagonal: Tensor
    grid_shape: Tensor
    vector_idx_to_flat: Tensor
    wrap_kind_from_idx3: Tensor
    surround_offset_idx3: Tensor
    wrap_offset_idx3: Tensor
    bucket_length_lower_bound: Tensor
    spherical_factor: Tensor

    def __init__(
        self,
        buckets_per_cutoff: int = 1,
        verlet: bool = False,
        skin: tp.Optional[float] = None,
        constant_volume: bool = False,
    ):
        super().__init__()

        # right now I will only support this, and the extra neighbors are
        # hardcoded, but full support for arbitrary buckets per cutoff is possible
        assert (
            buckets_per_cutoff == 1
        ), "Cell list currently only supports one bucket per cutoff"
        assert not verlet, "Verlet cell list has issues and should not be used"
        self.constant_volume = constant_volume
        self.verlet = verlet
        self.grid_numel: int = 0
        self.register_buffer(
            "spherical_factor", torch.full(size=(3,), fill_value=1.0), persistent=False
        )
        self.register_buffer("cell_diagonal", torch.zeros(1), persistent=False)
        self.register_buffer(
            "grid_shape", torch.zeros(3, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "vector_idx_to_flat", torch.zeros(1, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "wrap_kind_from_idx3", torch.zeros(1, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "surround_offset_idx3",
            torch.zeros(1, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "wrap_offset_idx3",
            torch.zeros(1, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "bucket_length_lower_bound", torch.zeros(1), persistent=False
        )
        if skin is None:
            self.skin = 1.0 if verlet else 0.0
        else:
            self.skin = skin

        # only used for verlet option
        self.register_buffer("old_cell_diagonal", torch.zeros(1), persistent=False)
        self.register_buffer(
            "old_shift_indices", torch.zeros(1, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "old_atom_pairs", torch.zeros(1, dtype=torch.long), persistent=False
        )
        self.register_buffer("old_coordinates", torch.zeros(1), persistent=False)

        # "buckets_per_cutoff" determines how fine grained the 3D grid is, with respect
        # to the distance cutoff. This is 2 for amber, but 1 for torchani.
        self.buckets_per_cutoff = buckets_per_cutoff

        # Get the grid_idx3 offsets for the surrounding buckets of an
        # arbitrary bucket (TODO: this is different from pmemd, check why!)
        #
        # In order to avoid double counting, consider only half of the
        # surrounding buckets.
        #
        # Choose all buckets in the bottom plane, and the lower half
        # half of the buckets in the same plane of a given bucket
        # (not sure if other choices are possible).
        #
        # 0-offset is not included
        #
        # Order is "right-to-left, top-to-bottom"
        #
        # The selected buckets in the planes are:
        # ("x" selected elements, "-" non-selected and "o" reference element)
        # top,   same,  bottom,
        # |---|  |---|  |xxx|
        # |---|  |xo-|  |xxx|
        # |---|  |xxx|  |xxx|

        grid_idx3_offsets = [
            # Surrounding buckets in the same plane (gz-offset = 0)
            [-1, 0, 0],  # 1
            [-1, -1, 0],  # 2
            [0, -1, 0],  # 3
            [1, -1, 0],  # 4
            # Surrounding buckets in bottom plane (gz-offset = -1)
            [-1, 1, -1],  # 5
            [0, 1, -1],  # 6
            [1, 1, -1],  # 7
            [-1, 0, -1],  # 8
            [0, 0, -1],  # 9
            [1, 0, -1],  # 10
            [-1, -1, -1],  # 11
            [0, -1, -1],  # 12
            [1, -1, -1],  # 13
        ]
        # These are used to get the surrounding buckets
        # shape (neighbors=13, 3)
        self.surround_offset_idx3 = torch.tensor(grid_idx3_offsets, dtype=torch.long)

        # In the case of the wrap-offsets, we may need to displace atoms in all
        # directions in the "same plane", so we have to add the extra displacements
        # (We still don't need to displace in the top plane)
        grid_idx3_offsets.insert(0, [0, 0, 0])  # 0
        grid_idx3_offsets.extend(
            [
                [-1, 1, 0],  # 14
                [0, 1, 0],  # 15
                [1, 1, 0],  # 16
                [1, 0, 0],  # 17
            ],
        )
        # shape (wraps=18, 3)
        self.wrap_offset_idx3 = torch.tensor(grid_idx3_offsets, dtype=torch.long)

        # These variables are not set until we have received a cell at least once
        self.last_cutoff = -1.0
        self.cell_variables_are_set = False
        self.old_values_are_cached = False

    @torch.jit.export
    def _recast_long_buffers(self) -> None:
        # for cell list
        self.grid_shape = self.grid_shape.to(dtype=torch.long)
        self.vector_idx_to_flat = self.vector_idx_to_flat.to(dtype=torch.long)
        self.wrap_kind_from_idx3 = self.wrap_kind_from_idx3.to(dtype=torch.long)
        self.surround_offset_idx3 = self.surround_offset_idx3.to(dtype=torch.long)
        self.wrap_offset_idx3 = self.wrap_offset_idx3.to(dtype=torch.long)

    def forward(
        self,
        species: Tensor,
        coordinates: Tensor,
        cutoff: float,
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
    ) -> NeighborData:
        assert cutoff >= 0.0, "Cutoff must be a positive float"
        assert coordinates.shape[0] == 1, "Cell list doesn't support batches"
        if cell is None:
            assert pbc is None or not pbc.any()
        # if cell is None then a bounding cell for the molecule is obtained
        # from the coordinates, in this case the coordinates are assumed to be
        # mapped to the central cell, since anything else would be meaningless
        pbc = pbc if pbc is not None else self.default_pbc
        assert pbc.all() or (not pbc.any()), "CellList supports PBC in all or no dirs"

        if cell is None:
            # Displaced coordinates only used for computation if pbc is not required
            coordinates_displaced, cell = self._compute_bounding_cell(
                coordinates.detach(), eps=1e-3
            )
        else:
            coordinates_displaced = coordinates.detach()

        if (
            (not self.constant_volume)
            or (not self.cell_variables_are_set)
            or (cutoff != self.last_cutoff)
        ):
            # Cell parameters need to be set only once for constant V
            # simulations, and every time for variable V  simulations If the
            # neighborlist cutoff is changed, the variables have to be reset
            # too
            self._setup_variables(cell.detach(), cutoff)

        if (
            self.verlet
            and self.old_values_are_cached
            and (not self._need_new_list(coordinates_displaced.detach()))
        ):
            # If a new cell list is not needed use the old cached values
            # IMPORTANT: here cached values should NOT be updated, moving cache
            # to the new step is incorrect
            atom_pairs = self.old_atom_pairs
            shift_indices: tp.Optional[Tensor] = self.old_shift_indices
        else:
            # The cell list is calculated with a skin here. Since coordinates are
            # fractionalized before cell calculation, it is not needed for them to
            # be imaged to the central cell, they can lie outside the cell.
            atom_pairs, shift_indices = self._calculate_cell_list(
                coordinates_displaced.detach(),
                cell,
                pbc,
            )
            # 'Verlet' prevent unnecessary rebuilds of the neighborlist
            if self.verlet:
                self._cache_values(
                    atom_pairs, shift_indices, coordinates_displaced.detach()
                )

        if pbc.any():
            assert shift_indices is not None
            shift_values = shift_indices.to(cell.dtype) @ cell
            # Before the screening step we map the coordinates to the central cell,
            # same as with a full pairwise calculation
            coordinates = map_to_central(coordinates, cell.detach(), pbc)
            # The final screening does not use the skin, the skin is only used
            # internally to prevent neighborlist recalculation.  We must screen
            # even if the list is not rebuilt, two atoms may have moved a long
            # enough distance that they are not neighbors anymore, but a short
            # enough distance that the neighborlist is not rebuilt. Rebuilds
            # happen only if it can't be guaranteed that the cached
            # neighborlist holds at least all atom pairs, but it may hold more.
            return self._screen_with_cutoff(
                cutoff, coordinates, atom_pairs, shift_values, (species == -1)
            )
        else:
            return self._screen_with_cutoff(
                cutoff, coordinates, atom_pairs, shift_values=None, mask=(species == -1)
            )

    def _calculate_cell_list(
        self,
        coordinates: Tensor,  # shape (C, A, 3)
        cell: Tensor,  # shape (3, 3)
        pbc: Tensor,  # shape (3,)
    ) -> tp.Tuple[Tensor, tp.Optional[Tensor]]:
        # The cell is spanned by a 3D grid of "buckets" or "grid elements",
        # which has grid_shape=(GX, GY, GZ) and grid_numel=G=(GX * GY * GZ)

        # 1) Get location of each atom in the grid, given by a "grid_idx3"
        # or by a single flat "grid_idx" (g).
        # Shapes (C, A, 3) and (C, A)
        atom_grid_idx3 = coords_to_grid_idx3(coordinates, cell, self.grid_shape)
        atom_grid_idx = flatten_grid_idx3(atom_grid_idx3, self.grid_shape)

        # 2) Get image pairs of atoms WITHIN atoms inside a bucket
        # To do this, first calculate:
        # - Num atoms in each bucket "count_in_grid[g]", and the max, (c-max)
        # - Cumulative num atoms *before* each bucket "comcount_in_grid[g]"
        # Both shapes (G,)
        count_in_grid, cumcount_in_grid = count_atoms_in_buckets(
            atom_grid_idx, self.grid_numel
        )
        count_in_grid_max: int = int(count_in_grid.max())
        # Shape (2, W)
        _image_pairs_within = image_pairs_within(
            count_in_grid,
            cumcount_in_grid,
            count_in_grid_max,
        )

        # 3) Get image pairs BETWEEN atoms inside a bucket and its surrounding buckets
        # To do this, first get a grid_idx3 of the surrounding buckets of each atom
        # roughly: "grid_idx3[a, n, 3]" and "grid_idx[a, n]"
        # shapes (C, A, N, 3) and (C, A, N) (N=13 for 1-bucket-per-cutoff)
        atom_surround_idx3, atom_surround_idx = self._get_surround_idxs(atom_grid_idx3)

        # 4) Calc upper and lower part of the image_pairs_between.
        # The "unpadded" upper part of the pairlist repeats each image idx a
        # number of times equal to the number of atoms on the surroundings of
        # each atom

        # All 3 shapes are (C, A, N)
        count_in_atom_surround = count_in_grid[atom_surround_idx]
        cumcount_in_atom_surround = cumcount_in_grid[atom_surround_idx]
        atom_surround_wrap_kind = self._wrap_kind_from_surround_idx3(atom_surround_idx3)

        # Both shapes are (B,)
        lower_between, wrap_kind_between = self._lower_image_pairs_between(
            count_in_atom_surround,
            cumcount_in_atom_surround,
            atom_surround_wrap_kind,
            count_in_grid_max,
        )

        # Total count of all atoms in buckets surrounding a given atom.
        # shape (C, A, N) -> (C, A) -> (C*A,) (get rid of C with view)
        total_count_in_atom_surround = count_in_atom_surround.sum(-1).view(-1)

        # Both shapes (C*A,) for i[a] and a[i]
        atom_to_image, image_to_atom = atom_image_converters(atom_grid_idx)

        # For each atom we have one image_pair_between associated with each of
        # the atoms in its surrounding buckets, so we repeat the image-idx of each
        # atom that many times.
        # shape (C*A,), (C*A) -> (B,)
        upper_between = torch.repeat_interleave(
            atom_to_image,
            total_count_in_atom_surround,
        )
        # shape (2, B)
        _image_pairs_between = torch.stack((upper_between, lower_between), dim=0)

        # 5) Get the necessary shifts. If no PBC is needed also get rid of the
        # image_pairs_between that need wrapping
        if not pbc.any():
            _image_pairs_between = _masked_select(
                _image_pairs_between,
                (wrap_kind_between == 0),
                1,
            )
            shift_idxs = None
        else:
            shift_idxs_between = self.wrap_offset_idx3.index_select(
                0, wrap_kind_between
            )
            shift_idxs_within = torch.zeros(
                _image_pairs_within.shape[1],
                3,
                device=cell.device,
                dtype=torch.long,
            )
            # -1 is necessary here (?!)
            shift_idxs = torch.cat((-shift_idxs_between, shift_idxs_within), dim=0)

        # 6) Concatenate all image pairs, and convert to atom pairs
        image_pairs = torch.cat((_image_pairs_between, _image_pairs_within), dim=1)
        atom_pairs = image_to_atom[image_pairs]
        return atom_pairs, shift_idxs

    def _setup_variables(self, cell: Tensor, cutoff: float, extra_space: float = 1e-5):
        device = cell.device
        # Get the shape (GX, GY, GZ) of the grid. Some extra space is used as slack
        # (consistent with SANDER neighborlist by default)
        #
        # The spherical factor is different from 1 in the case of nonorthogonal
        # boxes and accounts for the "spherical protrusion", which is related
        # to the fact that the sphere of radius "cutoff" around an atom needs
        # some extra space in nonorthogonal boxes.
        #
        # NOTE: This is not actually the bucket length used in the grid,
        # it is only a lower bound used to calculate the grid size
        spherical_factor = self.spherical_factor
        bucket_length_lower_bound = (
            spherical_factor * cutoff / self.buckets_per_cutoff
        ) + extra_space

        # 1) Update the cell diagonal and translation displacements
        # sizes of each side are given by norm of each basis vector of the unit cell
        self.cell_diagonal = torch.linalg.norm(cell, dim=0)

        # 2) Get max bucket index (Gx, Gy, Gz)
        # which give the size of the grid of buckets that fully covers the
        # whole volume of the unit cell U, given by "cell", and the number of
        # flat buckets (G,) (equal to the total number of buckets, F )
        #
        # Gx, Gy, Gz is 1 + maximum index for vector g. Flat bucket indices are
        # indices for the buckets written in row major order (or equivalently
        # dictionary order), the number G = GX * GY * GZ

        # bucket_length_lower_bound = B, unit cell U_mu = B * 3 - epsilon this
        # means I can cover it with 3 buckets plus some extra space that is
        # less than a bucket, so I just stretch the buckets a little bit. In
        # this particular case grid_shape = (3, 3, 3)
        self.grid_shape = torch.div(
            self.cell_diagonal, bucket_length_lower_bound, rounding_mode="floor"
        ).to(torch.long)

        self.grid_numel = int(self.grid_shape.prod())
        if self.grid_numel == 0:
            raise RuntimeError("Cell is too small to perform pbc calculations")

        # 4) create the vector_index -> flat_index conversion tensor
        vector_idx_to_flat = torch.arange(0, self.grid_numel, device=device)
        # shape (GX, GY, GZ)
        vector_idx_to_flat = vector_idx_to_flat.view(
            int(self.grid_shape[0]),
            int(self.grid_shape[1]),
            int(self.grid_shape[2]),
        )
        self.vector_idx_to_flat = _pad_circular_3d(vector_idx_to_flat)

        # 5) I now create a tensor that when indexed with vector indices
        # gives the shifting case for that atom/neighbor bucket
        self.wrap_kind_from_idx3 = torch.zeros_like(self.vector_idx_to_flat)
        # now I need to  fill the vector
        # in some smart way
        # this should fill the tensor in a smart way

        # top plane
        # [-1, 0, 0],  # 1
        # [-1, -1, 0],  # 2
        # [0, -1, 0],  # 3
        # [1, -1, 0],  # 4
        # # bottom plane
        # [-1, 1, -1],  # 5
        # [0, 1, -1],  # 6
        # [1, 1, -1],  # 7
        # [-1, 0, -1],  # 8
        # [0, 0, -1],  # 9
        # [1, 0, -1],  # 10
        # [-1, -1, -1],  # 11
        # [0, -1, -1],  # 12
        # [1, -1, -1],  # 13

        self.wrap_kind_from_idx3[0, 1:-1, 1:-1] = 1
        self.wrap_kind_from_idx3[0, 0, 1:-1] = 2
        self.wrap_kind_from_idx3[1:-1, 0, 1:-1] = 3
        self.wrap_kind_from_idx3[-1, 0, 1:-1] = 4
        self.wrap_kind_from_idx3[0, -1, 0] = 5
        self.wrap_kind_from_idx3[1:-1, -1, 0] = 6
        self.wrap_kind_from_idx3[-1, -1, 0] = 7
        self.wrap_kind_from_idx3[0, 1:-1, 0] = 8
        self.wrap_kind_from_idx3[1:-1, 1:-1, 0] = 9
        self.wrap_kind_from_idx3[-1, 1:-1, 0] = 10
        self.wrap_kind_from_idx3[0, 0, 0] = 11
        self.wrap_kind_from_idx3[1:-1, 0, 0] = 12
        self.wrap_kind_from_idx3[-1, 0, 0] = 13
        # extra
        self.wrap_kind_from_idx3[0, -1, 1:-1] = 14
        self.wrap_kind_from_idx3[1:-1, -1, 1:-1] = 15
        self.wrap_kind_from_idx3[-1, -1, 1:-1] = 16
        self.wrap_kind_from_idx3[-1, 1:-1, 1:-1] = 17

        self.cell_variables_are_set = True

    def _lower_image_pairs_between(
        self,
        count_in_atom_surround: Tensor,  # shape (C, A, N)
        cumcount_in_atom_surround: Tensor,  # shape (C, A, N)
        atom_surround_wrap_kind: Tensor,  # shape (C, A, N)
        count_in_grid_max: int,  # scalar
    ) -> tp.Tuple[Tensor, Tensor]:
        device = count_in_atom_surround.device
        # Calculate "lower" part of the image_pairs between buckets

        # this gives, for each atom, for each neighbor bucket, all the
        # unpadded, unshifted atom neighbors
        # this is basically broadcasted to the shape of fna
        mols, atoms, neighbors = count_in_atom_surround.shape

        # shape is (c-max)
        padded_atom_neighbors = torch.arange(0, count_in_grid_max, device=device)
        # shape (1, 1, 1, c-max)
        padded_atom_neighbors = padded_atom_neighbors.view(1, 1, 1, -1)
        # repeat is needed instead of expand here due to += neighbor_cumcount (?)
        # shape (C, A, N, c-max)
        padded_atom_neighbors = padded_atom_neighbors.repeat(mols, atoms, neighbors, 1)

        # repeat the surround wrap kinds to account for all neighboring atoms
        # repeat is needed instead of expand due to reshaping later (?)
        # shape  (C, A, N, 1)
        atom_surround_wrap_kind = atom_surround_wrap_kind.unsqueeze(-1)
        # shape  (C, A, N, c-max)
        atom_surround_wrap_kind = atom_surround_wrap_kind.repeat(
            1, 1, 1, count_in_grid_max
        )

        # now I need to add A(f' < fna) shift the padded atom neighbors to get
        # image indices I need to check here that the cumcount is correct since
        # it was technically done with imidx so I need to check correctnes of
        # both counting schemes, but first I create the mask to unpad
        # and then I shift to the correct indices
        # shape (C, A, N, c-max)
        mask = padded_atom_neighbors < count_in_atom_surround.unsqueeze(-1)
        padded_atom_neighbors.add_(cumcount_in_atom_surround.unsqueeze(-1))

        # Now apply the mask in order to unpad
        # Both shapes are (B,)
        lower_between = _masked_select(padded_atom_neighbors.view(-1), mask, 0)
        wrap_kind_between = _masked_select(atom_surround_wrap_kind.view(-1), mask, 0)
        return lower_between, wrap_kind_between

    def _get_surround_idxs(self, atom_grid_idx3: Tensor) -> tp.Tuple[Tensor, Tensor]:
        # Calc the grid_idx3 and grid_idx associated with the buckets
        # surrounding a given atom.
        #
        # The surrounding buckets will either lie in the central cell or wrap
        # around one or more dimensions due to PBC. In the latter case the
        # atom_surround_idx3 may be negative, or larger than the corrseponding
        # grid_shape dim
        #
        # Depending on whether there is 1, 2 or 3 negative (or overflowing)
        # idxs, the atoms in the bucket should be shifted along 1, 2 or 3 dim,
        # since the central bucket is up against a wall, an edge or a corner
        # respectively.
        mols, atoms, _ = atom_grid_idx3.shape
        neighbors, _ = self.surround_offset_idx3.shape

        # This is actually strict neighbors, so it doesn't have
        # "the bucket itself"
        # These are
        # - g(a, n),  shape 1 x A x N x 3
        # - f(a, n),  shape 1 x A x N
        # These give, for each atom, the flat index or the vector index of its
        # neighbor buckets (neighbor buckets indexed by n).
        # these vector indices have the information that says whether to shift
        # each pair and what amount to shift it

        # After this step some of the atom_surround_idx3 are negative, and some
        # may overflow
        atom_surround_idx3 = atom_grid_idx3.view(
            mols, atoms, 1, 3
        ) + self.surround_offset_idx3.view(mols, 1, neighbors, 3)
        atom_surround_idx3.add_(1)
        atom_surround_idx3 = atom_surround_idx3.view(-1, 3)

        # atom_surround_idx contains only idxs that are positive, it indexes
        # buckets inside the central cell
        #
        # NOTE: This is needed instead of unbind due to torchscript bug
        atom_surround_idx = self.vector_idx_to_flat[
            atom_surround_idx3[:, 0],
            atom_surround_idx3[:, 1],
            atom_surround_idx3[:, 2],
        ]
        return (
            atom_surround_idx3.view(mols, atoms, neighbors, 3),
            atom_surround_idx.view(mols, atoms, neighbors),
        )

    def _wrap_kind_from_surround_idx3(self, atom_surround_idx3: Tensor) -> Tensor:
        mols, atoms, neighbors, _ = atom_surround_idx3.shape
        atom_surround_idx3 = atom_surround_idx3.view(-1, 3)
        atom_surround_wrap_kind = self.wrap_kind_from_idx3[
            atom_surround_idx3[:, 0],
            atom_surround_idx3[:, 1],
            atom_surround_idx3[:, 2],
        ]
        atom_surround_wrap_kind = atom_surround_wrap_kind.view(mols, atoms, neighbors)
        return atom_surround_wrap_kind

    def _cache_values(
        self,
        atom_pairs: Tensor,
        shift_indices: tp.Optional[Tensor],
        coordinates: Tensor,
    ):
        self.old_atom_pairs = atom_pairs.detach()
        if shift_indices is not None:
            self.old_shift_indices = shift_indices.detach()
        self.old_coordinates = coordinates.detach()
        self.old_cell_diagonal = self.cell_diagonal.detach()
        self.old_values_are_cached = True

    def reset_cached_values(self) -> None:
        float_dtype = self.cell_diagonal.dtype
        device = self.cell_diagonal.device
        self._cache_values(
            torch.zeros(1, dtype=torch.long, device=device),
            torch.zeros(1, dtype=torch.long, device=device),
            torch.zeros(1, dtype=float_dtype, device=device),
        )
        self.old_values_are_cached = False

    def _need_new_list(self, coordinates: Tensor) -> bool:
        if not self.verlet:
            return True
        # Check if any coordinate exceedes half the skin depth,
        # if a coordinate exceedes this then the cell list has to be rebuilt
        box_scaling = self.cell_diagonal / self.old_cell_diagonal
        delta = coordinates - self.old_coordinates * box_scaling
        dist_squared = delta.pow(2).sum(-1)
        need_new_list = (dist_squared > (self.skin / 2) ** 2).any().item()
        return bool(need_new_list)


def coords_to_grid_idx3(
    coordinates: Tensor,  # shape (C, A, 3)
    cell: Tensor,  # shape (3, 3)
    grid_shape: Tensor,  # shape (3,)
) -> Tensor:
    # 1) Fractionalize coordinates. All coordinates will be relative to the
    # cell lengths after this step, which means they lie in the range [0., 1.)
    fractionals = coords_to_fractional(coordinates, cell)  # shape (C, A, 3)
    # 2) assign to each fractional the corresponding grid_idx3
    grid_idx3 = torch.floor(fractionals * grid_shape).to(torch.long)
    return grid_idx3


def coords_to_fractional(coordinates: Tensor, cell: Tensor) -> Tensor:
    # Scale coordinates to box size
    #
    # Make all coordinates relative to the box size. This means for
    # instance that if the coordinate is 3.15 times the cell length, it is
    # turned into 3.15; if it is 0.15 times the cell length, it is turned
    # into 0.15, etc
    fractional_coords = torch.matmul(coordinates, cell.inverse())
    # this is done to account for possible coordinates outside the box,
    # which amber does, in order to calculate diffusion coefficients, etc
    fractional_coords -= fractional_coords.floor()
    # fractional_coordinates should be in the range [0, 1.0)
    fractional_coords[fractional_coords >= 1.0] += -1.0
    fractional_coords[fractional_coords < 0.0] += 1.0
    return fractional_coords


def flatten_grid_idx3(grid_idx3: Tensor, grid_shape: Tensor) -> Tensor:
    # Converts a tensor that holds idx3 (all of which lie inside the central
    # grid) to one that holds flat idxs (last dimension is removed). For
    # row-major flattening the factors needed are: (GY * GZ, GZ, 1)
    grid_factors = grid_shape.clone()
    grid_factors[0] = grid_shape[1] * grid_shape[2]
    grid_factors[1] = grid_shape[2]
    grid_factors[2] = 1
    return (grid_idx3 * grid_factors).sum(-1)


def atom_image_converters(grid_idx: Tensor) -> tp.Tuple[Tensor, Tensor]:
    # NOTE: Since sorting is not stable this may scramble the atoms
    # so that the atidx you get after applying
    # atidx_from_imidx[something] will not be the correct order
    # since what we want is the pairs this is fine, pairs are agnostic to
    # species. (?)

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

    # atom_to_image returns tensors that convert image indices into atom
    # indices and viceversa
    # move to device necessary? not sure
    grid_idx = grid_idx.view(-1)  # shape (C, A) -> (A,), get rid of C
    image_to_atom = torch.argsort(grid_idx)
    atom_to_image = torch.argsort(image_to_atom)
    # output shapes are (A,) (A,)
    return atom_to_image, image_to_atom


def count_atoms_in_buckets(
    atom_grid_idx: Tensor,  # shape (C, A)
    grid_numel: int,
) -> tp.Tuple[Tensor, Tensor]:
    # NOTE: count in flat bucket: 3 0 0 0 ... 2 0 0 0 ... 1 0 1 0 ...,
    # shape is total grid elements G. grid_cumcount has the number of
    # atoms BEFORE a given bucket cumulative buckets count: 0 3 3 3 ... 3 5
    # 5 5 ... 5 6 6 7 ...
    atom_grid_idx = atom_grid_idx.view(-1)  # shape (A,), get rid of C
    # G = the total number of grid elements
    count_in_grid = torch.bincount(atom_grid_idx, minlength=grid_numel)
    # Both shape (G,)
    return count_in_grid, cumsum_from_zero(count_in_grid)


def image_pairs_within(
    count_in_grid: Tensor,  # shape (G,)
    cumcount_in_grid: Tensor,  # shape (G,)
    count_in_grid_max: int,  # max number of atoms in any bucket
) -> Tensor:
    device = count_in_grid.device
    # Calc all possible image-idx-pairs within each central bucket ("W" in total)
    # Output is shape (2, W)
    #
    # NOTE: Inside each central bucket there are grid_count[g] num atoms.
    # These atoms are indexed with an "image idx", "i", different from the "atom idx"
    # which indexes the atoms in the coords
    # For instance:
    # - central bucket g=0 has "image" atoms 0...grid_count[0]
    # - central bucket g=1 has "image" atoms grid_count[0]...grid_count[1], etc

    # 1) Get idxs "g" that have atom pairs inside ("H" in total).
    # Index them with 'h' using g[h], from this, get count_in_haspairs[h] and
    # cumcount_in_haspairs[h].
    # shapes are (H,)
    haspairs_idx_to_grid_idx = (count_in_grid > 1).nonzero().view(-1)
    count_in_haspairs = count_in_grid.index_select(0, haspairs_idx_to_grid_idx)
    cumcount_in_haspairs = cumcount_in_grid.index_select(0, haspairs_idx_to_grid_idx)

    # 2) Get image pairs pairs assuming every bucket (with pairs) has
    # the same num atoms as the fullest one. To do this:
    # - Get the image-idx-pairs for the fullest bucket
    # - Repeat (view) the image pairs in the fullest bucket H-times,
    # - Add to each repeat the cumcount of atoms in all previous buckets.
    #
    # After this step:
    # - There are more pairs than needed
    # - Some of the extra pairs may have out-of-bounds idxs
    # Screen the incorrect, unneeded pairs in the next step.
    # shapes are (2, cp-max) and (2, H*cp-max)
    image_pairs_in_fullest_bucket = torch.tril_indices(
        count_in_grid_max,
        count_in_grid_max,
        offset=-1,
        device=device,
    )
    _image_pairs_within = (
        image_pairs_in_fullest_bucket.view(2, 1, -1)
        + cumcount_in_haspairs.view(1, -1, 1)
    ).view(2, -1)

    # 3) Get actual number of pairs in each bucket (with pairs), and a
    # mask that selects those from the unscreened pairs
    # shapes (H,) (cp-max,), (H, cp-max)
    paircount_in_haspairs = torch.div(
        count_in_haspairs * torch.sub(count_in_haspairs, 1),
        2,
        rounding_mode="floor",
    )
    mask = torch.arange(0, image_pairs_in_fullest_bucket.shape[1], device=device)
    mask = mask.view(1, -1) < paircount_in_haspairs.view(-1, 1)

    # 4) Screen the incorrect, unneeded pairs.
    # shape (2, H*cp-max) -> (2, W)
    return _masked_select(_image_pairs_within, mask, 1)


def _masked_select(x: Tensor, mask: Tensor, idx: int) -> Tensor:
    # x.index_select(0, mask.view(-1).nonzero().view(-1)) is EQUIVALENT to:
    # torch.masked_select(x, mask) but FASTER
    # view(-1)...view(-1) is used to avoid reshape (not sure if that is faster)
    return x.index_select(idx, mask.view(-1).nonzero().view(-1))


def _pad_circular_3d(x: Tensor) -> Tensor:
    # pad a 3D tensor, (1-left, 1-right, 1-top, 1-bottom 1-back, 1-front)
    assert x.ndim == 3
    GX, GY, GZ = x.shape
    x = x.view(1, 1, GX, GY, GZ)
    x = torch.nn.functional.pad(x, (1, 1, 1, 1, 1, 1), mode="circular")
    return x.view(GX + 2, GY + 2, GZ + 2)


NeighborlistArg = tp.Union[
    tp.Literal[
        "full_pairwise",
        "cell_list",
        "verlet_cell_list",
        "base",
    ],
    Neighborlist,
]


def parse_neighborlist(neighborlist: NeighborlistArg = "base") -> Neighborlist:
    if neighborlist == "full_pairwise":
        neighborlist = FullPairwise()
    elif neighborlist == "cell_list":
        neighborlist = CellList()
    elif neighborlist == "verlet_cell_list":
        neighborlist = CellList(verlet=True)
    elif neighborlist == "base":
        neighborlist = Neighborlist()
    elif not isinstance(neighborlist, Neighborlist):
        raise ValueError(f"Unsupported neighborlist: {neighborlist}")
    return tp.cast(Neighborlist, neighborlist)
