import math
import sys
from typing import Tuple, Optional, NamedTuple, List
import warnings
import importlib_metadata

import torch
from torch import Tensor

from .cutoffs import CutoffCosine
from .aev_terms import AngularTerms, RadialTerms
from .neighbors import FullPairwise

cuaev_is_installed = 'torchani.cuaev' in importlib_metadata.metadata(
    __package__.split('.')[0]).get_all('Provides')

if cuaev_is_installed:
    # We need to import torchani.cuaev to tell PyTorch to initialize torch.ops.cuaev
    from .. import cuaev  # type: ignore # noqa: F401
else:
    warnings.warn("cuaev not installed")

if sys.version_info[:2] < (3, 7):

    class FakeFinal:
        def __getitem__(self, x):
            return x

    Final = FakeFinal()
else:
    from torch.jit import Final


class SpeciesAEV(NamedTuple):
    species: Tensor
    aevs: Tensor


def jit_unused_if_no_cuaev(condition=cuaev_is_installed):
    def decorator(func):
        if not condition:
            return torch.jit.unused(func)
        return torch.jit.export(func)
    return decorator


class AEVComputer(torch.nn.Module):
    r"""The AEV computer that takes coordinates as input and outputs aevs.

    Arguments:
        Rcr (float): :math:`R_C` in equation (2) when used at equation (3)
            in the `ANI paper`_.
        Rca (float): :math:`R_C` in equation (2) when used at equation (4)
            in the `ANI paper`_.
        EtaR (:class:`torch.Tensor`): The 1D tensor of :math:`\eta` in
            equation (3) in the `ANI paper`_.
        ShfR (:class:`torch.Tensor`): The 1D tensor of :math:`R_s` in
            equation (3) in the `ANI paper`_.
        EtaA (:class:`torch.Tensor`): The 1D tensor of :math:`\eta` in
            equation (4) in the `ANI paper`_.
        Zeta (:class:`torch.Tensor`): The 1D tensor of :math:`\zeta` in
            equation (4) in the `ANI paper`_.
        ShfA (:class:`torch.Tensor`): The 1D tensor of :math:`R_s` in
            equation (4) in the `ANI paper`_.
        ShfZ (:class:`torch.Tensor`): The 1D tensor of :math:`\theta_s` in
            equation (4) in the `ANI paper`_.
        num_species (int): Number of supported atom types.
        use_cuda_extension (bool): Whether to use cuda extension for faster calculation (needs cuaev installed).

    .. _ANI paper:
        http://pubs.rsc.org/en/Content/ArticleLanding/2017/SC/C6SC05720A#!divAbstract
    """
    num_species: Final[int]
    num_species_pairs: Final[int]

    angular_length: Final[int]
    angular_sublength: Final[int]
    radial_length: Final[int]
    radial_sublength: Final[int]
    aev_length: Final[int]

    use_cuda_extension: Final[bool]

    def __init__(self,
                Rcr: float,
                Rca: float,
                EtaR: Tensor,
                ShfR: Tensor,
                EtaA: Tensor,
                Zeta: Tensor,
                ShfA: Tensor,
                ShfZ: Tensor,
                num_species: int,
                use_cuda_extension=False,
                cutoff_function=CutoffCosine,
                neighborlist=FullPairwise):
        super().__init__()
        assert Rca <= Rcr, "Current implementation of AEVComputer assumes Rca <= Rcr"
        self.use_cuda_extension = use_cuda_extension
        self.num_species = num_species
        self.num_species_pairs = num_species * (num_species + 1) // 2

        self.register_buffer('triu_index',
                             self._calculate_triu_index(num_species))
        self.register_buffer('default_cell', torch.eye(3, dtype=torch.float))
        self.register_buffer('default_pbc', torch.zeros(3, dtype=torch.bool))

        # radial and angular calculators
        self.angular_terms = AngularTerms(EtaA,
                                          Zeta,
                                          ShfA,
                                          ShfZ,
                                          Rca,
                                          cutoff_function=cutoff_function)
        self.radial_terms = RadialTerms(EtaR,
                                        ShfR,
                                        Rcr,
                                        cutoff_function=cutoff_function)

        # neighborlist uses radial cutoff only
        self.neighborlist = neighborlist(Rcr) if neighborlist is not None else None

        # length variables are updated once radial and angular terms are initialized
        self._update_lengths()

        # cuda aev
        if self.use_cuda_extension:
            assert cuaev_is_installed, "AEV cuda extension is not installed"
        # cuaev_computer is created only when use_cuda_extension is True.
        # However jit needs to know cuaev_computer's Type even when
        # use_cuda_extension is False
        if cuaev_is_installed:
            self._init_cuaev_computer()

    @jit_unused_if_no_cuaev()
    def _init_cuaev_computer(self):
        self.cuaev_computer = torch.classes.cuaev.CuaevComputer(self.radial_terms.cutoff.item(),
                                                                self.angular_terms.cutoff.item(),
                                                                self.radial_terms.EtaR.flatten(),
                                                                self.radial_terms.ShfR.flatten(),
                                                                self.angular_terms.EtaA.flatten(),
                                                                self.angular_terms.Zeta.flatten(),
                                                                self.angular_terms.ShfA.flatten(),
                                                                self.angular_terms.ShfZ.flatten(),
                                                                self.num_species)

    @staticmethod
    def _calculate_triu_index(num_species: int) -> Tensor:
        # helper method for initialization
        species1, species2 = torch.triu_indices(num_species,
                                                num_species).unbind(0)
        pair_index = torch.arange(species1.shape[0], dtype=torch.long)
        ret = torch.zeros(num_species, num_species, dtype=torch.long)
        ret[species1, species2] = pair_index
        ret[species2, species1] = pair_index
        return ret

    def _update_lengths(self):
        # the lengths of buffers can't be changed with load_state_dict so we can
        # cache all lengths in the model itself
        self.radial_sublength = self.radial_terms.sublength()
        self.angular_sublength = self.angular_terms.sublength()
        self.radial_length = self.radial_sublength * self.num_species
        self.angular_length = self.angular_sublength * self.num_species_pairs
        self.aev_length = self.radial_length + self.angular_length

    @classmethod
    def cover_linearly(cls,
                       radial_cutoff: float,
                       angular_cutoff: float,
                       radial_eta: float,
                       angular_eta: float,
                       radial_dist_divisions: int,
                       angular_dist_divisions: int,
                       zeta: float,
                       angle_sections: int,
                       num_species: int,
                       angular_start: float = 0.9,
                       radial_start: float = 0.9):
        r""" Provides a convenient way to linearly fill cutoffs

        This is a user friendly constructor that builds an
        :class:`torchani.AEVComputer` where the subdivisions along the the
        distance dimension for the angular and radial sub-AEVs, and the angle
        sections for the angular sub-AEV, are linearly covered with shifts. By
        default the distance shifts start at 0.9 Angstroms.

        To reproduce the ANI-1x AEV's the signature ``(5.2, 3.5, 16.0, 8.0, 16, 4, 32.0, 8, 4)``
        can be used.
        """
        # This is intended to be self documenting code that explains the way
        # the AEV parameters for ANI1x were chosen. This is not necessarily the
        # best or most optimal way but it is a relatively sensible default.
        Rcr = radial_cutoff
        Rca = angular_cutoff
        EtaR = torch.tensor([float(radial_eta)])
        EtaA = torch.tensor([float(angular_eta)])
        Zeta = torch.tensor([float(zeta)])
        ShfR = torch.linspace(radial_start, radial_cutoff,
                              radial_dist_divisions + 1)[:-1]
        ShfA = torch.linspace(angular_start, angular_cutoff,
                              angular_dist_divisions + 1)[:-1]
        angle_start = math.pi / (2 * angle_sections)
        ShfZ = (torch.linspace(0, math.pi, angle_sections + 1) + angle_start)[:-1]
        return cls(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species)

    def forward(self,
                input_: Tuple[Tensor, Tensor],
                cell: Optional[Tensor] = None,
                pbc: Optional[Tensor] = None) -> SpeciesAEV:
        """Compute AEVs

        Arguments:
            input_ (tuple): Can be one of the following two cases:

                If you don't care about periodic boundary conditions at all,
                then input can be a tuple of two tensors: species, coordinates.
                species must have shape ``(N, A)``, coordinates must have shape
                ``(N, A, 3)`` where ``N`` is the number of molecules in a batch,
                and ``A`` is the number of atoms.

                .. warning::

                    The species must be indexed in 0, 1, 2, 3, ..., not the element
                    index in periodic table. Check :class:`torchani.SpeciesConverter`
                    if you want periodic table indexing.

                .. note:: The coordinates, and cell are in Angstrom.

                If you want to apply periodic boundary conditions, then the input
                would be a tuple of two tensors (species, coordinates) and two keyword
                arguments `cell=...` , and `pbc=...` where species and coordinates are
                the same as described above, cell is a tensor of shape (3, 3) of the
                three vectors defining unit cell:

                .. code-block:: python

                    tensor([[x1, y1, z1],
                            [x2, y2, z2],
                            [x3, y3, z3]])

                and pbc is boolean vector of size 3 storing if pbc is enabled
                for that direction.

        Returns:
            NamedTuple: Species and AEVs. species are the species from the input
            unchanged, and AEVs is a tensor of shape ``(N, A, self.aev_length)``
        """
        species, coordinates = input_
        # check shapes for correctness
        assert species.dim() == 2
        assert coordinates.dim() == 3
        assert (species.shape == coordinates.shape[:2]) and (coordinates.shape[2] == 3)
        assert (cell is not None and pbc is not None) or (cell is None and pbc is None)

        cell = cell if cell is not None else self.default_cell
        pbc = pbc if pbc is not None else self.default_pbc

        if self.use_cuda_extension:
            assert not pbc.any(), "cuaev currently does not support PBC"
            aev = self._compute_cuaev(species, coordinates)
            return SpeciesAEV(species, aev)

        atom_index12, shift_indices = self.neighborlist(species, coordinates, cell, pbc)
        shift_values = shift_indices.to(cell.dtype) @ cell
        aev = self._compute_aev(species, coordinates, atom_index12, shift_values)
        return SpeciesAEV(species, aev)

    @jit_unused_if_no_cuaev()
    def _compute_cuaev(self, species, coordinates):
        species_int = species.to(torch.int32)
        aev = torch.ops.cuaev.run(coordinates, species_int, self.cuaev_computer)
        return aev

    def _compute_aev(self, species: Tensor, coordinates: Tensor,
                    atom_index12: Tensor, shift_values: Tensor) -> Tensor:

        species12 = species.flatten()[atom_index12]
        vec = self._compute_difference_vector(coordinates, atom_index12,
                                              shift_values)

        distances = vec.norm(2, -1)
        radial_aev = self._compute_radial_aev(species12, distances,
                                              atom_index12, species.shape)

        # Rca is usually much smaller than Rcr, using neighbor list with
        # cutoff = Rcr is a waste of resources Now we will get a smaller neighbor
        # list that only cares about atoms with distances <= Rca
        even_closer_indices = (distances <= self.angular_terms.cutoff).nonzero().flatten()

        atom_index12 = atom_index12.index_select(1, even_closer_indices)
        species12 = species12.index_select(1, even_closer_indices)
        vec = vec.index_select(0, even_closer_indices)

        angular_aev = self._compute_angular_aev(species12, vec, atom_index12,
                                                species.shape)

        return torch.cat([radial_aev, angular_aev], dim=-1)

    @staticmethod
    def _compute_difference_vector(coordinates: Tensor, atom_index12: Tensor,
                                   shift_values: Tensor) -> Tensor:
        coordinates = coordinates.flatten(0, 1)
        selected_coordinates = coordinates.view(-1, 3).index_select(
            0, atom_index12.view(-1)).view(2, -1, 3)
        vec = selected_coordinates[0] - selected_coordinates[1] + shift_values
        return vec

    def _compute_angular_aev(self, species12: Tensor, vec: Tensor,
                             atom_index12: Tensor,
                             species_shape: List[int]) -> Tensor:
        num_molecules, num_atoms = species_shape

        central_atom_index, pair_index12, sign12 = self._triple_by_molecule(
            atom_index12)
        species12_small = species12[:, pair_index12]
        vec12 = vec.index_select(0, pair_index12.view(-1)).view(
            2, -1, 3) * sign12.unsqueeze(-1)
        species12_ = torch.where(sign12 == 1, species12_small[1],
                                 species12_small[0])

        angular_terms_ = self.angular_terms(vec12)
        angular_aev = angular_terms_.new_zeros(
            (num_molecules * num_atoms * self.num_species_pairs,
             self.angular_sublength))
        index = central_atom_index * self.num_species_pairs + self.triu_index[
            species12_[0], species12_[1]]
        angular_aev.index_add_(0, index, angular_terms_)
        angular_aev = angular_aev.reshape(num_molecules, num_atoms,
                                          self.angular_length)
        return angular_aev

    def _compute_radial_aev(self, species12: Tensor, distances: Tensor,
                            atom_index12: Tensor,
                            species_shape: List[int]) -> Tensor:
        num_molecules, num_atoms = species_shape

        radial_terms_ = self.radial_terms(distances)
        radial_aev = radial_terms_.new_zeros(
            (num_molecules * num_atoms * self.num_species,
             self.radial_sublength))
        print("species12 shape", species12.shape)
        print("atom_index12 shape", atom_index12.shape)
        index12 = atom_index12 * self.num_species + species12.flip(0)
        print("radial terms shape", radial_terms_.shape)
        print("index12 shape", index12.shape)
        radial_aev.index_add_(0, index12[0], radial_terms_)
        radial_aev.index_add_(0, index12[1], radial_terms_)
        radial_aev = radial_aev.reshape(num_molecules, num_atoms,
                                        self.radial_length)
        return radial_aev

    def _triple_by_molecule(
            self, atom_index12: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Input: indices for pairs of atoms that are close to each other.
        each pair only appear once, i.e. only one of the pairs (1, 2) and
        (2, 1) exists.

        Output: indices for all central atoms and it pairs of neighbors. For
        example, if input has pair (0, 1), (0, 2), (0, 3), (0, 4), (1, 2),
        (1, 3), (1, 4), (2, 3), (2, 4), (3, 4), then the output would have
        central atom 0, 1, 2, 3, 4 and for cental atom 0, its pairs of neighbors
        are (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)
        """
        # convert representation from pair to central-others
        ai1 = atom_index12.view(-1)
        sorted_ai1, rev_indices = ai1.sort()

        # sort and compute unique key
        uniqued_central_atom_index, counts = torch.unique_consecutive(
            sorted_ai1, return_inverse=False, return_counts=True)

        # compute central_atom_index
        one = torch.tensor(1, device=atom_index12.device, dtype=atom_index12.dtype)
        pair_sizes = torch.div(counts * (counts - one), 2, rounding_mode='floor')
        pair_indices = torch.repeat_interleave(pair_sizes)
        central_atom_index = uniqued_central_atom_index.index_select(
            0, pair_indices)

        # do local combinations within unique key, assuming sorted
        m = counts.max().item() if counts.numel() > 0 else 0
        n = pair_sizes.shape[0]
        intra_pair_indices = torch.tril_indices(
            m, m, -1, device=ai1.device).unsqueeze(1).expand(-1, n, -1)
        mask = (torch.arange(intra_pair_indices.shape[2], device=ai1.device) < pair_sizes.unsqueeze(1)).flatten()
        sorted_local_index12 = intra_pair_indices.flatten(1, 2)[:, mask]
        sorted_local_index12 += self._cumsum_from_zero(counts).index_select(
            0, pair_indices)

        # unsort result from last part
        local_index12 = rev_indices[sorted_local_index12]

        # compute mapping between representation of central-other to pair
        n = atom_index12.shape[1]
        sign12 = ((local_index12 < n).to(torch.int8) * 2) - 1
        return central_atom_index, local_index12 % n, sign12

    def _constants(self):
        return self.radial_terms.cutoff, self.radial_terms.EtaR,\
            self.radial_terms.ShfR, self.angular_terms.cutoff,\
            self.angular_terms.ShfZ, self.angular_terms.EtaA,\
            self.angular_terms.Zeta, self.angular_terms.ShfA, self.num_species

    @staticmethod
    def _cumsum_from_zero(input_: Tensor) -> Tensor:
        cumsum = torch.zeros_like(input_)
        torch.cumsum(input_[:-1], dim=0, out=cumsum[1:])
        return cumsum


class AEVComputerBare(AEVComputer):

    def __init__(self, *args, **kwargs):
        """Bare version of the AEVComputer, with no internal neighborlist"""

        if 'neighborlist' not in kwargs.keys():
            kwargs.update({'neighborlist': None})
        if 'use_cuda_extension' not in kwargs.keys():
            kwargs.update({'use_cuda_extension': False})

        assert kwargs['neighborlist'] is None, "AEVComputerBare doesn't use a neighborlist"
        assert not kwargs['use_cuda_extension'], "AEVComputerBare doesn't suport cuaev"
        super().__init__(*args, **kwargs)

    def forward(self, input_: Tuple[Tensor, Tensor],
                atom_index12: Optional[Tensor] = None,
                shift_values: Optional[Tensor] = None) -> SpeciesAEV:
        """Compute AEVs
        Returns:
            NamedTuple: Species and AEVs. species are the species from the input
            unchanged, and AEVs is a tensor of shape ``(N, A, self.aev_length)``
        """
        species, coordinates = input_

        # It is convenient to keep these arguments optional due to JIT, but
        # actually they are needed for this class
        assert atom_index12 is not None
        assert shift_values is not None
        # check shapes for correctness
        assert species.dim() == 2
        assert coordinates.dim() == 3
        assert (species.shape == coordinates.shape[:2]) and (coordinates.shape[2] == 3)

        # check shapes of neighborlist
        assert atom_index12.dim() == 2 and atom_index12.shape[0] == 2
        assert shift_values.dim() == 2 and shift_values.shape[1] == 3
        assert atom_index12.shape[1] == shift_values.shape[0]
        print('max index', atom_index12.max())
        print('min index', atom_index12.min())
        
        # first we prescreen the input neighborlist in case some of the values are
        # at distances larger than the cutoff for the radial terms
        # this may happen if the neighborlist uses some sort of skin value to rebuild
        atom_index12, shift_values = self._screen_with_cutoff(coordinates.detach(), 
                                                              atom_index12.detach(), 
                                                              shift_values.detach(), 
                                                              self.radial_terms.cutoff.detach())

        aev = self._compute_aev(species, coordinates, atom_index12, shift_values)
        return SpeciesAEV(species, aev)

    def _screen_with_cutoff(self, coordinates: Tensor, shift_values: Tensor, shift_values: Tensor, cutoff: Tensor):
        # screen neighbors that are further away than a given cutoff
        vec = self._compute_difference_vector(coordinates.detach(), atom_index12, shift_values.detach())
        distances_sq = vec.pow(2).sum(-1)
        close_indices = (distances_sq <= cutoff**2).nonzero().flatten()
        atom_index12 = atom_index12.index_select(1, even_closer_indices)
        return atom_index12, shift_values
        



