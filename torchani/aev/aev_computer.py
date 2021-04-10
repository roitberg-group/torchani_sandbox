import math
from typing import Tuple, Optional, NamedTuple
import warnings
import importlib_metadata

import torch
from torch import Tensor

from ..utils import cumsum_from_zero
from ..compat import Final
# modular parts of AEVComputer
from .cutoffs import _parse_cutoff_fn
from .aev_terms import _parse_angular_terms, _parse_radial_terms
from .neighbors import _parse_neighborlist, BaseNeighborlist


cuaev_is_installed = 'torchani.cuaev' in importlib_metadata.metadata(
    __package__.split('.')[0]).get_all('Provides')

if cuaev_is_installed:
    # We need to import torchani.cuaev to tell PyTorch to initialize torch.ops.cuaev
    from .. import cuaev  # type: ignore # noqa: F401
else:
    warnings.warn("cuaev not installed")


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
                cutoff_fn='cosine',
                cutoff_fn_kwargs=None,
                neighborlist='full_pairwise',
                neighborlist_kwargs=None,
                radial_terms='ani1',
                radial_kwargs=None,
                angular_terms='ani1',
                angular_kwargs=None):

        super().__init__()
        self.use_cuda_extension = use_cuda_extension
        self.num_species = num_species
        self.num_species_pairs = num_species * (num_species + 1) // 2

        self.register_buffer('triu_index',
                             self._calculate_triu_index(num_species))
        self.register_buffer('default_cell', torch.eye(3, dtype=torch.float))
        self.register_buffer('default_pbc', torch.zeros(3, dtype=torch.bool))
        self.triu_index: Tensor
        self.default_cell: Tensor
        self.default_pbc: Tensor

        # currently only cosine, smooth and custom cutoffs are supported
        # only ANI-1 style angular terms or radial terms
        # and only full pairwise neighborlist
        cutoff_fn = _parse_cutoff_fn(cutoff_fn)
        angular_terms = _parse_angular_terms(angular_terms)
        radial_terms = _parse_radial_terms(radial_terms)
        neighborlist = _parse_neighborlist(neighborlist)

        # build angular terms
        a_args = (EtaA, Zeta, ShfA, ShfZ, Rca)
        a_kwargs = dict() if angular_kwargs is None else angular_kwargs
        self.angular_terms = angular_terms(*a_args, **a_kwargs, cutoff_fn=cutoff_fn, cutoff_fn_kwargs=cutoff_fn_kwargs)

        # build radial terms
        r_args = (EtaR, ShfR, Rcr)
        r_kwargs = dict() if radial_kwargs is None else radial_kwargs
        self.radial_terms = radial_terms(*r_args, **r_kwargs, cutoff_fn=cutoff_fn, cutoff_fn_kwargs=cutoff_fn_kwargs)

        # build neighborlist
        nl_args = (Rcr,)
        nl_kwargs = dict() if neighborlist_kwargs is None else neighborlist_kwargs
        self.neighborlist = neighborlist(*nl_args, **nl_kwargs)

        self._validate_cutoffs_init()

        # length variables are updated once radial and angular terms are initialized
        # The lengths of buffers can't be changed with load_state_dict so we can
        # cache all lengths in the model itself
        self.radial_sublength = self.radial_terms.sublength()
        self.angular_sublength = self.angular_terms.sublength()
        self.radial_length = self.radial_sublength * self.num_species
        self.angular_length = self.angular_sublength * self.num_species_pairs
        self.aev_length = self.radial_length + self.angular_length

        # cuda aev
        if self.use_cuda_extension:
            assert cuaev_is_installed, "AEV cuda extension is not installed"
        # cuaev_computer is created only when use_cuda_extension is True.
        # However jit needs to know cuaev_computer's Type even when
        # use_cuda_extension is False. **NOTE: this is only a kind of "dummy"
        # initialization, it is always necessary to reinitialize in forward at
        # least once, since some tensors may be on CPU at this point**
        if cuaev_is_installed:
            self._init_cuaev_computer()

        # We defer true cuaev initialization to forward so that we ensure that
        # all tensors are in GPU once it is initialized.
        self.register_buffer('cuaev_is_initialized', torch.tensor(False))
        self.cuaev_is_initialized: Tensor

    def _validate_cutoffs_init(self):
        # validate cutoffs and emit warnings for strange configurations
        if self.neighborlist.cutoff > self.radial_terms.get_cutoff():
            raise ValueError(f"""The neighborlist cutoff {self.neighborlist.cutoff}
                    is larger than the radial cutoff, {self.radial_terms.get_cutoff()}.
                    AEVComputer can't possibly reuse the neighborlist for other
                    interactions, you probably want to use a different class,
                    since this configuration will will not use the extra atom pairs""")
        elif self.neighborlist.cutoff < self.radial_terms.get_cutoff():
            raise ValueError(f"""The neighborlist cutoff,
                             {self.neighborlist.cutoff} should be at least as
                             large as the radial cutoff, {self.radial_terms.get_cutoff()}""")
        if self.angular_terms.get_cutoff() > self.radial_terms.get_cutoff():
            raise ValueError(f"""Current implementation assumes angular cutoff
                             {self.angular_terms.get_cutoff()} < radial cutoff
                             {self.radial_terms.get_cutoff()}""")

    @jit_unused_if_no_cuaev()
    def _init_cuaev_computer(self):
        self.cuaev_computer = torch.classes.cuaev.CuaevComputer(self.radial_terms.get_cutoff().item(),
                                                                self.angular_terms.get_cutoff().item(),
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
                       radial_start: float = 0.9, **kwargs):
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
        EtaR = torch.tensor([radial_eta], dtype=torch.float)
        EtaA = torch.tensor([angular_eta], dtype=torch.float)
        Zeta = torch.tensor([zeta], dtype=torch.float)
        ShfR = torch.linspace(radial_start, radial_cutoff,
                              radial_dist_divisions + 1)[:-1].to(torch.float)
        ShfA = torch.linspace(angular_start, angular_cutoff,
                              angular_dist_divisions + 1)[:-1].to(torch.float)
        angle_start = math.pi / (2 * angle_sections)
        ShfZ = (torch.linspace(0, math.pi, angle_sections + 1) + angle_start)[:-1].to(torch.float)
        return cls(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species, **kwargs)

    @classmethod
    def like_1x(cls, **kwargs):
        cls_kwargs = {
            'radial_cutoff': 5.2,
            'angular_cutoff': 3.5,
            'radial_eta': 16.0,
            'angular_eta': 8.0,
            'radial_dist_divisions': 16,
            'angular_dist_divisions': 4,
            'zeta': 32.0,
            'angle_sections': 8,
            'num_species': 4,
            'angular_start': 0.9,
            'radial_start': 0.9
        }
        cls_kwargs.update(kwargs)
        return cls.cover_linearly(**cls_kwargs)

    @classmethod
    def like_2x(cls, **kwargs):
        cls_kwargs = {
            'radial_cutoff': 5.1,
            'angular_cutoff': 3.5,
            'radial_eta': 19.7,
            'angular_eta': 12.5,
            'radial_dist_divisions': 16,
            'angular_dist_divisions': 8,
            'zeta': 14.1,
            'angle_sections': 4,
            'num_species': 7,
            'angular_start': 0.8,
            'radial_start': 0.8
        }
        cls_kwargs.update(kwargs)
        # note that there is a small difference of 1 digit in one decimal place
        # in the eight element of ShfR this element is 2.6812 using this method
        # and 2.6813 for the actual network, but this is not significant for
        # retraining purposes
        # In any way we change this for consistency
        out = cls.cover_linearly(**cls_kwargs)
        out.radial_terms.ShfR[0, 7] = 2.6813
        return out

    @classmethod
    def like_1ccx(cls, **kwargs):
        # just a synonym
        return cls.like_1x(**kwargs)

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

        # validate cutoffs
        assert self.neighborlist.cutoff >= self.radial_terms.get_cutoff()
        assert self.angular_terms.get_cutoff() < self.radial_terms.get_cutoff()

        if self.use_cuda_extension:
            if not self.cuaev_is_initialized:
                self._init_cuaev_computer()
                self.cuaev_is_initialized = torch.tensor(True)
            assert pbc is None or (not pbc.any()), "cuaev currently does not support PBC"
            aev = self._compute_cuaev(species, coordinates)
            return SpeciesAEV(species, aev)

        atom_index12, _, diff_vector, distances = self.neighborlist(species, coordinates, cell, pbc)
        aev = self._compute_aev(species, atom_index12, diff_vector, distances)
        return SpeciesAEV(species, aev)

    @jit_unused_if_no_cuaev()
    def _compute_cuaev(self, species, coordinates):
        species_int = species.to(torch.int32)
        coordinates = coordinates.to(torch.float)
        aev = torch.ops.cuaev.run(coordinates, species_int, self.cuaev_computer)
        return aev

    def _compute_aev(self, species: Tensor,
            atom_index12: Tensor, diff_vector: Tensor, distances: Tensor) -> Tensor:

        species12 = species.flatten()[atom_index12]
        radial_aev = self._compute_radial_aev(species.shape[0], species.shape[1], species12,
                                              distances, atom_index12)

        # Rca is usually much smaller than Rcr, using neighbor list with
        # cutoff = Rcr is a waste of resources. Now we will get a smaller neighbor
        # list that only cares about atoms with distances <= Rca
        even_closer_indices = (distances <= self.angular_terms.get_cutoff()).nonzero().flatten()
        atom_index12 = atom_index12.index_select(1, even_closer_indices)
        species12 = species12.index_select(1, even_closer_indices)
        diff_vector = diff_vector.index_select(0, even_closer_indices)

        angular_aev = self._compute_angular_aev(species.shape[0], species.shape[1], species12,
                                                diff_vector, atom_index12)

        return torch.cat([radial_aev, angular_aev], dim=-1)

    def _compute_angular_aev(self, num_molecules: int, num_atoms: int, species12: Tensor, vec: Tensor,
                             atom_index12: Tensor) -> Tensor:

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

    def _compute_radial_aev(self, num_molecules: int, num_atoms: int, species12: Tensor, distances: Tensor,
                            atom_index12: Tensor) -> Tensor:

        radial_terms_ = self.radial_terms(distances)
        radial_aev = radial_terms_.new_zeros(
            (num_molecules * num_atoms * self.num_species,
             self.radial_sublength))
        index12 = atom_index12 * self.num_species + species12.flip(0)
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
        sorted_local_index12 += cumsum_from_zero(counts).index_select(
            0, pair_indices)

        # unsort result from last part
        local_index12 = rev_indices[sorted_local_index12]

        # compute mapping between representation of central-other to pair
        n = atom_index12.shape[1]
        sign12 = ((local_index12 < n).to(torch.int8) * 2) - 1
        return central_atom_index, local_index12 % n, sign12


class InternalAEVComputer(AEVComputer):

    def __init__(self, *args, **kwargs):
        r"""AEVComputer for internal use of ANI Models"""
        assert 'neighborlist' not in kwargs.keys(), "InternalAEVComputer doesn't use a neighborlist"
        kwargs.update({'neighborlist': BaseNeighborlist(0.0)})

    def forward(self, species: Tensor,
                neighborlist: Tensor,
                diff_vectors: Tensor,
                distances: Tensor) -> SpeciesAEV:
        r"""Arguments:
                species: internal indices of the atomic species used by ani models
                neighborlist: A tensor with atom indexes, with shape (2, P)
                    where P is the number unique atom pairs. The first dimension
                    has atom (a) in index 0 and atom (b) in index 1
                difference_vectors:
                    The displacement 3-vector that points from the first neighborlist
                    atom (a) to the second atom (b), shape (2, P, 3)
                distances:
                    The magnitudes of the displacement 3-vectors, shape (2, P)"""

        aev = self._compute_aev(species, neighborlist, diff_vectors, distances)
        return SpeciesAEV(species, aev)


class AEVComputerInterfaceExternal(AEVComputer):

    assume_screened_input: Final[bool]

    def __init__(self, *args, **kwargs):
        r"""Modified AEVComputer, for interfaces

        Used to interface with pmemd-cpu or similar MD code, which provides a
        (usually not fully screened) neighborlist and cell shifts, but no
        distances / position differences """

        if 'assume_screened_input' not in kwargs:
            kwargs.update({'assume_screened_input': False})
        assert 'neighborlist' not in kwargs.keys(), "AEVComputerBare doesn't use a neighborlist"
        kwargs.update({'neighborlist': BaseNeighborlist(0.0)})
        assert not kwargs['use_cuda_extension'] or ('use_cuda_extension' not in kwargs),\
                "AEVComputerBare doesn't suport cuaev"
        kwargs.update({'use_cuda_extension': False})
        super().__init__(*args, **kwargs)
        self.assume_screened_input = kwargs['assume_screened_input']

    def forward(self, input_: Tuple[Tensor, Tensor],
                atom_index12: Optional[Tensor] = None,
                shift_values: Optional[Tensor] = None) -> SpeciesAEV:
        species, coordinates = input_
        # check shapes for correctness
        assert species.dim() == 2
        assert coordinates.dim() == 3
        assert (species.shape == coordinates.shape[:2]) and (coordinates.shape[2] == 3)

        # It is convenient to keep these arguments optional due to JIT, but
        # actually they are needed for this class
        assert atom_index12 is not None
        assert shift_values is not None

        # check consistency of shapes of neighborlist
        assert atom_index12.dim() == 2 and atom_index12.shape[0] == 2
        assert shift_values.dim() == 2 and shift_values.shape[1] == 3
        assert atom_index12.shape[1] == shift_values.shape[0]

        if not self.assume_screened_input:
            # first we screen the input neighborlist in case some of the
            # values are at distances larger than the radial cutoff, or some of
            # the values are masked with dummy atoms. The first may happen if
            # the neighborlist uses some sort of skin value to rebuild itself
            # (as in Loup Verlet lists).
            nl_out = self.neighborlist._screen_with_cutoff(self.radial_terms.get_cutoff(),
                                                           coordinates,
                                                           atom_index12,
                                                           shift_values,
                                                           (species == -1))
            atom_index12, _, diff_vec, distances = nl_out
        else:
            # if the input neighborlist is assumed to be pre screened then we
            # just calculate the distances and diff_vector here
            coordinates = coordinates.view(-1, 3)
            coords0 = coordinates.index_select(0, atom_index12[0])
            coords1 = coordinates.index_select(0, atom_index12[1])
            diff_vec = coords0 - coords1 + shift_values
            distances = diff_vec.norm(2, -1)

        aev = self._compute_aev(species, atom_index12, diff_vec, distances)
        return SpeciesAEV(species, aev)
