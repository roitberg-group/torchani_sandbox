import typing as tp

import torch
from torch import Tensor
from torch.jit import Final

from torchani.tuples import SpeciesAEV
from torchani.utils import cumsum_from_zero
from torchani.neighbors import _parse_neighborlist
from torchani.cutoffs import _parse_cutoff_fn
from torchani.aev.aev_terms import (
    _parse_angular_terms,
    _parse_radial_terms,
)


class ExCorrAEVComputer(torch.nn.Module):
    num_species: Final[int]
    num_species_pairs: Final[int]

    angular_length: Final[int]
    angular_sublength: Final[int]
    radial_length: Final[int]
    radial_sublength: Final[int]
    aev_length: Final[int]

    triu_index: Tensor

    def __init__(
        self,
        Rcr: tp.Optional[float] = None,
        Rca: tp.Optional[float] = None,
        EtaR: tp.Optional[Tensor] = None,
        ShfR: tp.Optional[Tensor] = None,
        EtaA: tp.Optional[Tensor] = None,
        Zeta: tp.Optional[Tensor] = None,
        ShfA: tp.Optional[Tensor] = None,
        ShfZ: tp.Optional[Tensor] = None,
        # New init:
        num_species: tp.Optional[int] = None,
        cutoff_fn='cosine',
        neighborlist='full_pairwise',
        radial_terms='standard',
        angular_terms='standard',
    ):

        # due to legacy reasons num_species is a kwarg, but it should always be
        # provided
        assert num_species is not None, "num_species should be provided to construct an AEVComputer"

        super().__init__()
        self.num_species = num_species
        self.num_species_pairs = num_species * (num_species + 1) // 2

        # currently only cosine, smooth and custom cutoffs are supported
        # only ANI-1 style angular terms or radial terms
        # and only full pairwise neighborlist
        # if a cutoff function is passed, it is used for both radial and
        # angular terms.
        cutoff_fn = _parse_cutoff_fn(cutoff_fn)
        self.angular_terms = _parse_angular_terms(angular_terms, cutoff_fn, EtaA, Zeta, ShfA, ShfZ, Rca)
        self.radial_terms = _parse_radial_terms(radial_terms, cutoff_fn, EtaR, ShfR, Rcr)
        self.neighborlist = _parse_neighborlist(neighborlist, self.radial_terms.cutoff)

        self._validate_cutoffs_init()

        self.register_buffer('triu_index',
                             self._calculate_triu_index(num_species).to(device=self.radial_terms.EtaR.device))

        # length variables are updated once radial and angular terms are initialized
        # The lengths of buffers can't be changed with load_state_dict so we can
        # cache all lengths in the model itself
        self.radial_sublength = self.radial_terms.sublength
        self.angular_sublength = self.angular_terms.sublength
        self.radial_length = self.radial_sublength * self.num_species
        self.angular_length = self.angular_sublength * self.num_species_pairs
        self.aev_length = self.radial_length + self.angular_length

    def _validate_cutoffs_init(self):
        # validate cutoffs and emit warnings for strange configurations
        if self.neighborlist.cutoff > self.radial_terms.cutoff:
            raise ValueError(f"""The neighborlist cutoff {self.neighborlist.cutoff}
                    is larger than the radial cutoff,
                    {self.radial_terms.cutoff}.  please fix this since
                    AEVComputer can't possibly reuse the neighborlist for other
                    interactions, so this configuration would not use the extra
                    atom pairs""")
        elif self.neighborlist.cutoff < self.radial_terms.cutoff:
            raise ValueError(f"""The neighborlist cutoff,
                             {self.neighborlist.cutoff} should be at least as
                             large as the radial cutoff, {self.radial_terms.cutoff}""")
        if self.angular_terms.cutoff > self.radial_terms.cutoff:
            raise ValueError(f"""Current implementation assumes angular cutoff
                             {self.angular_terms.cutoff} < radial cutoff
                             {self.radial_terms.cutoff}""")

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

    def forward(
        self,
        input_: tp.Tuple[Tensor, Tensor],  # species, coordinates
        coefficients: Tensor,
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
    ) -> SpeciesAEV:
        species, coordinates = input_
        # check shapes for correctness
        assert species.dim() == 2
        assert coordinates.dim() == 3
        assert (species.shape == coordinates.shape[:2]) and (coordinates.shape[2] == 3)
        # TODO: check that coefficients have the correct shape

        # validate cutoffs
        assert self.neighborlist.cutoff >= self.radial_terms.cutoff
        assert self.angular_terms.cutoff < self.radial_terms.cutoff

        # WARNING: The coordinates that are input into the neighborlist are **not** assumed to be
        # mapped into the central cell for pbc calculations,
        # and **in general are not**
        neighbor_data = self.neighborlist(species, coordinates, cell, pbc)
        # Neighborlist data has:
        # data.neighbors  shape (2, pares)
        # data.distances  (pares,)
        # data.diff_vectors (pares, 3)
        aev = self._compute_aev(
            element_idxs=species,
            neighbor_idxs=neighbor_data.indices,
            distances=neighbor_data.distances,
            diff_vectors=neighbor_data.diff_vectors,
        )
        return SpeciesAEV(species, aev)

    def _compute_aev(
        self,
        element_idxs: Tensor,  # for H -> 0, for C -> 1, for O -> 2
        # this corresponds to atoms
        neighbor_idxs: Tensor,  # all with all but there are issues with s orbitals =S
        distances: Tensor,
        diff_vectors: Tensor
        # You would need smth like this but for the "AOV"
    ) -> Tensor:
        num_molecules = element_idxs.shape[0]
        num_atoms = element_idxs.shape[1]
        species12 = element_idxs.flatten()[neighbor_idxs]

        radial_aev = self._compute_radial_aev(
            num_molecules,
            num_atoms,
            species12,
            neighbor_idxs=neighbor_idxs,
            distances=distances
        )

        # Rca is usually much smaller than Rcr, using neighbor list with
        # cutoff = Rcr is a waste of resources. Now we will get a smaller neighbor
        # list that only cares about atoms with distances <= Rca
        even_closer_indices = (distances <= self.angular_terms.cutoff).nonzero().flatten()
        neighbor_idxs = neighbor_idxs.index_select(1, even_closer_indices)
        species12 = species12.index_select(1, even_closer_indices)
        diff_vectors = diff_vectors.index_select(0, even_closer_indices)

        angular_aev = self._compute_angular_aev(
            num_molecules,
            num_atoms,
            species12,
            neighbor_idxs=neighbor_idxs,
            diff_vectors=diff_vectors
        )
        # Your featurizer should spit this + coeff-features
        # return torch.cat([radial_aev, angular_aev, coeff_aev_radial, coeff_aev_angular], dim=-1)
        # coords shape is (conformer, atoms, 3)
        # aev shape is (conformer, atoms, aev-dim)
        # coeff_aev shape is the same (conformer, atoms, coeff-aev-dim)
        return torch.cat([radial_aev, angular_aev], dim=-1)

    def _compute_radial_aev(
        self,
        num_molecules: int,
        num_atoms: int,
        species12: Tensor,
        neighbor_idxs: Tensor,
        distances: Tensor,
    ) -> Tensor:
        #  pairs: all pairs in all the molecules in the batch
        radial_terms_ = self.radial_terms(distances)  # shape (pairs, radial-aev-dim)
        radial_aev = radial_terms_.new_zeros(
            (num_molecules * num_atoms * self.num_species,
             self.radial_sublength))

        # Assembly of the radial aev is different
        index12 = neighbor_idxs * self.num_species + species12.flip(0)
        radial_aev.index_add_(0, index12[0], radial_terms_)
        radial_aev.index_add_(0, index12[1], radial_terms_)
        radial_aev = radial_aev.reshape(num_molecules, num_atoms, self.radial_length)
        return radial_aev

    def _compute_angular_aev(
        self,
        num_molecules: int,
        num_atoms: int,
        species12: Tensor,
        neighbor_idxs: Tensor,
        diff_vectors: Tensor,
    ) -> Tensor:

        central_atom_index, pair_index12, sign12 = self._triple_by_molecule(
            neighbor_idxs)
        species12_small = species12[:, pair_index12]
        vec12 = diff_vectors.index_select(0, pair_index12.view(-1)).view(
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

    def _triple_by_molecule(
            self, atom_index12: Tensor) -> tp.Tuple[Tensor, Tensor, Tensor]:
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
        pair_sizes = (counts * (counts - 1)).div(2, rounding_mode='floor')
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
