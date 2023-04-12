import math
from typing import Sequence, Union, Optional

import torch
from torch import Tensor
from torch.jit import Final

from torchani.nn import AtomicModule, IdentityAtomicModule
from torchani.aev.cutoffs import _parse_cutoff_fn
from torchani.aev.neighbors import rescreen_with_cutoff


class Potential(AtomicModule):
    r"""Base class for all atomic potentials

    Potentials may be many-body potentials or pairwise potentials
    Subclasses must override 'forward' and 'atomic_energies'
    """

    cutoff: Final[float]

    def __init__(self,
                 *args,
                 cutoff: float = math.inf,
                 symbols: Sequence[str] = ('H', 'C', 'N', 'O'),
                 name: Optional[str] = None,
                 **kwargs):
        super().__init__(symbols=symbols, cutoff=cutoff, name=name)

    def forward(
        self,
        element_idxs: Tensor,
        neighbor_idxs: Tensor,
        distances: Tensor,
        diff_vectors: Optional[Tensor] = None,
        ghost_flags: Optional[Tensor] = None
    ) -> Tensor:
        r"""
        Outputs "energy", with shape (N,)

        All distances are assumed to lie inside self.cutoff (which may be infinite).
        """
        raise NotImplementedError

    @torch.jit.export
    def atomic_energies(self,
                        element_idxs: Tensor,
                        neighbor_idxs: Tensor,
                        distances: Tensor,
                        diff_vectors: Optional[Tensor] = None,
                        ghost_flags: Optional[Tensor] = None,
                        average: bool = False,
                        ) -> Tensor:
        r"""Outputs "atomic_energies"

        All distances are assumed to lie inside self.cutoff (which may be infinite)

        'average' controls whether the atomic energies are averaged over
        the ensemble models.

        Shape is (M, N, A) if not averaged over the models,
        or (N, A) if averaged over models

        Potentials that don't have an ensemble of models output shape (1, N, A)
        if average=False.
        """
        raise NotImplementedError


class PairwisePotential(Potential):
    r"""Base class for all pairwise potentials

    Subclasses must override 'pair_energies'
    """
    def __init__(
        self,
        *args,
        cutoff: float = math.inf,
        symbols: Sequence[str] = ('H', 'C', 'N', 'O'),
        cutoff_fn: Union[str, torch.nn.Module] = 'smooth',
        name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(cutoff=cutoff, symbols=symbols, name=name)
        self.cutoff_fn = _parse_cutoff_fn(cutoff_fn)

    def raw_pair_energies(
        self,
        element_idxs: Tensor,
        neighbor_idxs: Tensor,
        distances: Tensor,
        diff_vectors: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Calculate the raw (non-smoothed with cutoff function) energy of all pairs of neighbors

        element_idxs is shape (N, A)
        neighbor_idxs is shape (P,)
        distances is shape (P,)
        diff_vectors is shape (P, 3)

        This must return a tensor of shape (P,)"""
        raise NotImplementedError

    # This function wraps raw_pair_energies
    # It smooths out the energies using a cutoff function,
    # and it scales pair energies of ghost atoms by 1/2
    def pair_energies(
        self,
        element_idxs: Tensor,
        neighbor_idxs: Tensor,
        distances: Tensor,
        diff_vectors: Optional[Tensor] = None,
        ghost_flags: Optional[Tensor] = None
    ) -> Tensor:
        # Validation
        assert distances.ndim == 1, "distances should be 1 dimensional"
        assert element_idxs.ndim == 2, "species should be 2 dimensional"
        assert neighbor_idxs.ndim == 2, "atom_index12 should be 2 dimensional"
        assert len(distances) == neighbor_idxs.shape[1]

        pair_energies = self.raw_pair_energies(
            element_idxs,
            neighbor_idxs,
            distances,
            diff_vectors,
        )
        assert pair_energies.shape == neighbor_idxs.shape, "raw_pair_energies must return a tensor with the same shape as neighbor_idxs"

        if self.cutoff_fn is not None:
            pair_energies *= self.cutoff_fn(
                distances,
                self.cutoff
            )

        if ghost_flags is not None:
            assert ghost_flags.numel() == element_idxs.numel(), "ghost_flags and species should have the same number of elements"
            ghost12 = ghost_flags.flatten()[neighbor_idxs]
            ghost_mask = torch.logical_or(ghost12[0], ghost12[1])
            pair_energies = torch.where(ghost_mask, pair_energies * 0.5, pair_energies)
        return pair_energies

    def forward(
        self,
        element_idxs: Tensor,
        neighbor_idxs: Tensor,
        distances: Tensor,
        diff_vectors: Optional[Tensor] = None,
        ghost_flags: Optional[Tensor] = None
    ) -> Tensor:
        molecules_num = element_idxs.shape[0]
        pair_energies = self.pair_energies(
            element_idxs,
            neighbor_idxs,
            distances,
            diff_vectors,
            ghost_flags,
        )
        energies = torch.zeros(
            molecules_num,
            dtype=pair_energies.dtype,
            device=pair_energies.device
        )
        molecule_indices = torch.div(neighbor_idxs[0], molecules_num, rounding_mode='floor')
        energies.index_add_(0, molecule_indices, pair_energies)
        return energies

    @torch.jit.export
    def atomic_energies(
        self,
        element_idxs: Tensor,
        neighbor_idxs: Tensor,
        distances: Tensor,
        diff_vectors: Optional[Tensor] = None,
        ghost_flags: Optional[Tensor] = None,
        average: bool = False,
    ) -> Tensor:
        pair_energies = self.pair_energies(
            element_idxs,
            neighbor_idxs,
            distances,
            diff_vectors,
            ghost_flags,
        )
        molecules_num = element_idxs.shape[0]
        atoms_num = element_idxs.shape[1]

        atomic_energies = torch.zeros(
            molecules_num * atoms_num,
            dtype=pair_energies.dtype,
            device=pair_energies.device
        )
        atomic_energies.index_add_(0, neighbor_idxs[0], pair_energies / 2)
        atomic_energies.index_add_(0, neighbor_idxs[1], pair_energies / 2)
        atomic_energies = atomic_energies.view(molecules_num, atoms_num)
        if not average:
            return atomic_energies.unsqueeze(0)
        return atomic_energies


class ScaledPairwisePotential(PairwisePotential):
    def __init__(
        self,
        *args,
        cutoff: float = math.inf,
        symbols: Sequence[str] = ('H', 'C', 'N', 'O'),
        cutoff_fn: Union[str, torch.nn.Module] = 'smooth',
        name: Optional[str] = None,
        scaler: Optional[AtomicModule] = None,
        **kwargs,
    ):
        super().__init__(cutoff=cutoff, symbols=symbols, name=name, cutoff_fn=cutoff_fn)
        if scaler is None:
            scaler = IdentityAtomicModule()
        self.scaler = scaler
        self.scaler.name = name

    def scale_with_factors(
        self,
        factors: Tensor,
        neighbor_idxs: Tensor,
        unscaled_pair_energies: Tensor,
    ) -> Tensor:
        # by default the scaling is
        # F_ij = f_i * f_j
        # E_ij = F_ij * U_ij
        # shape of factors is (N, A,)
        # shape of neighbor_idxs is (2, P)
        # shape of pair_energies is (P,)
        return factors.flatten()[neighbor_idxs].prod(0) * unscaled_pair_energies

    def pair_energies(
        self,
        element_idxs: Tensor,
        neighbor_idxs: Tensor,
        distances: Tensor,
        diff_vectors: Optional[Tensor] = None,
        ghost_flags: Optional[Tensor] = None
    ) -> Tensor:
        unscaled_pair_energies = super().pair_energies(
            element_idxs,
            neighbor_idxs,
            distances,
            diff_vectors,
            ghost_flags,
        )
        if self.scaler.cutoff < self.cutoff:
            neighbor_data = rescreen_with_cutoff(
                self.scaler.cutoff,
                neighbor_idxs,
                distances,
                diff_vectors,
            )
            neighbor_idxs = neighbor_data.neighbor_idxs
            distances = neighbor_data.distances
            diff_vectors = neighbor_data.diff_vectors
        factors = self.scaler(
            element_idxs,
            neighbor_idxs,
            distances,
            diff_vectors,
            ghost_flags,
        )
        return self.scale_with_factors(factors, neighbor_idxs, unscaled_pair_energies)
