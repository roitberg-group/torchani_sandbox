from typing import Sequence, Union, Dict

import torch
from torch import Tensor

from torchani.nn import AtomicModule
from torchani.aev.cutoffs import _parse_cutoff_fn
from torchani.neighbors import rescreen_with_cutoff, NeighborData


class AtomicScalars(AtomicModule):
    r"""Module that calculates scalar quantities"""
    def __init__(
        self,
        *args,
        scalars: Sequence[str] = ("energies",),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.scalars_num = len(scalars)
        self.scalars = scalars

    def forward(
        self,
        element_idxs: Tensor,
        neighbor_data: NeighborData,
    ) -> Dict[str, Tensor]:
        r"""Outputs "{scalar_name: scalar}", with each scalar having shape (N,)

        All distances are assumed to lie inside self.cutoff (which may be infinite).
        """
        out: Dict[str, Tensor] = dict()
        for k in self.scalars:
            out[k] = torch.sum(self.atomic(element_idxs, neighbor_data), dim=-1)
        return out

    @torch.jit.export
    def atomic(
        self,
        element_idxs: Tensor,
        neighbor_data: NeighborData,
        average: bool = False,
    ) -> Dict[str, Tensor]:
        r"""Outputs "atomic scalars {scalar_name: scalar}"

        All distances are assumed to lie inside self.cutoff (which may be infinite)

        'average' controls whether the atomic scalars are averaged over
        the ensemble models.

        Shape is (M, N, A) for each scalar if not averaged over the models,
        or (N, A) if averaged over models

        Scalars that don't have an ensemble of models have output shape (1, N, A)
        if average=False.
        """
        raise NotImplementedError


class AtomicScalarsFromAEV(AtomicModule):
    r"""Module that calculates scalar quantities from atomic environment vectors (AEV)"""
    def __init__(
        self,
        *args,
        scalars: Sequence[str] = ("energies",),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.scalars_num = len(scalars)
        self.scalars = scalars

    def forward(
        self,
        element_idxs: Tensor,
        aevs: Tensor,
    ) -> Dict[str, Tensor]:
        r"""Outputs "{scalar_name: scalar}", with each scalar having shape (N,)

        All distances are assumed to lie inside self.cutoff (which may be infinite).
        """
        out: Dict[str, Tensor] = dict()
        for k in self.scalars:
            out[k] = torch.sum(self.atomic(element_idxs, aevs), dim=-1)
        return out

    @torch.jit.export
    def atomic(
        self,
        element_idxs: Tensor,
        aevs: Tensor,
        average: bool = False,
    ) -> Dict[str, Tensor]:
        r"""Outputs "atomic scalars {scalar_name: scalar}"

        All distances are assumed to lie inside self.cutoff (which may be infinite)

        'average' controls whether the atomic scalars are averaged over
        the ensemble models.

        Shape is (M, N, A) for each scalar if not averaged over the models,
        or (N, A) if averaged over models

        Scalars that don't have an ensemble of models have output shape (1, N, A)
        if average=False.
        """
        raise NotImplementedError


class NetworkAtomicScalarsFromAEV(AtomicScalarsFromAEV):
    r"""Module that calculates scalar quantities from atomic feature vectors (AEV)"""
    def __init__(
        self,
        networks: Dict[str, torch.nn.Module],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.networks = torch.nn.ModuleDict(networks)

    @torch.jit.export
    def atomic(
        self,
        element_idxs: Tensor,
        aevs: Tensor,
    ) -> Dict[str, Tensor]:
        # Obtain the atomic energies associated with a given tensor of AEV's
        assert element_idxs.shape == aevs.shape[:-1]
        molecule_num = element_idxs.shape[0]
        atoms_num = element_idxs.shape[1]
        element_idxs_ = element_idxs.flatten()
        aevs = aevs.flatten(0, 1)

        output = aevs.new_zeros()
        output = torch.zeros(
            (molecule_num * atoms_num, self.scalars_num),
            dtype=aevs.dtype,
            device=aevs.device
        )
        for j, nn in enumerate(self.networks.values()):
            mask_idx = (element_idxs_ == j).nonzero().view(-1)
            if mask_idx.shape[0] > 0:
                input_ = aevs.index_select(0, mask_idx)
                output.index_add_(0, mask_idx, nn(input_).view(-1, self.scalars_num))
        output = output.view(molecule_num, atoms_num, self.scalars_num)
        out: Dict[str, Tensor] = dict()
        for key, scalar in zip(self.scalars, output.unbind(-1)):
            out[key] = scalar
        return out


class Potential(AtomicModule):
    r"""Base class for all atomic potentials

    Potentials may be many-body potentials or pairwise potentials
    Subclasses must override 'forward' and 'atomic_energies'
    """
    def forward(
        self,
        element_idxs: Tensor,
        neighbor_data: NeighborData,
    ) -> Tensor:
        r"""
        Outputs "energy", with shape (N,)

        All distances are assumed to lie inside self.cutoff (which may be infinite).
        """
        raise NotImplementedError

    @torch.jit.export
    def atomic_energies(self,
                        element_idxs: Tensor,
                        neighbor_data: NeighborData,
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
        cutoff_fn: Union[str, torch.nn.Module] = 'smooth',
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.cutoff_fn = _parse_cutoff_fn(cutoff_fn)

    def raw_pair_energies(
        self,
        element_idxs: Tensor,
        neighbor_data: NeighborData,
    ) -> Tensor:
        r"""Calculate the raw (non-smoothed with cutoff function) energy of all pairs of neighbors

        element_idxs is shape (N, A)

        neighbor data is a named tuple that has:
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
        neighbor_data: NeighborData,
    ) -> Tensor:
        assert element_idxs.ndim == 2, "species should be 2 dimensional"

        # Validation of neighbor data
        distances = neighbor_data.distances
        diff_vectors = neighbor_data.diff_vectors
        neighbor_idxs = neighbor_data.indices
        assert distances.ndim == 1, "distances should be 1 dimensional"
        assert neighbor_idxs.ndim == 2, "neighbor idxs should be 2 dimensional"
        assert diff_vectors.ndim == 2, "diff vecotrs must be 2 dimensional"
        assert len(distances) == neighbor_idxs.shape[1]

        pair_energies = self.raw_pair_energies(
            element_idxs,
            neighbor_data,
        )
        assert pair_energies.shape == neighbor_idxs.shape, "raw_pair_energies must return a tensor with the same shape as neighbor_idxs"

        if self.cutoff_fn is not None:
            pair_energies *= self.cutoff_fn(distances, self.cutoff)
        ghost_flags = neighbor_data.ghost_flags
        if ghost_flags is not None:
            assert ghost_flags.numel() == element_idxs.numel(), "ghost_flags and species should have the same number of elements"
            ghost12 = ghost_flags.flatten()[neighbor_idxs]
            ghost_mask = torch.logical_or(ghost12[0], ghost12[1])
            pair_energies = torch.where(ghost_mask, pair_energies * 0.5, pair_energies)
        return pair_energies

    def forward(
        self,
        element_idxs: Tensor,
        neighbor_data: NeighborData,
    ) -> Tensor:
        molecules_num = element_idxs.shape[0]
        pair_energies = self.pair_energies(
            element_idxs,
            neighbor_data,
        )
        energies = torch.zeros(
            molecules_num,
            dtype=pair_energies.dtype,
            device=pair_energies.device
        )
        molecule_indices = torch.div(neighbor_data.indices[0], molecules_num, rounding_mode='floor')
        energies.index_add_(0, molecule_indices, pair_energies)
        return energies

    @torch.jit.export
    def atomic_energies(
        self,
        element_idxs: Tensor,
        neighbor_data: NeighborData,
        average: bool = False,
    ) -> Tensor:
        pair_energies = self.pair_energies(
            element_idxs,
            neighbor_data,
        )
        molecules_num = element_idxs.shape[0]
        atoms_num = element_idxs.shape[1]

        atomic_energies = torch.zeros(
            molecules_num * atoms_num,
            dtype=pair_energies.dtype,
            device=pair_energies.device
        )
        atomic_energies.index_add_(0, neighbor_data.indices[0], pair_energies / 2)
        atomic_energies.index_add_(0, neighbor_data.indices[1], pair_energies / 2)
        atomic_energies = atomic_energies.view(molecules_num, atoms_num)
        if not average:
            return atomic_energies.unsqueeze(0)
        return atomic_energies


class ScaledPairwisePotential(PairwisePotential):
    def __init__(
        self,
        scaler: AtomicModule,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.scaler = scaler

    def pair_energies(
        self,
        element_idxs: Tensor,
        neighbor_data: NeighborData,
    ) -> Tensor:
        unscaled_pair_energies = super().pair_energies(
            element_idxs,
            neighbor_data,
        )
        if self.scaler.cutoff < self.cutoff:
            neighbor_data = rescreen_with_cutoff(self.scaler.cutoff, neighbor_data)
        return self.scaler(unscaled_pair_energies, element_idxs, neighbor_data)
