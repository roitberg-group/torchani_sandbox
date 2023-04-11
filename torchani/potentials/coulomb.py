from typing import Sequence, Union, Optional

import torch
from torch import Tensor
from torch.jit import Final

from torchani.units import ANGSTROM_TO_BOHR
from torchani.utils import ATOMIC_NUMBERS
from torchani.wrappers import StandaloneWrapper
from torchani.potentials._repulsion_constants import alpha_constants, y_eff_constants
from torchani.potentials.core import PairwisePotential

_ELEMENTS_NUM = len(ATOMIC_NUMBERS)


class ChargeNormalization(torch.nn.Module):

    def __init__(self):
        pass

    def forward(
        self,
        element_idxs: Tensor,
        raw_charges: Tensor,
    ) -> Tensor:
        raw_charges = raw_charges.masked_fill(element_idxs == -1)
        total_raw_charge = torch.sum(raw_charges, dim=-1)
        num_charges = raw_charges
        return raw_charges


class AtomicCharges(torch.nn.Module):
    r"""Base Module that calculates atomic charges

    Constant base charges are assigned to each element and
    then, potentially, a trainable or non-trainable correction
    is calculated on forward.

    By default the base charges are all zero

    Optionally, the base charges can be forced to add up to a given number
    exactly, by passing a charge normalization function

    The normalization function must map a tensor of charges to a tensor
    of normalized charges, which fulfill sum(normalized_charges) = 0
    """

    def __init__(
        self,
        base_charges: Sequence[float] = None,
        normalization_fn: Optional[torch.nn.Module] = None,
    ):
        # Pre-calculate pairwise parameters for efficiency
        if base_charges is not None:
            _charges = torch.tensor(base_charges)
            base_charge_factors = torch.outer(_charges, _charges)
        else:
            charge_factors = None

    def raw_charges(
        self,
        element_idxs: Tensor,
        neighbor_idxs: Tensor,
        distances: Tensor,
        diff_vectors: Optional[Tensor] = None
    ) -> Tensor:
        r"""Output unnormalized charges"""
        return None

    def charges(
        self,
        element_idxs: Tensor,
        neighbor_idxs: Tensor,
        distances: Tensor,
        diff_vectors: Optional[Tensor] = None,
        total_charge: Optional[float] = 0.0,
    ) -> Tensor:
        r"""Output normalized charges"""
        raw_charges = self.raw_charges(
            element_idxs=element_idxs,
            neighbor_idxs=neighbor_idxs,
            distances=distances,
            diff_vectors=diff_vectors,
        )
        return self.normalization_fn(raw_charges)

    def raw_products(
        self,
        element_idxs: Tensor,
        neighbor_idxs: Tensor,
        distances: Tensor,
        diff_vectors: Optional[Tensor] = None
    ) -> Tensor:
        r"""Output all pairwise products of unnormalized charges"""
        raw_charges = self.raw_charges(
            element_idxs=element_idxs,
            neighbor_idxs=neighbor_idxs,
            distances=distances,
            diff_vectors=diff_vectors,
        )
        return raw_charges[:, neighbor_idxs[0]] * raw_charges[:, neighbor_idxs[1]]

    def products(
        self,
        element_idxs: Tensor,
        neighbor_idxs: Tensor,
        distances: Tensor,
        diff_vectors: Optional[Tensor] = None
    ) -> Tensor:
        r"""Output all pairwise products of normalized charges

        Note: Directly calling the module will call this function under the hood,
        you should call the module directly to be able to
        better interface with torch hooks, etc."""
        charges = self.charges(
            element_idxs=element_idxs,
            neighbor_idxs=neighbor_idxs,
            distances=distances,
            diff_vectors=diff_vectors,
        )
        return charges[:, neighbor_idxs[0]] * charges[:, neighbor_idxs[1]]

    def forward(
        self,
        element_idxs: Tensor,
        neighbor_idxs: Tensor,
        distances: Tensor,
        diff_vectors: Optional[Tensor] = None
    ) -> Tensor:
        return self.products(
            element_idxs=element_idxs,
            neighbor_idxs=neighbor_idxs,
            distances=distances,
            diff_vectors=diff_vectors,
        )


class Coulomb(PairwisePotential):
    r"""Calculates the coulombic interaction energy given some charges

    pairwise_kwargs are passed to PairwisePotential
    """

    ANGSTROM_TO_BOHR: Final[float]
    charges: Tensor

    def __init__(
        self,
        atomic_charges: AtomicCharges,
        **pairwise_kwargs,
    ):
        super().__init__(**pairwise_kwargs)
        # Override the cutoff of the charge calculator with the cutoff of the
        # potential
        atomic_charges.cutoff = self.cutoff
        self._atomic_charges = atomic_charges

    def pair_energies(
        self,
        element_idxs: Tensor,
        neighbor_idxs: Tensor,
        distances: Tensor,
        diff_vectors: Optional[Tensor] = None,
    ) -> Tensor:

        # Clamp distances to prevent singularities when dividing by zero
        distances = torch.clamp(distances, min=1e-7)

        # All internal calculations of this module are made with atomic units,
        # so distances are first converted to bohr
        distances = distances * self.ANGSTROM_TO_BOHR

        charge_products = self._atomic_charges.products(
            element_idxs,
            neighbor_idxs,
            distances,
            diff_vectors
        )
        return charge_producs / distances


def StandaloneCoulomb(
    cutoff: float = 5.2,
    alpha: Sequence[float] = None,
    y_eff: Sequence[float] = None,
    k_rep_ab: Optional[Tensor] = None,
    symbols: Sequence[str] = ('H', 'C', 'N', 'O'),
    cutoff_fn: Union[str, torch.nn.Module] = 'smooth',
    **standalone_kwargs,
) -> StandaloneWrapper:
    module = RepulsionXTB(
        alpha=alpha,
        y_eff=y_eff,
        k_rep_ab=k_rep_ab,
        cutoff=cutoff,
        symbols=symbols,
        cutoff_fn=cutoff_fn
    )
    return StandaloneWrapper(module, **standalone_kwargs)
