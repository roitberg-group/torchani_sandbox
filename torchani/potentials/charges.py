from typing import Sequence, Optional

import torch
from torch import Tensor

from torchani.potentials.core import AtomicModule


class ChargeNorm(AtomicModule):
    r"""Subclasses will probably want to override 'factor'"""
    def factor(
        self,
        element_idxs: Tensor,
        raw_charges: Tensor,
    ) -> Tensor:
        num_charges = (element_idxs != -1).sum(-1)
        return 1 / num_charges

    def forward(
        self,
        element_idxs: Tensor,
        raw_charges: Tensor,
        total_charge: float = 0.0
    ) -> Tensor:
        total_raw_charge = torch.sum(raw_charges, dim=-1, keepdim=True)
        charge_excess = total_charge - total_raw_charge
        factor = self.factor(element_idxs, raw_charges)
        return raw_charges + charge_excess * factor


class WeightedChargeNorm(ChargeNorm):
    def __init__(
        self,
        electronegativities: Sequence[float],
        hardnesses: Sequence[float],
        symbols: Sequence[str] = ("H", "C", "N", "O"),
    ):
        super().__init__(symbols=symbols)
        _electronegativities = torch.tensor(electronegativities)
        _hardnesses = torch.tensor(hardnesses)
        self.register_buffer("weights", (_electronegativities / _hardnesses) ** 2)
        assert len(self.weights) == len(self.atomic_numbers)

    def factor(
        self,
        element_idxs: Tensor,
        raw_charges: Tensor,
    ) -> Tensor:
        weights = self.weights[element_idxs]
        weights = weights.masked_fill(element_idxs == -1, 0.)
        return weights / torch.sum(weights, dim=-1, keepdim=True)


class WeightedQ2ChargeNorm(WeightedChargeNorm):
    def factor(
        self,
        element_idxs: Tensor,
        raw_charges: Tensor,
    ) -> Tensor:
        weights = self.weights[element_idxs]
        weights = weights.masked_fill(element_idxs == -1, 0.)
        weights = weights * raw_charges ** 2
        return weights / torch.sum(weights, dim=-1, keepdim=True)


class AtomicChargeScaler(AtomicModule):
    r"""Module that calculates atomic charges

    Constant base charges are assigned to each element and
    then, a trainable (or non-trainable) correction
    is calculated on forward.

    By default the base charges are all zero.

    Optionally, the output charges can be forced to add up to a given number
    exactly, by passing a charge normalization function.

    The normalization function must map a tensor of charges to a tensor
    of normalized charges, which fulfill sum(normalized_charges) = total_charge

    In summary, the module computes:

    output_charges = normalization(base_charges + correction(structure), total_charge)

    Most sublcasses should only override 'unshifted_raw_charges'
    """
    def __init__(
        self,
        base_charges: Sequence[float] = None,
        symbols: Sequence[str] = ("H", "C", "N", "O"),
        normalization_fn: Optional[ChargeNorm] = None,
        **kwargs
    ):
        super().__init__(symbols=symbols, **kwargs)
        if normalization_fn is not None:
            self.normalization_fn = normalization_fn
            assert self.normalization_fn.atomic_numbers == self.atomic_numbers
        else:
            self.normalization_fn = ChargeNorm(symbols=symbols)

        if base_charges is not None:
            base_charges = torch.tensor(base_charges)
            assert len(base_charges) == len(self.atomic_numbers)
        else:
            base_charges = torch.tensor(0.)
        torch.register_buffer("base_charges", base_charges)

    def unshifted_raw_charges(
        self,
        element_idxs: Tensor,
        neighbor_idxs: Tensor,
        distances: Tensor,
        diff_vectors: Optional[Tensor] = None
    ) -> Tensor:
        r"""Output unshifted, unnormalized charges"""
        return torch.zeros(
            element_idxs.shape,
            dtype=distances.dtype,
            device=distances.device,
        )

    def raw_charges(
        self,
        element_idxs: Tensor,
        neighbor_idxs: Tensor,
        distances: Tensor,
        diff_vectors: Optional[Tensor] = None
    ) -> Tensor:
        r"""Output unnormalized charges"""
        return self.base_charges + self.unshifted_raw_charges(
            element_idxs=element_idxs,
            neighbor_idxs=neighbor_idxs,
            distances=distances,
            diff_vectors=diff_vectors,
        )

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
