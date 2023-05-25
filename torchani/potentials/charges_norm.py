r"""Charge normalization utilities"""
from enum import Enum
import typing as tp

import torch
from torch import Tensor

from torchani.utils import ATOMIC_NUMBERS


class EqualChargeFactor(torch.nn.Module):
    def forward(
        self,
        element_idxs: Tensor,
        raw_charges: Tensor,
    ) -> Tensor:
        num_charges = (element_idxs != -1).sum(-1)
        return 1 / num_charges


ELECTRONEGATIVITY = {
    "H": 7.18,
    "C": 6.26,
    "N": 7.27,
    "O": 7.54,
    "S": 6.22,
    "F": 10.41,
    "Cl": 8.29,
}
HARDNESS = {
    "H": 12.84,
    "C": 10.00,
    "N": 14.53,
    "O": 12.16,
    "S": 8.28,
    "F": 14.02,
    "Cl": 9.35,
}


class WeightedChargeFactor(torch.nn.Module):
    weights: Tensor

    def __init__(
        self,
        electronegativities: tp.Optional[tp.Sequence[float]] = None,
        hardnesses: tp.Optional[tp.Sequence[float]] = None,
        symbols: tp.Sequence[str] = ("H", "C", "N", "O"),
    ):
        super().__init__()
        self.atomic_numbers = torch.tensor(
            [ATOMIC_NUMBERS[e] for e in symbols], dtype=torch.long
        )
        if electronegativities is None:
            electronegativities = [ELECTRONEGATIVITY[s] for s in symbols]
        assert electronegativities is not None  # mypy
        if hardnesses is None:
            hardnesses = [HARDNESS[s] for s in symbols]
        assert hardnesses is not None  # mypy

        self.register_buffer(
            "weights",
            (torch.tensor(electronegativities) / torch.tensor(hardnesses)) ** 2,
        )
        assert len(self.weights) == len(self.atomic_numbers)

    def forward(
        self,
        element_idxs: Tensor,
        raw_charges: Tensor,
    ) -> Tensor:
        weights = self.weights[element_idxs]
        weights = weights.masked_fill(element_idxs == -1, 0.0)
        return weights / torch.sum(weights, dim=-1, keepdim=True)


class SquaredWeightedChargeFactor(WeightedChargeFactor):
    def forward(
        self,
        element_idxs: Tensor,
        raw_charges: Tensor,
    ) -> Tensor:
        weights = self.weights[element_idxs]
        weights = weights.masked_fill(element_idxs == -1, 0.0)
        weights = weights * raw_charges**2
        return weights / torch.sum(weights, dim=-1, keepdim=True)


class ChargeFactor(Enum):
    EQUAL = EqualChargeFactor
    WEIGHTED = WeightedChargeFactor
    SQUARED_WEIGHTED = SquaredWeightedChargeFactor


def _parse_factor(
    factor: tp.Union[ChargeFactor, torch.nn.Module],
    kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
) -> torch.nn.Module:
    if isinstance(factor, ChargeFactor):
        assert kwargs is not None  # mypy
        return factor.value(**kwargs)
    assert kwargs is None  # mypy
    return factor


class ChargeNormalizer(torch.nn.Module):
    r"""
    Usage:

    .. code-block::python

        normalizer = ChargeNorm()
        total_charge = 0.0
        norm_charges = normalizer(species, raw_charges, total_charge)

    """

    def __init__(
        self,
        factor: tp.Union[ChargeFactor, torch.nn.Module] = ChargeFactor.EQUAL,
        factor_args: tp.Optional[tp.Dict[str, tp.Any]] = None,
    ):
        super().__init__()
        self.factor = _parse_factor(factor, factor_args or {})

    def forward(
        self, element_idxs: Tensor, raw_charges: Tensor, total_charge: float = 0.0
    ) -> Tensor:
        total_raw_charge = torch.sum(raw_charges, dim=-1, keepdim=True)
        charge_excess = total_charge - total_raw_charge
        factor = self.factor(element_idxs, raw_charges)
        return raw_charges + charge_excess * factor
