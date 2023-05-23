from typing import Sequence, Union, Optional

import torch
from torch import Tensor
from torch.jit import Final

from torchani.units import ANGSTROM_TO_BOHR
from torchani.utils import ATOMIC_NUMBERS
from torchani.wrappers import StandaloneWrapper
from torchani.potentials._repulsion_constants import alpha_constants, y_eff_constants
from torchani.potentials.core import PairwisePotential
from torchani.aev.neighbors import NeighborData
from torchani.aev.cutoffs import Cutoff

_ELEMENTS_NUM = len(ATOMIC_NUMBERS)


class RepulsionXTB(PairwisePotential):
    r"""Calculates the xTB repulsion energy terms for a given molecule

    Potential used is as in work by Grimme: https://pubs.acs.org/doi/10.1021/acs.jctc.8b01176
    By default alpha, y_eff and krep parameters are taken from Grimme et. al.
    pairwise_kwargs are passed to PairwisePotential
    """

    ANGSTROM_TO_BOHR: Final[float]
    y_ab: Tensor
    sqrt_alpha_ab: Tensor
    k_rep_ab: Tensor

    def __init__(
        self,
        alpha: Sequence[float] = None,
        y_eff: Sequence[float] = None,
        k_rep_ab: Optional[Tensor] = None,
        cutoff_fn: Union[str, Cutoff] = "smooth",
        **pairwise_kwargs,
    ):
        super().__init__(cutoff_fn=cutoff_fn, **pairwise_kwargs)

        if alpha is None:
            _alpha = torch.tensor(alpha_constants)[self.atomic_numbers]
        else:
            _alpha = torch.tensor(alpha)
        if y_eff is None:
            _y_eff = torch.tensor(y_eff_constants)[self.atomic_numbers]
        else:
            _y_eff = torch.tensor(y_eff)

        if k_rep_ab is None:
            k_rep_ab = torch.full((_ELEMENTS_NUM + 1, _ELEMENTS_NUM + 1), 1.5)
            k_rep_ab[1, 1] = 1.0
            k_rep_ab = k_rep_ab[self.atomic_numbers, :][:, self.atomic_numbers]

        # Validation
        num_znumbers = len(self.atomic_numbers)
        assert k_rep_ab.shape == (num_znumbers, num_znumbers)
        assert _y_eff.shape == (num_znumbers,)
        assert _alpha.shape == (num_znumbers,)

        # Pre-calculate pairwise parameters for efficiency
        self.register_buffer('y_ab', torch.outer(_y_eff, _y_eff))
        self.register_buffer('sqrt_alpha_ab', torch.sqrt(torch.outer(_alpha, _alpha)))
        self.register_buffer('k_rep_ab', k_rep_ab)
        self.ANGSTROM_TO_BOHR = ANGSTROM_TO_BOHR

    def pair_energies(
        self,
        neighbors: NeighborData,
    ) -> Tensor:
        # Clamp distances to prevent singularities when dividing by zero.
        # All internal calculations of this module are made with atomic units,
        # so distances are first converted to Bohr
        distances = torch.clamp(neighbors.distances * self.ANGSTROM_TO_BOHR, min=1e-7)

        # Find pre-computed constant multiplications for every element-idx pair
        y_ab = self.y_ab[neighbors.element_indices]
        sqrt_alpha_ab = self.sqrt_alpha_ab[neighbors.element_indices]
        k_rep_ab = self.k_rep_ab[neighbors.element_indices]

        return (y_ab / distances) * torch.exp(-sqrt_alpha_ab * (distances ** k_rep_ab))


def StandaloneRepulsionXTB(
    cutoff: float = 5.2,
    alpha: Sequence[float] = None,
    y_eff: Sequence[float] = None,
    k_rep_ab: Optional[Tensor] = None,
    symbols: Sequence[str] = ('H', 'C', 'N', 'O'),
    cutoff_fn: Union[str, Cutoff] = 'smooth',
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
