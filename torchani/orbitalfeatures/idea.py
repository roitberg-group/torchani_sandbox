import typing as tp
import torch
from torch import Tensor

from torchani.aev import AEVComputer
from torchani.aev.aev_terms import StandardAngular, StandardRadial


class ExCorrAEVComputerVariation(torch.nn.Module):
    def __init__(self, aev_computer) -> None:
        self._geometric_aev = aev_computer

    def forward(
        self,
        species_coords: tp.Tuple[Tensor, Tensor],
        coefficients: Tensor,
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
    ) -> tp.Tuple[Tensor, Tensor]:
        species, coords = species_coords

        _, aevs = self._geometric_aev(species_coords, cell, pbc)

        # Here you do your own stuff with the coeffs etc
        return species, torch.tensor(0)
