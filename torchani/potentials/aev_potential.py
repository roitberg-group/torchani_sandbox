import typing as tp
from typing import Optional, Union, Tuple

import torch
from torch import Tensor

from torchani.neighbors import NeighborData
from torchani.nn import Ensemble, ANIModel
from torchani.utils import PERIODIC_TABLE
from torchani.aev.aev_computer import AEVComputer
from torchani.potentials.core import Potential
from torchani.potentials.charges_norm import ChargeNormalizer, ChargeFactor

NN = Union[ANIModel, Ensemble]


# Adaptor to use the aev computer as a three body potential
class AEVPotential(Potential):
    def __init__(self, aev_computer: AEVComputer, neural_networks: NN):
        if isinstance(neural_networks, Ensemble):
            any_nn = neural_networks[0]
        else:
            any_nn = neural_networks
        # Fetch the symbols or "Dummy" if they are not actually elements
        # NOTE: symbols that are not elements is supported for backwards
        # compatibility, since ANIModel supports arbitrary ordered dicts
        # as inputs.
        symbols = tuple(k if k in PERIODIC_TABLE else "Dummy" for k in any_nn)
        super().__init__(cutoff=aev_computer.radial_terms.cutoff, symbols=symbols)
        self.aev_computer = aev_computer
        self.neural_networks = neural_networks
        if isinstance(neural_networks, Ensemble):
            self.size = neural_networks.size
        else:
            self.size = 1

    def forward(
        self,
        element_idxs: Tensor,
        neighbors: NeighborData,
        ghost_flags: Optional[Tensor] = None,
    ) -> Tensor:
        aevs = self.aev_computer._compute_aev(
            element_idxs=element_idxs,
            neighbor_idxs=neighbors.indices,
            distances=neighbors.distances,
            diff_vectors=neighbors.diff_vectors,
        )
        energies = self.neural_networks((element_idxs, aevs)).energies
        return energies

    def atomic_energies(
        self,
        element_idxs: Tensor,
        neighbors: NeighborData,
        ghost_flags: Optional[Tensor] = None,
        average: bool = False,
    ) -> Tensor:
        aevs = self.aev_computer._compute_aev(
            element_idxs=element_idxs,
            neighbor_idxs=neighbors.indices,
            distances=neighbors.distances,
            diff_vectors=neighbors.diff_vectors,
        )
        atomic_energies = self.neural_networks._atomic_energies((element_idxs, aevs))
        if atomic_energies.dim() == 2:
            atomic_energies = atomic_energies.unsqueeze(0)
        if average:
            return atomic_energies.sum(0)
        return atomic_energies


class AEVScalars(Potential):
    def __init__(
        self,
        aev_computer: AEVComputer,
        neural_networks: NN,
        charge_networks: NN,
        charge_factor: tp.Union[ChargeFactor, torch.nn.Module] = ChargeFactor.EQUAL,
        charge_factor_args: tp.Optional[tp.Dict[str, tp.Any]] = None,

    ):
        if isinstance(neural_networks, Ensemble):
            any_nn = neural_networks[0]
        else:
            any_nn = neural_networks
        symbols = tuple(k if k in PERIODIC_TABLE else "Dummy" for k in any_nn)
        super().__init__(cutoff=aev_computer.radial_terms.cutoff, symbols=symbols)

        self.aev_computer = aev_computer
        self.neural_networks = neural_networks
        self.charge_networks = charge_networks
        self.charge_normalizer = ChargeNormalizer(factor=charge_factor, factor_args=charge_factor_args)

        if isinstance(neural_networks, Ensemble):
            self.size = neural_networks.size
        else:
            self.size = 1

    def forward(
        self,
        element_idxs: Tensor,
        neighbors: NeighborData,
        ghost_flags: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        aevs = self.aev_computer._compute_aev(
            element_idxs=element_idxs,
            neighbor_idxs=neighbors.indices,
            distances=neighbors.distances,
            diff_vectors=neighbors.diff_vectors,
        )
        energies = self.neural_networks((element_idxs, aevs)).energies
        raw_charges = self.charge_networks((element_idxs, aevs)).charges
        charges = self.charge_normalizer(element_idxs, raw_charges)
        return energies, charges
