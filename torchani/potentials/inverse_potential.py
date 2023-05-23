r"""Contains bare pairwise potentials of the form [-]1 / r ** n

Most of the times these potentials have to be scaled by some constant
before adding them to the energy, for example, if you calculate the bare coulombic
potential, P_ij = 1 / r_ij, then the coulombic energy is actually
Coulomb_ij = Q_ij * P_ij

where Q_ij = q_i * q_j

Similarly, for the VDW terms:

VdwRepulsion_ij = A_ij * P_ij = B_ij * (1 / r_ij ** 12)
VdwAttraction_ij = B_ij * P_ij = B_ij * (- 1 / r_ij ** 6)
"""
import torch
from torch import Tensor

from torchani.neighbors import NeighborData
from torchani.potentials.core import ScaledPairwisePotential


class InversePotential(ScaledPairwisePotential):
    r"""Calculate interaction energies for potentials that decay as 1 / r

    Note that the interaction energies calculated are **not** scaled
    by the charges or vdw parameters,
    they are only 1 / r ** n, potentially modulated by a cutoff function
    """
    def __init__(
        self,
        order: int = 1,
        factor: int = 1,
        **pairwise_kwargs,
    ):
        super().__init__(**pairwise_kwargs)
        self.order = order
        self.factor = factor

    def raw_pair_energies(
        self,
        element_idxs: Tensor,
        neighbor_data: NeighborData,
    ) -> Tensor:
        # Clamp distances to prevent singularities when dividing by zero
        return self.factor / torch.clamp(neighbor_data.distances, min=1e-7) ** self.order
