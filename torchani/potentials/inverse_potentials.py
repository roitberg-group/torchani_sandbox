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
from typing import Optional

import torch
from torch import Tensor

from torchani.potentials.core import PairwisePotential


class BareCoulomb(PairwisePotential):
    r"""Calculates the coulombic interaction energy

    Note that the interaction energies calculated are **not** scaled
    by the charges, they are only 1 / r, potentially modulated by a cutoff function

    pairwise_kwargs are passed to PairwisePotential
    """

    charges: Tensor

    def __init__(
        self,
        **pairwise_kwargs,
    ):
        super().__init__(**pairwise_kwargs)

    def pair_energies(
        self,
        element_idxs: Tensor,
        neighbor_idxs: Tensor,
        distances: Tensor,
        diff_vectors: Optional[Tensor] = None,
    ) -> Tensor:
        # Clamp distances to prevent singularities when dividing by zero
        return 1 / torch.clamp(distances, min=1e-7)


class BareVdwRepulsion(PairwisePotential):
    r"""Calculates the repulsive VDW energy

    Note that the interaction energies calculated are **not** scaled by A,B
    params, they are only 1 / r ** 12, potentially modulated by a cutoff
    function

    pairwise_kwargs are passed to PairwisePotential
    """

    charges: Tensor

    def __init__(
        self,
        **pairwise_kwargs,
    ):
        super().__init__(**pairwise_kwargs)

    def pair_energies(
        self,
        element_idxs: Tensor,
        neighbor_idxs: Tensor,
        distances: Tensor,
        diff_vectors: Optional[Tensor] = None,
    ) -> Tensor:
        # Clamp distances to prevent singularities when dividing by zero
        return - 1 / torch.clamp(distances, min=1e-7) ** 12


class BareVdwAttraction(PairwisePotential):
    r"""Calculates the attractive VDW energy

    Note that the interaction energies calculated are **not** scaled
    by A,B params, they are only - 1 / r **6, potentially modulated by a cutoff function

    pairwise_kwargs are passed to PairwisePotential
    """

    charges: Tensor

    def __init__(
        self,
        **pairwise_kwargs,
    ):
        super().__init__(**pairwise_kwargs)

    def pair_energies(
        self,
        element_idxs: Tensor,
        neighbor_idxs: Tensor,
        distances: Tensor,
        diff_vectors: Optional[Tensor] = None,
    ) -> Tensor:
        # Clamp distances to prevent singularities when dividing by zero
        return 1 / torch.clamp(distances, min=1e-7) ** 6
