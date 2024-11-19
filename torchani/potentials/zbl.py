import math
import typing as tp
import torch
from torch import Tensor
from torchani.cutoffs import CutoffArg
from torchani.potentials.core import BasePairPotential
from torchani.neighbors import Neighbors


# Ziegler-Biersack-Littman screened nuclear repulsion
#  Chapter 2, "Universal screening fn" etc
# TODO: trainable
class RepulsionZBL(BasePairPotential):
    def __init__(
        self,
        symbols: tp.Sequence[str],
        # Potential
        # one value is from Ziegler et.al., the other from lammps, I suspect they
        # use different units for coords
        k: float = 0.8853,  # or 0.46850?
        phi_coeffs: tp.Sequence[float] = (),
        phi_exponents: tp.Sequence[float] = (),
        eff_exponent: float = 0.23,
        eff_atomic_nums: tp.Sequence[float] = (),
        trainable: tp.Sequence[str] = (),
        *,  # Cutoff
        cutoff: float = math.inf,
        cutoff_fn: CutoffArg = "smooth",
    ):
        super().__init__(symbols, cutoff=cutoff, cutoff_fn=cutoff_fn)
        eff_atomic_nums = self._validate_elem_seq(
            "eff_atomic_nums",
            eff_atomic_nums,
            torch.arange(118, dtype=torch.float).tolist(),
        )

        if not len(phi_exponents) == len(phi_coeffs):
            raise ValueError("phi_exponents and phi_coeffs must have the same len")

        # Defaults from Ziegler et. al.
        if not phi_exponents:
            phi_exponents = [0.18175, 0.50986, 0.28022, 0.02817]
        if not phi_coeffs:
            phi_coeffs = [3.19980, 0.94229, 0.40290, 0.20162]

        # Params and buffers
        self.register_buffer(
            "eff_atomic_nums", torch.tensor(eff_atomic_nums), persistent=False
        )
        self.register_buffer(
            "_coeffs", torch.tensor(phi_coeffs).view(1, -1), persistent=False
        )
        self.register_buffer(
            "_exponents", torch.tensor(phi_exponents).view(1, -1), persistent=False
        )
        self._k = k
        self._kz = eff_exponent

    def pair_energies(self, elem_idxs: Tensor, neighbors: Neighbors) -> Tensor:
        # Clamp distances to prevent singularities when dividing by zero
        # Also, all internal calculations of this module are made with atomic units,
        # so distances are first converted to bohr
        dists = self.clamp(neighbors.distances) * self.ANGSTROM_TO_BOHR

        elem_pairs = elem_idxs.flatten()[neighbors.indices]
        eff_zi = self.eff_atomic_nums[elem_pairs[0]]
        eff_zj = self.eff_atomic_nums[elem_pairs[1]]

        prefactor = eff_zi * eff_zj / dists
        inv_screen_length = (eff_zi**self._kz + eff_zj**self._kz) / self._k
        reduced_dists = dists * inv_screen_length
        return prefactor * self._phi(reduced_dists)

    def _phi(self, dists: Tensor) -> Tensor:  # Output shape is same as input
        return (self._coeffs * torch.exp(-self._exponents * dists.view(-1, 1))).sum(-1)
