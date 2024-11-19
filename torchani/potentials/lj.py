import math
import typing as tp

import torch
from torch import Tensor
from torch.nn import Parameter

from torchani.potentials.core import BasePairPotential
from torchani.neighbors import Neighbors
from torchani.cutoffs import CutoffArg


class _LJ(BasePairPotential):

    def __init__(
        self,
        symbols: tp.Sequence[str],
        eps: tp.Sequence[float] = (),
        sigma: tp.Sequence[float] = (),
        *,  # Cutoff
        trainable: tp.Sequence[str] = (),
        cutoff: float = math.inf,
        cutoff_fn: CutoffArg = "smooth",
    ):
        super().__init__(symbols, cutoff=cutoff, cutoff_fn=cutoff_fn)
        if not set(trainable).issubset(("sigma", "eps")):
            raise ValueError(f"Unsupported parameters in {trainable}")

        for k, v in (("sigma", sigma), ("eps", eps)):
            v = self._validate_elem_seq(k, v)
            if k in trainable:
                self.register_parameter(f"_{k}", Parameter(torch.tensor(v)))
            else:
                self.register_buffer(f"_{k}", torch.tensor(v), persistent=False)

    def combine_eps(self, elem_pairs: Tensor) -> Tensor:
        # Berthelot rule
        return torch.sqrt(self._eps[elem_pairs[0]] * self._eps[elem_pairs[1]])

    def combine_sigma(self, elem_pairs: Tensor) -> Tensor:
        # Lorentz rule
        return (self._sigma[elem_pairs[0]] + self._sigma[elem_pairs[1]]) / 2


class DispersionLJ(_LJ):
    def pair_energies(self, elem_idxs, neighbors: Neighbors):
        elem_pairs = elem_idxs.flatten()[neighbors.indices]
        eps = self.combine_eps(elem_pairs)
        sigma = self.combine_sigma(elem_pairs)
        x = sigma / self.clamp(neighbors.distances)
        return -4 * eps * x ** 6


class RepulsionLJ(_LJ):
    def pair_energies(self, elem_idxs, neighbors: Neighbors):
        elem_pairs = elem_idxs.flatten()[neighbors.indices]
        eps = self.combine_eps(elem_pairs)
        sigma = self.combine_sigma(elem_pairs)
        x = sigma / self.clamp(neighbors.distances)
        return 4 * eps * x ** 12


class LennardJones(_LJ):
    def pair_energies(self, elem_idxs, neighbors: Neighbors):
        elem_pairs = elem_idxs.flatten()[neighbors.indices]
        eps = self.combine_eps(elem_pairs)
        sigma = self.combine_sigma(elem_pairs)
        x = sigma / self.clamp(neighbors.distances)
        return 4 * eps * (x ** 12 - x ** 6)
