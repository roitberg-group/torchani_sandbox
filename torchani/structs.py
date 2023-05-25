from typing import NamedTuple

from torch import Tensor


class SpeciesEnergies(NamedTuple):
    species: Tensor
    energies: Tensor


class SpeciesEnergiesAtomicCharges(NamedTuple):
    species: Tensor
    energies: Tensor
    charges: Tensor
