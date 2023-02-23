from typing import NamedTuple
from torch import Tensor


class SpeciesEnergies(NamedTuple):
    species: Tensor
    energies: Tensor


class EvaluesEvectors(NamedTuple):
    evalues: Tensor
    evectors: Tensor
