"""
Location of data classes for use in ANI built-in models:
    * SpeciesEnergiesQBC: 
    * AtomicQBCs: 
    * SpeciesForces: 
    * ForceQBC: 
    * ForceMagnitudes: 

"""
from typing import NamedTuple
from torch import Tensor


class SpeciesCoordinates(NamedTuple):
    species: Tensor
    coordinates: Tensor


class SpeciesEnergies(NamedTuple):
    species: Tensor
    energies: Tensor


class SpeciesEnergiesQBC(NamedTuple):
    species: Tensor
    energies: Tensor
    qbcs: Tensor


class AtomicStdev(NamedTuple):
    species: Tensor
    energies: Tensor
    stdev_atomic_energies: Tensor


class SpeciesForces(NamedTuple):
    species: Tensor
    energies: Tensor
    model_forces: Tensor


class ForceStdev(NamedTuple):
    species: Tensor
    energies: Tensor
    mean_forces: Tensor
    stdev_forces: Tensor


class ForceMagnitudes(NamedTuple):
    species: Tensor
    magnitudes: Tensor
    relative_range: Tensor
    relative_stdev: Tensor


class NeighborData(NamedTuple):
    indices: Tensor
    distances: Tensor
    diff_vectors: Tensor
