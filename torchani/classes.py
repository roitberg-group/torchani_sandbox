"""
Location of data classes for use in ANI built-in models:
    * SpeciesEnergiesQBC: 
    * AtomicQBCs: 
    * SpeciesForces: 
    * ForceQBC: 
    * ForceMagnitudes: 

"""
from torch import Tensor
from typing import NamedTuple


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


class AtomicQBCs(NamedTuple):
    species: Tensor
    energies: Tensor
    stdev_atomic_energies: Tensor


class SpeciesForces(NamedTuple):
    species: Tensor
    energies: Tensor
    model_forces: Tensor


class ForceQBCs(NamedTuple):
    # NOTE: this needs to be updated, should not output so much stuff -- what do i want to see?
    species: Tensor
    energies: Tensor
    mean_forces: Tensor
    stdev_forces: Tensor
    mean_magnitudes: Tensor
    stdev_magnitudes: Tensor
    relative_range: Tensor


class ForceMagnitudes(NamedTuple):
    species: Tensor
    magnitudes: Tensor
    relative_range: Tensor
    relative_stdev: Tensor


class NeighborData(NamedTuple):
    indices: Tensor
    distances: Tensor
    diff_vectors: Tensor
