import torch
from typing import Tuple, Tensor
from torchani.utils import vibrational_analysis, hessian
from torchani import utils


class NormalModeSampler:

    # given a batch of conformations performs normal mode sampling on each of
    # the conformations by optimizing the structure, obtaining normal mode
    # displacements, and finally displacing the minimized structure along the
    # normal mode displacements

    def __init__(self, model: torch.nn.Module):
        self.model = model

    def sample(self, species_coordinates: Tuple[Tensor, Tensor]):
        # the idea here is to obtain the hessian for the batch and do the stuff
        species, coordinates = species_coordinates
        for s, c in zip(species, coordinates):
            c.requires_grad_(True)
            energy = self.model((s, c)).energies
            forces = -torch.autograd.grad(energy, c)
            hessian = utils.hessian(c, energy, forces)
            vib_analysis = vibrational_analysis(utils.get_atomic_masses(s), hessian, mode_type='MDN')
            modes = vib_analysis.modes


class TrajectorySampler:

    def __init__(self, model: torch.nn.Module):
        self.model = model

    def sample(self, species_coordinates: Tuple[Tensor, Tensor]):

