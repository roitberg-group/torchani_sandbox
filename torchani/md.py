from torchani.utils import get_atomic_masses
import math
import torch
from torch import Tensor
from ase import units


class TorchLangevin:

    def __init__(self, model, species: Tensor, coordinates: Tensor, temperature: float, gamma: float, timestep: float):
        # need to recheck this to be sure that  units are correct
        temperature *= units.kB
        gamma *= 1 / units.fs
        timestep *= units.fs

        # temperature must be in kb, timestep and gamma in ps
        self.model = model
        self.model.requires_grad_(False)
        assert model.periodic_table_index, "Only periodic_table_index models are allowed"

        self.species = species  # must be atomic numbers
        self.coordinates = coordinates.squeeze().detach()
        self.gamma = gamma
        self.dt = timestep

        self.forces = None
        self.velocities = None
        self.sigma = torch.sqrt(2 * temperature * self.gamma / get_atomic_masses(species).squeeze()).unsqueeze(-1)

    def run(self, num_steps: int):
        if self.velocities is None:
            # setting velocities is only needed on step 0
            self.velocities = torch.zeros_like(self.coordinates)

        if self.forces is None:
            # calculation of forces for x_n is only needed on step 0
            self.forces = self.calc_forces(self.coordinates)

        for _ in range(num_steps):
            self.step()

    def step(self):
        with torch.no_grad():
            # update random constants
            self.xi = torch.zeros_like(self.coordinates).normal_(mean=0.0, std=1.0)
            self.eta = torch.zeros_like(self.coordinates).normal_(mean=0.0, std=1.0)
            # these terms are the same in both half step velocity
            # updates, so they can be cached
            term3 = 0.5 * math.sqrt(self.dt) * self.sigma * self.xi
            term5 = - (1 / 4) * self.dt**(3 / 2) * self.gamma * self.sigma * (0.5 * self.xi + self.eta / math.sqrt(3))
            velocity_terms35 = term3 + term5
            # half step on v
            self.velocities += self._calc_velocities_half_step_terms124(self.forces, self.velocities)
            self.velocities += velocity_terms35
            # full step on x, x_n -> x_n+1
            self.coordinates += self._calc_coordinates_full_step_terms(self.velocities)

        # we now calculate forces on x_n+1
        self.forces = self.calc_forces(self.coordinates)

        with torch.no_grad():
            # half step on v using forces on x_n+1
            self.velocities += self._calc_velocities_half_step_terms124(self.forces, self.velocities)
            self.velocities += velocity_terms35

    def _calc_velocities_half_step_terms124(self, forces: Tensor, velocities: Tensor) -> Tensor:
        term1 = 0.5 * self.dt * forces  # not cacheable
        term2 = - 0.5 * self.dt * self.gamma * velocities  # not cacheable
        term4 = - (1 / 8) * math.sqrt(self.dt) * self.gamma * (forces - self.gamma * velocities)  # not cacheable
        return term1 + term2 + term4

    def _calc_coordinates_full_step_terms(self, velocities: Tensor) -> Tensor:
        term1 = self.dt * velocities
        term2 = self.dt**(3 / 2) * self.sigma * (1 / (2 * math.sqrt(3))) * self.eta
        return term1 + term2

    def calc_forces(self, coordinates: Tensor) -> Tensor:
        coordinates = coordinates.unsqueeze(0)
        coordinates = coordinates.detach()
        coordinates.requires_grad_(True)
        energy = self.model((self.species, coordinates)).energies
        energy = energy * units.Hartree
        forces = -torch.autograd.grad(energy.squeeze(), coordinates)[0].squeeze()
        coordinates = coordinates.detach()
        coordinates.requires_grad_(False)
        coordinates = coordinates.squeeze()
        return forces
