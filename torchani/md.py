from torchani.utils import get_atomic_masses, tensor_to_xyz
import math
import torch
from pathlib import Path
from torch import Tensor
from ase import units


class TorchLangevin:

    def __init__(self, model, species: Tensor, coordinates: Tensor, temperature: float, gamma: float, timestep: float, path: str = './trajectory.xyz'):
        # this class allows also for batchwise dynamics, where all dynamics are
        # run in parallel but with different random langevin, so they
        # are independent
        assert coordinates.dim() == 3
        assert coordinates.shape[-1] == 3
        assert species.dim() == 2
        assert species.shape == coordinates.shape[:2]

        temperature *= units.kB
        gamma *= 1 / units.fs
        timestep *= units.fs

        # temperature must be in kb, timestep and gamma in ps
        self.model = model
        self.model.requires_grad_(False)
        assert model.periodic_table_index, "Only periodic_table_index models are allowed"

        self.species = species  # must be atomic numbers
        self.coordinates = coordinates.detach()
        self.gamma = gamma
        self.dt = timestep

        self.forces = None
        self.velocities = None
        self.sigma = torch.sqrt(2 * temperature * self.gamma / get_atomic_masses(species)).unsqueeze(-1)
        self.path = Path(path).resolve()

    def run(self, num_steps: int, print_every: int = 1):
        if self.velocities is None:
            # setting velocities is only needed on step 0
            self.velocities = torch.zeros_like(self.coordinates)

        if self.forces is None:
            # calculation of forces for x_n is only needed on step 0
            self.forces = self.calc_forces(self.coordinates)

        for s in range(num_steps):
            if s % print_every == 0:
                for j, (s, c) in enumerate(zip(self.species, self.coordinates)):
                    filename = self.path.stem
                    path = self.path.parent.joinpath(f'{filename}_{j}.xyz')
                    tensor_to_xyz(path, (s.unsqueeze(0), c.unsqueeze(0)), append=True)
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
        coordinates = coordinates.detach()
        coordinates.requires_grad_(True)
        energy = self.model((self.species, coordinates)).energies
        energy = energy * units.Hartree
        forces = -torch.autograd.grad(energy.sum(), coordinates)[0]
        coordinates = coordinates.detach()
        coordinates.requires_grad_(False)
        return forces
