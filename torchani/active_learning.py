import torch
from torch import Tensor
from typing import Tuple
from torchani import utils
from torchani import models


class Sampler:

    def __init__(self, model: torch.nn.Module):
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model

    def sample(self, species_coordinates: Tuple[Tensor, Tensor]):
        raise NotImplementedError


class NormalModeSampler(Sampler):

    # given a batch of conformations performs normal mode sampling on each of
    # the conformations by optimizing the structure, obtaining normal mode
    # displacements, and finally displacing the minimized structure along the
    # normal mode displacements

    def sample(self, species_coordinates: Tuple[Tensor, Tensor], temperature_K=300):
        # First we optimize all geometries in the batch using LBFGS
        species, coordinates = self._optimize_geometry(species_coordinates)
        # here we are good, the optimizer does not optimize
        # geometry of dummy atoms, they remain with zero coordinates

        # Now we obtain the normal modes, reduced masses, etc
        freqs, fconstants, rmasses, modes = self._analyze_vibrations((species, coordinates))

        random_constants = torch.rand((modes.shape[0], modes.shape[1]), dtype=coordinates.dtype, device=species.device)
        # normalize so that the sum across all normal modes is 1
        random_constants /= random_constants.sum(-1, keepdim=True)
        assert torch.isclose(random_constants.sum(-1), torch.tensor(1.0)).all()

        kb = 1.0

        num_atoms = (species >= 0).sum(dim=-1, dtype=coordinates.dtype).unsqueeze(-1)
        factors = torch.sqrt(3 * random_constants * num_atoms * kb * temperature_K / fconstants.unsqueeze(0))

        print(factors)

        # Finally we perform displacements randomly according to the normal modes

    def _optimize_geometry(self,
                          species_coordinates: Tuple[Tensor, Tensor],
                          max_iter: int = 200,
                          tolerance_grad: float = 1e-7) -> Tuple[Tensor, Tensor]:
        # optimizes a batch of species_coordinates using torch's LBFGS optimizer
        species, coordinates = species_coordinates
        coordinates.requires_grad_(True)

        opt = torch.optim.LBFGS((coordinates,),
                                max_iter=max_iter,
                                tolerance_grad=tolerance_grad,
                                line_search_fn='strong_wolfe')

        def closure() -> Tensor:
            opt.zero_grad()
            energy = self.model((species, coordinates)).energies.sum()
            energy.backward()
            return energy

        opt.step(closure)

        return species, coordinates.detach()

    def _analyze_vibrations(self, species_coordinates: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

        modes = []
        freqs = []
        fconstants = []
        rmasses = []

        # it would be nice to have a parallel hessian
        for s, c in zip(species, coordinates):
            vibs = self._analyze_vibrations_single((s.unsqueeze(0), c.unsqueeze(0)))
            modes.append(vibs.modes.unsqueeze(0))
            freqs.append(vibs.freqs.unsqueeze(0))
            fconstants.append(vibs.fconstants.unsqueeze(0))
            rmasses.append(vibs.rmasses.unsqueeze(0))

        # all these have shape (N, (3A), A, 3)
        freqs = torch.cat(freqs)
        # bad_freqs = torch.isnan(freqs) | freqs < 0.0
        freqs = freqs
        fconstants = torch.cat(fconstants)
        rmasses = torch.cat(rmasses)
        modes = torch.cat(modes)

        return freqs, fconstants, rmasses, modes

    def _analyze_vibrations_single(self,
                           species_coordinates: Tuple[Tensor, Tensor]) -> utils.VibAnalysis:
        # here the species may be padded
        # frequencies for padded stuff will be None
        species, coordinates = species_coordinates
        coordinates.requires_grad_(True)
        energy = self.model((species, coordinates)).energies
        hessian = utils.hessian(coordinates, energy)
        # hessian is also good, those atoms have values of zero for the hessian
        vib_analysis = utils.vibrational_analysis(utils.get_atomic_masses(species), hessian)
        return vib_analysis


class TrajectorySampler(Sampler):

    def sample(self, species_coordinates: Tuple[Tensor, Tensor]):
        pass


device = torch.device('cuda')

coordinates = torch.tensor([[[0.03192167, 0.00638559, 0.01301679],
                             [-0.83140486, 0.39370209, -0.26395324],
                             [-0.66518241, -0.84461308, 0.20759389],
                             [0.45554739, 0.54289633, 0.81170881],
                             [0.66091919, -0.16799635, -0.91037834],
                             [0.0, 0.0, 0.0]]], device=device)

species = torch.tensor([[6, 1, 1, 1, 1, -1]], device=device)
model = models.ANI2x(periodic_table_index=True).to(device)
sampler = NormalModeSampler(model)

sampler.sample((species, coordinates))
