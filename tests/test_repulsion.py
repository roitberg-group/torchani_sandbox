import unittest
import torch
import torchani
from torchani.testing import TestCase
from torchani.repulsion import RepulsionCalculator, StandaloneRepulsionCalculator
import pickle


def load_model(path, aev_dim):
    H_network = torch.nn.Sequential(
        torch.nn.Linear(aev_dim, 160, bias=False),
        torch.nn.GELU(),
        torch.nn.Linear(160, 128, bias=False),
        torch.nn.GELU(),
        torch.nn.Linear(128, 96, bias=False),
        torch.nn.GELU(),
        torch.nn.Linear(96, 1, bias=False)
    )
    C_network = torch.nn.Sequential(
        torch.nn.Linear(aev_dim, 144, bias=False),
        torch.nn.GELU(),
        torch.nn.Linear(144, 112, bias=False),
        torch.nn.GELU(),
        torch.nn.Linear(112, 96, bias=False),
        torch.nn.GELU(),
        torch.nn.Linear(96, 1, bias=False)
    )
    N_network = torch.nn.Sequential(
        torch.nn.Linear(aev_dim, 128, bias=False),
        torch.nn.GELU(),
        torch.nn.Linear(128, 112, bias=False),
        torch.nn.GELU(),
        torch.nn.Linear(112, 96, bias=False),
        torch.nn.GELU(),
        torch.nn.Linear(96, 1, bias=False)
    )
    O_network = torch.nn.Sequential(
        torch.nn.Linear(aev_dim, 128, bias=False),
        torch.nn.GELU(),
        torch.nn.Linear(128, 112, bias=False),
        torch.nn.GELU(),
        torch.nn.Linear(112, 96, bias=False),
        torch.nn.GELU(),
        torch.nn.Linear(96, 1, bias=False)
    )
    nn = torchani.ANIModel([H_network, C_network, N_network, O_network])
    nn.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    return nn


class TestRepulsion(TestCase):

    def testRepulsionCalculator(self):
        rep = RepulsionCalculator(5.2)
        atom_index12 = torch.tensor([[0], [1]])
        distances = torch.tensor([3.5])
        species = torch.tensor([[0, 0]])
        energies = torch.tensor([0.0])
        energies = rep((species, energies), atom_index12, distances).energies
        self.assertTrue(torch.isclose(torch.tensor(3.5325e-08), energies))

    def testRepulsionStandalone(self):
        rep = StandaloneRepulsionCalculator(cutoff=5.2, neighborlist_cutoff=5.2)
        coordinates = torch.tensor([[0.0, 0.0, 0.0],
                                    [3.5, 0.0, 0.0]]).unsqueeze(0)
        species = torch.tensor([[0, 0]])
        energies = rep((species, coordinates)).energies
        self.assertTrue(torch.isclose(torch.tensor(3.5325e-08), energies))

    def testRepulsionLongDistances(self):
        rep = RepulsionCalculator(5.2)
        atom_index12 = torch.tensor([[0], [1]])
        distances = torch.tensor([6.0])
        species = torch.tensor([[0, 0]])
        energies = torch.tensor([0.0])
        energies = rep((species, energies), atom_index12, distances).energies
        self.assertTrue(torch.isclose(torch.tensor(0.0), energies))

    def testRepulsionEnergy(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = torchani.models.ANI1x(repulsion=True, model_index=0)
        model.neural_networks = load_model('repulsion_model_1x.pt', model.aev_computer.aev_length)
        model.energy_shifter = torchani.EnergyShifter([-0.506930115400, -37.814410115700, -54.55653828400, -75.02918133970])
        model = model.to(device=device, dtype=torch.double)

        species = torch.tensor([[3, 0, 0]], device=device)
        energies = []
        distances = torch.linspace(0.1, 6.0, 100)
        for d in distances:
            coordinates = torch.tensor([[[0.0, 0.0, 0.0],
                                        [0.97, 0.0, 0.0],
                                        [-0.250380004 * d, 0.96814764 * d, 0.0]]],
                                       requires_grad=True, device=device, dtype=torch.double)
            energies.append(model((species, coordinates)).energies.item())

        energies = torch.tensor(energies)
        with open('energies.pkl', 'rb') as f:
            energies_salva = torch.tensor(pickle.load(f))
        self.assertTrue(torch.isclose(energies_salva, energies).all())


if __name__ == '__main__':
    unittest.main()
