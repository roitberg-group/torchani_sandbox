import unittest
import torch
import torchani
from pathlib import Path
from torchani.models import _fetch_state_dict
from torchani.testing import TestCase
from torchani.repulsion import RepulsionCalculator, StandaloneRepulsionCalculator
from torchani.short_range_basis import EnergySRB, StandaloneEnergySRB


class TestRepulsion(TestCase):
    def setUp(self):
        self.rep = RepulsionCalculator(5.2)
        self.stand_rep = StandaloneRepulsionCalculator(cutoff=5.2, neighborlist_cutoff=5.2)

    def testCalculator(self):
        self._testCalculator(3.5325e-08)

    def testStandalone(self):
        self._testStandalone(3.5325e-08)

    def testModelEnergy(self):
        path = Path(__file__).resolve().parent.joinpath('test_data/energies_repulsion_1x.pkl')
        self._testModelEnergy(path, repulsion=True)

    def _testCalculator(self, expected_energy):
        atom_index12 = torch.tensor([[0], [1]])
        distances = torch.tensor([3.5])
        species = torch.tensor([[0, 0]])
        energies = torch.tensor([0.0])
        energies = self.rep((species, energies), atom_index12, distances).energies
        self.assertTrue(torch.isclose(torch.tensor(expected_energy), energies))

    def _testStandalone(self, expected_energy):
        coordinates = torch.tensor([[0.0, 0.0, 0.0],
                                    [3.5, 0.0, 0.0]]).unsqueeze(0)
        species = torch.tensor([[0, 0]])
        energies = self.stand_rep((species, coordinates)).energies
        self.assertTrue(torch.isclose(torch.tensor(expected_energy), energies))

    def testBatches(self):
        coordinates1 = torch.tensor([[0.0, 0.0, 0.0],
                                    [1.5, 0.0, 0.0],
                                    [3.0, 0.0, 0.0]]).unsqueeze(0)
        coordinates2 = torch.tensor([[0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0],
                                    [2.5, 0.0, 0.0]]).unsqueeze(0)
        coordinates3 = torch.tensor([[0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0],
                                     [3.5, 0.0, 0.0]]).unsqueeze(0)
        species1 = torch.tensor([[0, 1, 2]])
        species2 = torch.tensor([[-1, 0, 1]])
        species3 = torch.tensor([[-1, 0, 0]])
        coordinates_cat = torch.cat((coordinates1, coordinates2, coordinates3), dim=0)
        species_cat = torch.cat((species1, species2, species3), dim=0)

        energy1 = self.stand_rep((species1, coordinates1)).energies
        # avoid first atom since it isdummy
        energy2 = self.stand_rep((species2[:, 1:], coordinates2[:, 1:, :])).energies
        energy3 = self.stand_rep((species3[:, 1:], coordinates3[:, 1:, :])).energies
        energies_cat = torch.cat((energy1, energy2, energy3))
        energies = self.stand_rep((species_cat, coordinates_cat)).energies
        self.assertTrue(torch.isclose(energies, energies_cat).all())

    def testLongDistances(self):
        atom_index12 = torch.tensor([[0], [1]])
        distances = torch.tensor([6.0])
        species = torch.tensor([[0, 0]])
        energies = torch.tensor([0.0])
        energies = self.rep((species, energies), atom_index12, distances).energies
        self.assertTrue(torch.isclose(torch.tensor(0.0), energies))

    def _testModelEnergy(self, path, repulsion=False, srb=False):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = torchani.models.ANI1x(repulsion=repulsion, srb=srb, pretrained=False, model_index=0, cutoff_fn='smooth')
        model.load_state_dict(_fetch_state_dict('ani1x_state_dict.pt', 0), strict=False)
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
        with open(path, 'rb') as f:
            energies_expect = torch.tensor(torch.load(f))
        self.assertEqual(energies_expect, energies)


class TestSRB(TestRepulsion):
    def setUp(self):
        self.rep = EnergySRB(cutoff=5.2)
        self.stand_rep = StandaloneEnergySRB(cutoff=5.2, neighborlist_cutoff=5.2)

    def testCalculator(self):
        self._testCalculator(-1.8757)

    def testStandalone(self):
        self._testStandalone(-1.8757)

    def testModelEnergy(self):
        path = Path(__file__).resolve().parent.joinpath('test_data/energies_srb_1x.pkl')
        self._testModelEnergy(path, srb=True)


if __name__ == '__main__':
    unittest.main()
