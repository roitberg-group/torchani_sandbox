import unittest

import torch

import torchani
from torchani.testing import TestCase


class TestANI2xCharges(TestCase):

    def setUp(self):
        self.model = torchani.models.ANI2x(model_index=0)
        self.model_charges = torchani.models.ANI2xCharges(model_index=0)

    def testDiatomics(self):
        coordinates = torch.tensor([[[0.0, 0.0, 0.0],
                                    [0.0, 0.0, 2.0]]])
        coordinates = coordinates.repeat(4, 1, 1)
        species = torch.tensor([[9, 9], [16, 16], [8, 8], [17, 17]])
        energy_expect = self.model((species, coordinates)).energies
        energy = self.model_charges((species, coordinates)).energies
        self.assertEqual(energy_expect, energy)

        # Compare against 2x energies calculated directly from neurochem by kdavis
        energy_neurochem_expect = torch.tensor(
            [-125100.7729, -499666.2354, -94191.3460, -577504.1792]
        )
        self.assertEqual(
            torchani.units.hartree2kcalmol(energy_expect.to(torch.float)),
            energy_neurochem_expect.to(torch.float),
        )

    def testChargesSanity(self):
        coordinates = torch.tensor([[[0.0, 0.0, 0.0],
                                    [0.0, 0.0, 2.0]]], requires_grad=True)
        coordinates = coordinates.repeat(4, 1, 1)
        species = torch.tensor([[9, 9], [16, 16], [8, 8], [17, 17]])

        charges = self.model_charges((species, coordinates)).atomic_charges
        _ = - torch.autograd.grad(charges.sum(), coordinates)[0]
        self.assertEqual(charges.sum(), torch.tensor(0.0))
        # By symmetry this must hold
        self.assertEqual(charges[:, 0], charges[:, 1])

    def testChargesPolarized(self):
        coordinates = torch.tensor([[[0.0, 0.0, 0.0],
                                    [0.0, 0.0, 2.0]]], requires_grad=True)
        coordinates = coordinates.repeat(2, 1, 1)
        species = torch.tensor([[1, 9], [1, 17]])

        charges = self.model_charges((species, coordinates)).atomic_charges
        _ = - torch.autograd.grad(charges.sum(), coordinates)[0]
        self.assertEqual(charges.sum(), torch.tensor(0.0))
        # Check correct polarization
        self.assertGreater(charges[0, 0], 0.0)
        self.assertGreater(charges[1, 0], 0.0)
        self.assertLess(charges[0, 1], 0.0)
        self.assertLess(charges[0, 1], 0.0)

    def testForces(self):
        coordinates = torch.tensor([[[0.0, 0.0, 0.0],
                                    [0.0, 0.0, 2.0]]], requires_grad=True)
        coordinates = coordinates.repeat(4, 1, 1)
        species = torch.tensor([[9, 9], [16, 16], [8, 8], [17, 17]])
        energy_expect = self.model((species, coordinates)).energies
        forces_expect = - torch.autograd.grad(energy_expect.sum(), coordinates)[0]

        coordinates = coordinates.detach()
        coordinates.requires_grad_(True)
        energy = self.model_charges((species, coordinates)).energies
        forces = - torch.autograd.grad(energy.sum(), coordinates)[0]

        self.assertEqual(forces_expect, forces)


class TestANI2xChargesJIT(TestCase):
    def setUp(self):
        self.model = torch.jit.script(torchani.models.ANI2x(model_index=0))
        self.model_charges = torch.jit.script(torchani.models.ANI2xCharges(model_index=0))


if __name__ == '__main__':
    unittest.main()
