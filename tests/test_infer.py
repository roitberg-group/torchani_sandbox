import os
import torch
import torchani
from ase.io import read
from itertools import product
import unittest
from torchani.testing import TestCase
from parameterized import parameterized_class

# Disable Tensorfloat, errors between two run of same model for large system could reach 1e-3.
# However note that this error for large system is not that big actually.
torch.backends.cuda.matmul.allow_tf32 = False

use_mnps = [True, False] if torchani.infer.mnp_is_installed else [False]
devices = ['cuda', 'cpu']
ani2x = torchani.models.ANI2x(periodic_table_index=True, model_index=None)


@parameterized_class(('device', 'use_mnp'), product(devices, use_mnps))
@unittest.skipIf(not torch.cuda.is_available(), "Infer model needs cuda is available")
class TestInfer(TestCase):

    def setUp(self):
        self.ani2x = ani2x
        self.path = os.path.dirname(os.path.realpath(__file__))

    def testBmmEnsemble(self):
        model_iterator = self.ani2x.neural_networks
        aev_computer = torchani.AEVComputer.like_2x(use_cuda_extension=(self.device == 'cuda'))
        ensemble = torchani.nn.Sequential(aev_computer, model_iterator).to(self.device)
        bmm_ensemble = torchani.nn.Sequential(aev_computer, self.ani2x.neural_networks.to_infer_model(use_mnp=self.use_mnp)).to(self.device)
        files = ['small.pdb', '1hz5.pdb', '6W8H.pdb']
        for file in files:
            filepath = os.path.join(self.path, f'../dataset/pdb/{file}')
            mol = read(filepath)
            species = torch.tensor([mol.get_atomic_numbers()], device=self.device)
            positions = torch.tensor([mol.get_positions()], dtype=torch.float32, requires_grad=False, device=self.device)
            speciesPositions = self.ani2x.species_converter((species, positions))
            species, coordinates = speciesPositions
            coordinates.requires_grad_(True)

            _, energy1 = ensemble((species, coordinates))
            force1 = torch.autograd.grad(energy1.sum(), coordinates)[0]
            _, energy2 = bmm_ensemble((species, coordinates))
            force2 = torch.autograd.grad(energy2.sum(), coordinates)[0]

            self.assertEqual(energy1, energy2, atol=1e-5, rtol=1e-5)
            self.assertEqual(force1, force2, atol=1e-5, rtol=1e-5)

    def testANIInferModel(self):
        model_iterator = self.ani2x.neural_networks
        aev_computer = torchani.AEVComputer.like_2x(use_cuda_extension=(self.device == 'cuda'))
        model_ref = torchani.nn.Sequential(aev_computer, model_iterator[0]).to(self.device)
        model_infer = torchani.nn.Sequential(aev_computer, model_iterator[0].to_infer_model(use_mnp=self.use_mnp)).to(self.device)
        files = ['small.pdb', '1hz5.pdb', '6W8H.pdb']
        for file in files:
            filepath = os.path.join(self.path, f'../dataset/pdb/{file}')
            mol = read(filepath)
            species = torch.tensor([mol.get_atomic_numbers()], device=self.device)
            positions = torch.tensor([mol.get_positions()], dtype=torch.float32, requires_grad=False, device=self.device)
            speciesPositions = self.ani2x.species_converter((species, positions))
            species, coordinates = speciesPositions
            coordinates.requires_grad_(True)

            _, energy1 = model_ref((species, coordinates))
            force1 = torch.autograd.grad(energy1.sum(), coordinates)[0]
            _, energy2 = model_infer((species, coordinates))
            force2 = torch.autograd.grad(energy2.sum(), coordinates)[0]

            self.assertEqual(energy1, energy2, atol=1e-5, rtol=1e-5)
            self.assertEqual(force1, force2, atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
