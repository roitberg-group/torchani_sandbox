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

use_mnps = [False]
devices = ['cuda', 'cpu']
ani2x = torchani.models.ANI2x()
species_converter = torchani.nn.SpeciesConverter(ani2x.get_chemical_symbols())


@parameterized_class(('device', 'use_mnp'), product(devices, use_mnps))
@unittest.skipIf(not torch.cuda.is_available(), "Infer model needs cuda is available")
class TestInfer(TestCase):

    def setUp(self):
        self.ani2x = ani2x.to(self.device)
        self.species_converter = species_converter.to(self.device)
        self.path = os.path.dirname(os.path.realpath(__file__))

    def _test(self, model_ref, model_infer):
        files = ['small.pdb', '1hz5.pdb', '6W8H.pdb']
        # Skip 6W8H.pdb (slow on cpu) if device is cpu
        files = files[:-1] if self.device == 'cpu' else files
        for file in files:
            filepath = os.path.join(self.path, f'../dataset/pdb/{file}')
            mol = read(filepath)
            species = torch.tensor([mol.get_atomic_numbers()], device=self.device)
            positions = torch.tensor([mol.get_positions()], dtype=torch.float32, requires_grad=False, device=self.device)
            speciesPositions = self.species_converter((species, positions))
            species, coordinates = speciesPositions
            coordinates.requires_grad_(True)

            _, energy1 = model_ref((species, coordinates))
            force1 = torch.autograd.grad(energy1.sum(), coordinates)[0]
            _, energy2 = model_infer((species, coordinates))
            force2 = torch.autograd.grad(energy2.sum(), coordinates)[0]

            self.assertEqual(energy1, energy2, atol=1e-5, rtol=1e-5)
            self.assertEqual(force1, force2, atol=1e-5, rtol=1e-5)

    def testANI2xInfer(self):
        ani2x_infer = torchani.models.ANI2x().to_infer_model(use_mnp=self.use_mnp).to(self.device)
        self._test(ani2x, ani2x_infer)

    def testBmmEnsemble(self):
        model_iterator = self.ani2x.neural_networks
        aev_computer = torchani.AEVComputer.like_2x(use_cuda_extension=(self.device == 'cuda'))
        ensemble = torchani.nn.Sequential(aev_computer, model_iterator).to(self.device)
        bmm_ensemble = torchani.nn.Sequential(aev_computer, self.ani2x.neural_networks.to_infer_model(use_mnp=self.use_mnp)).to(self.device)
        self._test(ensemble, bmm_ensemble)

    def testANI2xInferJIT(self):
        ani2x_infer_jit = torchani.models.ANI2x().to_infer_model(use_mnp=self.use_mnp).to(self.device)
        ani2x_infer_jit = torch.jit.script(ani2x_infer_jit)
        self._test(ani2x, ani2x_infer_jit)

    def testBmmEnsembleJIT(self):
        model_iterator = self.ani2x.neural_networks
        aev_computer = torchani.AEVComputer.like_2x(use_cuda_extension=(self.device == 'cuda'))
        ensemble = torchani.nn.Sequential(aev_computer, model_iterator).to(self.device)
        # jit
        bmm_ensemble = torchani.nn.Sequential(aev_computer, self.ani2x.neural_networks.to_infer_model(use_mnp=self.use_mnp)).to(self.device)
        bmm_ensemble_jit = torch.jit.script(bmm_ensemble)
        self._test(ensemble, bmm_ensemble_jit)

    def testBenchmarkJIT(self):
        """
        Sample benchmark result on 2080 Ti
        cuda:
            run_ani2x                          : 21.739 ms/step
            run_ani2x_infer                    : 9.630 ms/step
        cpu:
            run_ani2x                          : 756.459 ms/step
            run_ani2x_infer                    : 32.482 ms/step
        """
        def run(model, file):
            filepath = os.path.join(self.path, f'../dataset/pdb/{file}')
            mol = read(filepath)
            species = torch.tensor([mol.get_atomic_numbers()], device=self.device)
            positions = torch.tensor([mol.get_positions()], dtype=torch.float32, requires_grad=False, device=self.device)
            speciesPositions = self.species_converter((species, positions))
            species, coordinates = speciesPositions
            coordinates.requires_grad_(True)

            _, energy1 = model((species, coordinates))
            force1 = torch.autograd.grad(energy1.sum(), coordinates)[0]

        use_cuaev = self.device == "cuda"
        ani2x_jit = torch.jit.script(torchani.models.ANI2x(use_cuda_extension=use_cuaev).to(self.device))
        ani2x_infer_jit = torchani.models.ANI2x(use_cuda_extension=use_cuaev).to_infer_model(use_mnp=False).to(self.device)
        ani2x_infer_jit = torch.jit.script(ani2x_infer_jit)

        file = 'small.pdb'

        def run_ani2x():
            run(ani2x_jit, file)

        def run_ani2x_infer():
            run(ani2x_infer_jit, file)

        steps = 10 if self.device == "cpu" else 30
        print()
        torchani.utils.timeit(run_ani2x, steps=steps)
        torchani.utils.timeit(run_ani2x_infer, steps=steps)

if __name__ == '__main__':
    unittest.main()
