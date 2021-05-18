import unittest
import torch
import math
import torchani
from torchani.testing import TestCase


class TestUtils(TestCase):

    def testChemicalSymbolsToInts(self):
        str2i = torchani.utils.ChemicalSymbolsToInts(['A', 'B', 'C', 'D', 'E', 'F'])
        self.assertEqual(len(str2i), 6)
        self.assertListEqual(str2i('BACCC').tolist(), [1, 0, 2, 2, 2])

    def testChemicalSymbolsToAtomicNumbers(self):
        symbols_to_atomic_nums = torchani.utils.ChemicalSymbolsToAtomicNumbers()
        atomic_nums = symbols_to_atomic_nums(['H', 'H', 'C', 'Cl', 'N', 'H'])
        self.assertEqual(atomic_nums, torch.tensor([1, 1, 6, 17, 7, 1], dtype=torch.long))

    def testHessianJIT(self):
        torch.jit.script(torchani.utils.hessian)

    def testAtomicNumbersToMasses(self):
        znums_to_masses = torchani.utils.AtomicNumbersToMasses()
        species = torch.tensor([1, 1, 6, 17, 7, 1], dtype=torch.long)
        out = znums_to_masses(species)
        base_expect = torch.tensor([1.0080, 1.0080, 12.0110, 35.4500, 14.0070, 1.0080], dtype=torch.float)
        self.assertEqual(out, base_expect)
        # padding
        species = torch.tensor([1, 1, 6, 17, 7, 1, -1, -1], dtype=torch.long)
        out = znums_to_masses(species)
        base_expect_zeros = torch.cat((base_expect, torch.zeros(2, dtype=base_expect.dtype)))
        self.assertEqual(out, base_expect_zeros)

        # many molecules
        species = torch.tensor([[1, 1, 6, 17, 7, 1, 1, 1], [1, 1, 6, 17, 7, 1, -1, -1]], dtype=torch.long)
        out = znums_to_masses(species)
        base_expect_plus = torch.cat((base_expect, torch.full(size=(2,), fill_value=1.0080, dtype=base_expect.dtype)))
        self.assertEqual(out, torch.stack((base_expect_plus, base_expect_zeros), dim=0))

    def testHessian(self):
        # methane
        species = torch.tensor([[6, 1, 1, 1, 1]], dtype=torch.long)
        coordinates = torch.tensor([[[0.03192167, 0.00638559, 0.01301679],
                                     [-0.83140486, 0.39370209, -0.26395324],
                                     [-0.66518241, -0.84461308, 0.20759389],
                                     [0.45554739, 0.54289633, 0.81170881],
                                     [0.66091919, -0.16799635, -0.91037834]]], dtype=torch.float)

        model = torchani.models.ANI2x(periodic_table_index=True)
        coordinates = coordinates.squeeze(0)
        coordinates.requires_grad_(True)
        hessian_expect = torch.autograd.functional.hessian(lambda c: model((species, c.unsqueeze(0))).energies, inputs=coordinates, vectorize=False)
        hessian_expect = hessian_expect.flatten(0, 1).flatten(1, 2).unsqueeze(0)

        # we keep our own hand made implementation of hessian because torch's
        # implementation is slower
        coordinates = coordinates.detach()
        coordinates = coordinates.unsqueeze(0)
        coordinates.requires_grad_(True)
        energy = model((species, coordinates)).energies
        hessian = torchani.utils.hessian(coordinates, energy)

        self.assertEqual(hessian, hessian_expect)

    def testHessianPadding(self):
        # padded water
        species = torch.tensor([[1, 1, 8, -1]], dtype=torch.long)
        coordinates = torch.tensor([[[9.6105e-01, 2.5066e-03, -3.3340e-19],
                                     [-2.3836e-01, 9.3103e-01, -2.7523e-20],
                                     [-5.0882e-03, -6.5727e-03, 1.6920e-22],
                                     [0.0, 0.0, 0.0]]], dtype=torch.float)

        model = torchani.models.ANI2x(periodic_table_index=True)
        coordinates = coordinates.squeeze(0)
        coordinates.requires_grad_(True)
        hessian_expect = torch.autograd.functional.hessian(lambda c: model((species, c.unsqueeze(0))).energies, inputs=coordinates, vectorize=False)
        hessian_expect = hessian_expect.flatten(0, 1).flatten(1, 2).unsqueeze(0)

        coordinates = coordinates.detach()
        coordinates = coordinates.unsqueeze(0)
        coordinates.requires_grad_(True)
        energy = model((species, coordinates)).energies
        hessian = torchani.utils.hessian(coordinates, energy)
        self.assertEqual(hessian, hessian_expect)

    def testBatchedHessian(self):
        coordinates = torch.tensor([[[9.6105e-01, 2.5066e-03, -3.3340e-19],
                                     [-2.3836e-01, 9.3103e-01, -2.7523e-20],
                                     [-5.0882e-03, -6.5727e-03, 1.6920e-22],
                                     [0.0, 0.0, 0.0]]], dtype=torch.float)
        species = torch.tensor([[1, 1, 8, -1]], dtype=torch.long)
        model = torchani.models.ANI2x(periodic_table_index=True)

        coordinates = coordinates.detach()
        coordinates.requires_grad_(True)
        energy = model((species, coordinates)).energies
        hessian_expect = torchani.utils.hessian(coordinates, energy).repeat(2, 1, 1)
        coordinates.requires_grad_(False)

        coordinates = coordinates.repeat(2, 1, 1)
        species = species.repeat(2, 1)
        hessian = torchani.utils.batched_hessian(model, (species, coordinates))
        self.assertEqual(hessian_expect, hessian)

    def testCOM(self):
        com_calc = torchani.utils.CenterOfMass()

        # symmetric
        coordinates0 = torch.tensor([[[0.0, 0.0, 0.0],
                                     [1.0, 0.0, 0.0],
                                     [-1.0, 0.0, 0.0]]], dtype=torch.float)
        species0 = torch.tensor([[6, 8, 8]], dtype=torch.long)
        com = com_calc((species0, coordinates0))
        expect0 = torch.tensor([[0.0, 0.0000, 0.0000]], dtype=torch.float)
        self.assertEqual(com, expect0)

        # asymmetric
        coordinates1 = torch.tensor([[[0.0, 0.0, 0.0],
                                     [1.0, 0.0, 0.0],
                                     [-2.0, 0.0, 0.0]]], dtype=torch.float)
        species1 = torch.tensor([[6, 8, 8]], dtype=torch.long)
        com = com_calc((species1, coordinates1))
        expect1 = torch.tensor([[-0.3635392, 0.0000, 0.0000]], dtype=torch.float)
        self.assertEqual(com, expect1)

        # batched
        com = com_calc((torch.cat((species0, species1), dim=0), torch.cat((coordinates0, coordinates1), dim=0)))
        self.assertEqual(com, torch.cat((expect0, expect1), dim=0))

    def testMOI(self):
        coordinates0 = torch.tensor([[[0.0, 0.0, 0.0],
                                     [1.0, 0.0, 0.0],
                                     [-1.0, 0.0, 0.0]]], dtype=torch.float)
        species0 = torch.tensor([[6, 8, 8]], dtype=torch.long)

        coordinates0 = torch.tensor([[[0.0, 1.0, 0.0],
                                     [math.sqrt(3) / 2, -1 / 2, 0.0],
                                     [-math.sqrt(3) / 2, -1 / 2, 0.0]]], dtype=torch.float)
        species0 = torch.tensor([[1, 1, 1]], dtype=torch.long)

        moi_calc = torchani.utils.MomentOfInertia()
        moi_out = moi_calc((species0, coordinates0))
        moi_expect = torch.tensor([[[1.5060, 0.0000, 0.0000],
                                    [0.0000, 1.5060, 0.0000],
                                    [0.0000, 0.0000, 3.0119762]]], dtype=torch.float)
        evalues_expect = torch.tensor([[1.5060, 1.5060, 3.0119762]], dtype=torch.float)
        evectors_expect = torch.tensor([[[1., -0., 0.],
                                         [0., 1., 0.],
                                         [0., 0., 1.]]], dtype=torch.float)
        self.assertEqual(moi_expect, moi_out.moi)
        self.assertEqual(evalues_expect, moi_out.eigenvalues)
        self.assertEqual(evectors_expect, moi_out.eigenvectors)

        # padded
        coordinates0 = torch.tensor([[[0.0, 1.0, 0.0],
                                     [math.sqrt(3) / 2, -1 / 2, 0.0],
                                     [-math.sqrt(3) / 2, -1 / 2, 0.0],
                                     [0.0, 0.0, 0.0]]], dtype=torch.float)

        species0 = torch.tensor([[1, 1, 1, -1]], dtype=torch.long)
        moi_out = moi_calc((species0, coordinates0))
        self.assertEqual(moi_expect, moi_out.moi)
        self.assertEqual(evalues_expect, moi_out.eigenvalues)
        self.assertEqual(evectors_expect, moi_out.eigenvectors)

        # batched
        moi_out = moi_calc((species0.repeat(2, 1), coordinates0.repeat(2, 1, 1)))
        self.assertEqual(moi_expect.repeat(2, 1, 1), moi_out.moi)
        self.assertEqual(evalues_expect.repeat(2, 1), moi_out.eigenvalues)
        self.assertEqual(evectors_expect.repeat(2, 1, 1), moi_out.eigenvectors)


if __name__ == '__main__':
    unittest.main()
