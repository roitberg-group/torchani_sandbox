import unittest
import torch
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


if __name__ == '__main__':
    unittest.main()
