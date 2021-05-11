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
        symbols_to_indices = torchani.utils.ChemicalSymbolsToAtomicNumbers()
        atomic = symbols_to_indices(['H', 'H', 'C', 'Cl', 'N', 'H'])
        self.assertEqual(atomic, torch.tensor([1, 1, 6, 17, 7, 1], dtype=torch.long))

    def testHessianJIT(self):
        torch.jit.script(torchani.utils.hessian)


if __name__ == '__main__':
    unittest.main()
