import unittest
from torchani.testing import TestCase
from torchani.aev.aev_terms import PhysNetRadial
from torchani.aev.cutoffs import CutoffPhysNet


class TestPhysNet(TestCase):

    def testPhysNetRadial(self):
        radial = PhysNetRadial()
        self.assertTrue(isinstance(radial.cutoff_fn, CutoffPhysNet))
        self.assertEqual(radial.cutoff, 10.0)
        print(radial.Mu)
        print(radial.Beta)


if __name__ == '__main__':
    unittest.main()
