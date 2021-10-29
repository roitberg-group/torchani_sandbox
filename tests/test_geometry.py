import unittest
import torch
import torchani
from torchani.testing import TestCase


class TestGeometry(TestCase):

    def testCenterToComFrame(self):
        species = torch.tensor([[1, 1, 1, 1, 6]], dtype=torch.long)


if __name__ == '__main__':
    unittest.main()
