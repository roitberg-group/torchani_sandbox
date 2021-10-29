import unittest
import torch
import torchani
from torchani.testing import TestCase


class TestGeometry(TestCase):

    def testCenterToComFrame(self):
        species = torch.tensor([[1, 1, 1, 1, 6]], dtype=torch.long)
        coordinates = torch.tensor([[[0.0, 0.0, 0.0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [0.5, 0.5, 0.5]]], dtype=torch.float)
        species, displaced_coordinates, com = torchani.geometry.displace_to_com_frame((species, coordinates))
        self.assertEqual(displaced_coordinates, coordinates - torch.tensor([[0.5, 0.5, 0.5]]).unsqueeze(1))

    def testCenterToComFrameDummy(self):
        species = torch.tensor([[1, 1, 1, 1, 6, -1]], dtype=torch.long)
        coordinates = torch.tensor([[[0.0, 0.0, 0.0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [0.5, 0.5, 0.5], [0, 0, 0]]], dtype=torch.float)
        species, displaced_coordinates, com = torchani.geometry.displace_to_com_frame((species, coordinates))
        expect_coordinates = coordinates - torch.tensor([[0.5, 0.5, 0.5]]).unsqueeze(1)
        expect_coordinates[(species == -1), :] = 0
        self.assertEqual(displaced_coordinates, expect_coordinates)

    def testCenterToComFrameMany(self):
        species = torch.tensor([[1, 1, 1, 1, 6, -1], [6, 6, 6, 6, 8, -1]], dtype=torch.long)
        coordinates = torch.tensor([[[0.0, 0.0, 0.0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [0.5, 0.5, 0.5], [0, 0, 0]]], dtype=torch.float)
        coordinates = torch.cat((coordinates, coordinates.clone()), dim=0)
        species, displaced_coordinates, com = torchani.geometry.displace_to_com_frame((species, coordinates))
        expect_coordinates = coordinates - torch.tensor([[0.5, 0.5, 0.5]]).unsqueeze(1)
        expect_coordinates[(species == -1), :] = 0
        self.assertEqual(displaced_coordinates, expect_coordinates)

if __name__ == '__main__':
    unittest.main()
