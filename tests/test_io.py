import unittest
import torch
from torchani.io import tensor_from_xyz, tensor_to_xyz, tensor_to_lammpstrj, tensor_from_lammpstrj
from torchani.testing import TestCase


class TestUtils(TestCase):
    def testToXyz(self):
        species, coordinates, cell = tensor_from_xyz('./sample.xyz', get_cell=False, step=1)

    def testFromXyz(self):
        species, coordinates, cell = tensor_from_xyz('./sample.xyz', get_cell=False, step=1)

    def testToLammps(self):
        species, coordinates, cell = tensor_from_xyz('./sample.xyz', get_cell=False, step=1)
        tensor_to_lammpstrj('./sample.lammpstrj', (species, coordinates), cell=torch.eye(3).unsqueeze(0).repeat(len(species), 1, 1), truncate_output_file=True)
    def testFromLammps(self):
        species, coordinates, cell, _, _ = tensor_from_lammpstrj('./sample.lammpstrj', get_cell=True, extract_atoms=(0, 2))
        print(coordinates)
        print(species)


if __name__ == '__main__':
    unittest.main()
