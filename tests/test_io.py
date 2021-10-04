import unittest
import tempfile
import torch
from torchani.io import tensor_from_xyz, tensor_to_xyz, tensor_to_lammpstrj, tensor_from_lammpstrj
from torchani.testing import TestCase
from pathlib import Path

path = Path(__file__).resolve().parent.joinpath('test_data/sample.xyz')

expect_species = torch.tensor([[1, 1, 8],
                               [1, 1, 8],
                               [1, 1, 8],
                               [1, 1, 8]], dtype=torch.long)

expect_coords = torch.tensor([[[1., 2., 3.],
                               [1., 2., 3.],
                               [1., 2., 3.]],
                              [[2., 2., 3.],
                               [2., 2., 3.],
                               [2., 2., 3.]],
                              [[3., 2., 3.],
                               [3., 2., 3.],
                               [3., 2., 3.]],
                              [[4., 2., 3.],
                               [4., 2., 3.],
                               [4., 2., 3.]]], dtype=torch.float)


class TestUtils(TestCase):

    def testToXyz(self):
        with tempfile.NamedTemporaryFile(mode='r+') as f:
            tensor_to_xyz(f, (expect_species, expect_coords))
            print(expect_species)
            species, coordinates, cell = tensor_from_xyz(f, step=1)

    def testFromXyz(self):
        species, coordinates, cell = tensor_from_xyz(path, step=1)
        self.assertEqual(coordinates, expect_coords)
        self.assertEqual(species, expect_species)

    def testToLammps(self):
        species, coordinates, cell = tensor_from_xyz(path, get_cell=False, step=1)
        tensor_to_lammpstrj(path.with_suffix('.lammpstrj'), (species, coordinates),
                            cell=torch.eye(3).unsqueeze(0).repeat(len(species), 1, 1), truncate_output_file=True)

    def testFromLammps(self):
        species, coordinates, cell, _, _ = tensor_from_lammpstrj(path.with_suffix('.lammpstrj'), get_cell=True, extract_atoms=(0, 2))


if __name__ == '__main__':
    unittest.main()
