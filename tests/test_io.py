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
            tensor_to_xyz(f.name, (expect_species, expect_coords), truncate_output_file=True)
            species, coordinates, cell = tensor_from_xyz(f.name, step=1)
            self.assertEqual(species, expect_species)
            self.assertEqual(coordinates, expect_coords)

    def testFromXyz(self):
        species, coordinates, cell = tensor_from_xyz(path, step=1)
        self.assertEqual(coordinates, expect_coords)
        self.assertEqual(species, expect_species)

    def testToLammps(self):
        with tempfile.NamedTemporaryFile(mode='r+') as f:
            tensor_to_lammpstrj(f.name, (expect_species, expect_coords),
                                cell=torch.eye(3).unsqueeze(0).repeat(len(expect_species), 1, 1), truncate_output_file=True)
            species, coordinates, cell, _, _ = tensor_from_lammpstrj(f.name, step=1)
            self.assertEqual(coordinates, expect_coords)
            self.assertEqual(species, expect_species)

    def testFromLammps(self):
        species, coordinates, cell, _, _ = tensor_from_lammpstrj(path.with_suffix('.lammpstrj'), get_cell=True)
        self.assertEqual(coordinates, expect_coords)
        self.assertEqual(species, expect_species)

    def testExtractFromLammps(self):
        species, coordinates, cell, _, _ = tensor_from_lammpstrj(path.with_suffix('.lammpstrj'), get_cell=True, extract_atoms=(0, 1))
        self.assertEqual(coordinates, expect_coords.index_select(1, torch.tensor([0, 1]).long()))
        self.assertEqual(species, expect_species.index_select(1, torch.tensor([0, 1]).long()))


if __name__ == '__main__':
    unittest.main()
