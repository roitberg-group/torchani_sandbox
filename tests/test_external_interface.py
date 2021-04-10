import torch
import torchani
import unittest
from torchani.testing import TestCase


class TestExternalInterface(TestCase):

    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model_interface = torchani.models.ANI1x(periodic_table_index=False, external_cell_list=True)
        self.model_interface = self.model_interface.to(device=self.device, dtype=torch.float)

        self.model = torchani.models.ANI1x(periodic_table_index=False)
        self.model = self.model.to(device=self.device, dtype=torch.float)

        cutoff = self.model.aev_computer.radial_terms.get_cutoff()
        self.neighborlist = torchani.aev.neighbors.FullPairwise(cutoff).to(device=self.device, dtype=torch.float)
        self.unscreened_neighborlist = torchani.aev.neighbors.FullPairwise(cutoff + 1.0)

    def testEnergisEqualWithExternal(self):
        N = 25

        for j in range(N):
            c = torch.randn((3, 10, 3), dtype=torch.float, device=self.device)
            s = torch.randint(low=0, high=4, size=(3, 10), dtype=torch.long, device=self.device)
            e_expect = self.model((s, c)).energies
            neighborlist, shift_values, _, _ = self.neighborlist(s, c)
            e = self.model_interface((s, c), neighborlist, shift_values).energies
            self.assertEqual(e_expect, e)


if __name__ == '__main__':
    unittest.main()
