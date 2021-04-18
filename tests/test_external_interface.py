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

        self.cutoff = self.model.aev_computer.radial_terms.cutoff
        self.N = 20

    def testForcesEqualUnscreened(self):
        neighborlist = torchani.aev.neighbors.FullPairwise(self.cutoff + 1.0).to(device=self.device)
        self._testForcesEqualWithExternal(neighborlist, self.N)

    def testEnergiesEqualUnscreened(self):
        neighborlist = torchani.aev.neighbors.FullPairwise(self.cutoff + 1.0).to(device=self.device)
        self._testEnergiesEqualWithExternal(neighborlist, self.N)

    def testForcesEqualScreened(self):
        neighborlist = torchani.aev.neighbors.FullPairwise(self.cutoff).to(device=self.device)
        self._testForcesEqualWithExternal(neighborlist, self.N)

    def testEnergiesEqualScreened(self):
        neighborlist = torchani.aev.neighbors.FullPairwise(self.cutoff).to(device=self.device)
        self._testEnergiesEqualWithExternal(neighborlist, self.N)

    def _testForcesEqualWithExternal(self, neighborlist, N=20):

        for j in range(N):
            c = torch.randn((2, 30, 3), dtype=torch.float, device=self.device) * 6
            s = torch.randint(low=0, high=4, size=(2, 30), dtype=torch.long, device=self.device)

            c.requires_grad_(True)
            e_expect = self.model((s, c)).energies
            f_expect = -torch.autograd.grad(e_expect.sum(), c)[0]

            c = c.detach().requires_grad_(True)
            neighbors, shift_values, _, _ = neighborlist(s, c)
            e = self.model_interface((s, c), neighbors, shift_values).energies
            f = -torch.autograd.grad(e.sum(), c)[0]

            self.assertEqual(f_expect, f)

    def _testEnergiesEqualWithExternal(self, neighborlist, N=20):

        for j in range(N):
            c = torch.randn((2, 30, 3), dtype=torch.float, device=self.device) * 6
            s = torch.randint(low=0, high=4, size=(2, 30), dtype=torch.long, device=self.device)

            e_expect = self.model((s, c)).energies

            neighbors, shift_values, _, _ = neighborlist(s, c)
            e = self.model_interface((s, c), neighbors, shift_values).energies

            self.assertEqual(e_expect, e)


class TestExternalInterfaceJIT(TestExternalInterface):

    def setUp(self):
        super().setUp()
        # make the test faster due to JIT bug with dynamic shapes
        torch._C._jit_set_bailout_depth(1)
        self.model_interface = torch.jit.script(self.model_interface)
        self.N = 5


if __name__ == '__main__':
    unittest.main()
