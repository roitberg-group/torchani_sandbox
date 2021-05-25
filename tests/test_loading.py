import unittest
import torchani
from torchani.testing import TestCase


class TestLoading(TestCase):
    # check that models loaded from a neurochem source are equal to models
    # loaded directly from a state_dict

    def testANI1x(self):
        model = torchani.models.ANI1x(periodic_table_index=True, use_neurochem_source=False)
        model_nc = torchani.models.ANI1x(periodic_table_index=True, use_neurochem_source=True)
        for p1, p2 in zip(model_nc.state_dict(), model.state_dict()):
            self.assertEqual(p1, p2)

    def testANI2x(self):
        model = torchani.models.ANI2x(periodic_table_index=True, use_neurochem_source=False)
        model_nc = torchani.models.ANI2x(periodic_table_index=True, use_neurochem_source=True)
        for p1, p2 in zip(model_nc.parameters(), model.parameters()):
            self.assertEqual(p1, p2)

    def testANI1ccx(self):
        model = torchani.models.ANI1ccx(periodic_table_index=True, use_neurochem_source=False)
        model_nc = torchani.models.ANI1ccx(periodic_table_index=True, use_neurochem_source=True)
        for p1, p2 in zip(model_nc.parameters(), model.parameters()):
            self.assertEqual(p1, p2)


if __name__ == '__main__':
    unittest.main()
