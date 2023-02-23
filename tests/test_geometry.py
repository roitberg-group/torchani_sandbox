from typing import Optional
import unittest

import torch

from torchani import units
from torchani import geometry
from torchani.testing import TestCase


class TestGeometry(TestCase):

    def testCenterToComFrame(self):
        species = torch.tensor([[1, 1, 1, 1, 6]], dtype=torch.long)
        coordinates = torch.tensor([[[0.0, 0.0, 0.0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [0.5, 0.5, 0.5]]], dtype=torch.float)
        displaced_coordinates = geometry.displace_to_com_frame(coordinates, atomic_numbers=species)
        self.assertEqual(displaced_coordinates, coordinates - torch.tensor([[0.5, 0.5, 0.5]]).unsqueeze(1))

    def testCenterToComFrameDummy(self):
        species = torch.tensor([[1, 1, 1, 1, 6, -1]], dtype=torch.long)
        coordinates = torch.tensor([[[0.0, 0.0, 0.0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [0.5, 0.5, 0.5], [0, 0, 0]]], dtype=torch.float)
        displaced_coordinates = geometry.displace_to_com_frame(coordinates, atomic_numbers=species)
        expect_coordinates = coordinates - torch.tensor([[0.5, 0.5, 0.5]]).unsqueeze(1)
        expect_coordinates[(species == -1)] = 0
        self.assertEqual(displaced_coordinates, expect_coordinates)

    def testCenterToComFrameMany(self):
        species = torch.tensor([[1, 1, 1, 1, 6, -1], [6, 6, 6, 6, 8, -1]], dtype=torch.long)
        coordinates = torch.tensor([[[0.0, 0.0, 0.0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [0.5, 0.5, 0.5], [0, 0, 0]]], dtype=torch.float)
        coordinates = torch.cat((coordinates, coordinates.clone()), dim=0)
        displaced_coordinates = geometry.displace_to_com_frame(coordinates, atomic_numbers=species)
        expect_coordinates = coordinates - torch.tensor([[0.5, 0.5, 0.5]]).unsqueeze(1)
        expect_coordinates[(species == -1)] = 0
        self.assertEqual(displaced_coordinates, expect_coordinates)

    def testCenterOfMassWaterOrca(self):
        # this was taken from ORCA 4.2
        species = torch.tensor([[1, 1, 8]], dtype=torch.long)
        coordinates = torch.tensor([[[0.600000, 0.000000, 0.000000],
                                    [-0.239517, 0.927144, 0.000000],
                                    [0, 0, 0]]], dtype=torch.float)
        displaced_coordinates = geometry.displace_to_com_frame(coordinates, atomic_numbers=species)
        # com = coordinates + displaced_coordinates
        expect_com = torch.tensor([[0.038116, 0.098033, 0.000000]])
        expect_com = units.bohr2angstrom(expect_com)
        self.assertEqual(displaced_coordinates, coordinates - expect_com)

    def _testStructures(self, diatomic: bool = True, idx: Optional[int] = None):
        species = torch.tensor([[1, 1, 8],
                                [1, -1, -1],
                                [1, 1, -1 if diatomic else 1]], dtype=torch.long)
        coordinates = torch.tensor([[[0.600000, 0.000000, 0.000000],
                                     [-0.239517, 0.927144, 0.000000],
                                     [0, 0, 0]],
                                    [[1.0, 0.0, 0.0],
                                     [0., 0., 0.],
                                     [0., 0., 0.]],
                                    [[1.0, 0.0, 0.0],
                                     [2., 0., 0.],
                                     [0. if diatomic else 3., 0., 0.]]], dtype=torch.float)
        if idx is not None:
            species = species[idx].unsqueeze(0)
            coordinates = coordinates[idx].unsqueeze(0)
        return species, coordinates

    def testAtomic(self):
        atomic_numbers, coordinates = self._testStructures()
        bool_ = geometry.structure_is_monoatomic(coordinates, atomic_numbers == -1)
        self.assertEqual(torch.tensor([False, True, False]), bool_)

    def testDiatomic(self):
        atomic_numbers, coordinates = self._testStructures()
        bool_ = geometry.structure_is_linear(coordinates, atomic_numbers == -1)
        self.assertEqual(torch.tensor([False, False, True]), bool_)

    def testPolyAtomicLinear(self):
        atomic_numbers, coordinates = self._testStructures(diatomic=False)
        bool_ = geometry.structure_is_linear(coordinates, atomic_numbers == -1)
        self.assertEqual(torch.tensor([False, False, True]), bool_)

    def testPolyatomicNonlinear(self):
        atomic_numbers, coordinates = self._testStructures(diatomic=False)
        bool_ = geometry.structure_is_polyatomic_nonlinear(coordinates, atomic_numbers == -1)
        self.assertEqual(torch.tensor([True, False, False]), bool_)

    def testInertiaTensor(self):
        atomic_numbers, coordinates = self._testStructures(diatomic=False)
        inertia_tensor = geometry.inertia_tensor(coordinates, atomic_numbers)
        expect_inertia = torch.tensor([[[0.81799078, 0.24269365, 0.00000000],
                                        [0.24269365, 0.41337830, 0.00000000],
                                        [0.00000000, 0.00000000, 1.23136902]],

                                       [[0.00000000, 0.00000000, 0.00000000],
                                        [0.00000000, 0.00000000, 0.00000000],
                                        [0.00000000, 0.00000000, 0.00000000]],

                                       [[0.00000000, 0.00000000, 0.00000000],
                                        [0.00000000, 2.01600027, 0.00000000],
                                        [0.00000000, 0.00000000, 2.01600027]]])
        self.assertEqual(expect_inertia, inertia_tensor)

    def testInertiaAxes(self):
        atomic_numbers, coordinates = self._testStructures(diatomic=False)
        evec_eval = geometry.principal_inertia_axes(coordinates, atomic_numbers)
        evectors = evec_eval.evectors
        torch.set_printoptions(precision=8)
        expect_evectors = torch.tensor([[[0.42408764, -0.90562111, 0.00000000],
                                          [-0.90562111, -0.42408764, 0.00000000],
                                          [0.00000000, -0.00000000, 1.00000000]],

                                         [[1.00000000, 0.00000000, 0.00000000],
                                          [0.00000000, 1.00000000, 0.00000000],
                                          [0.00000000, 0.00000000, 1.00000000]],

                                         [[1.00000000, 0.00000000, 0.00000000],
                                          [0.00000000, 1.00000000, 0.00000000],
                                          [0.00000000, 0.00000000, 1.00000000]]])
        evalues = evec_eval.evalues
        is_linear = torch.tensor([False, False, True])
        is_monoatomic = torch.tensor([False, True, False])
        is_polyatomic_nonlinear = torch.tensor([True, False, False])
        self.assertEqual(
            is_linear,
            geometry.inertia_is_linear(evalues)
        )
        self.assertEqual(
            is_polyatomic_nonlinear,
            geometry.inertia_is_polyatomic_nonlinear(evalues)
        )
        self.assertEqual(
            is_monoatomic,
            geometry.inertia_is_monoatomic(evalues)
        )
        self.assertEqual(expect_evectors, evectors)

    def testInternals(self):
        atomic_numbers, coordinates = self._testStructures(diatomic=False, idx=0)
        coordinates = coordinates.to("cpu")
        atomic_numbers = atomic_numbers.to("cpu")
        # the internal coordinates require a gram-schmidt decomposition,
        # which starts from random vectors, so a seed is needed to ensure
        # reproducibility
        internals = geometry.internals_matrix_transform(
            coordinates,
            atomic_numbers=atomic_numbers,
            orthonormality_check=True,
            geometry_check=True,
            seed=1234567890,
        )
        torch.set_printoptions(precision=8, linewidth=200)
        internals_expect = torch.tensor(
            [[[2.36544654e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, -4.69363853e-02, 3.31485659e-01, 9.11694765e-01, -2.78116446e-02],
              [0.00000000e+00, 2.36544654e-01, 0.00000000e+00, 0.00000000e+00, -0.00000000e+00, -5.24610221e-01, 2.05436066e-01, -1.25545353e-01, -7.81578779e-01],
              [0.00000000e+00, 0.00000000e+00, 2.36544654e-01, -9.22625363e-01, -3.04645717e-01, 0.00000000e+00, 3.68170490e-08, -2.15500535e-07, 1.69167345e-06],
              [2.36544654e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 7.91911781e-01, -9.26277861e-02, -3.86306434e-03, -5.55272341e-01],
              [0.00000000e+00, 2.36544654e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.34956130e-01, 8.73177826e-01, -2.99543113e-01, 1.91511571e-01],
              [0.00000000e+00, 0.00000000e+00, 2.36544654e-01, -2.49426410e-01, 9.39059734e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
              [9.42386925e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, -1.86993182e-01, -5.99547401e-02, -2.27871075e-01, 1.46357521e-01],
              [0.00000000e+00, 9.42386925e-01, 0.00000000e+00, 0.00000000e+00, -0.00000000e+00, 7.27048516e-02, -2.70738363e-01, 1.06699854e-01, 1.48109958e-01],
              [0.00000000e+00, 0.00000000e+00, 9.42386925e-01, 2.94191837e-01, -1.59241632e-01, -0.00000000e+00, -2.45446987e-08, -2.56062496e-08, -4.65349075e-07]]]
        )
        self.assertEqual(internals_expect, internals)


if __name__ == '__main__':
    unittest.main()
