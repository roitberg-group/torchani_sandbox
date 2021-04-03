import torch
import torchani
import unittest
from torchani.testing import TestCase
from torchani.dispersion import DispersionD3, StandaloneDispersionD3, constants
from torchani.aev import AEVComputerForRepulsion
from torchani import units


class TestDispersion(TestCase):
    def setUp(self):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        ani1x_values = {
            'radial_cutoff': 5.2,
            'angular_cutoff': 3.5,
            'radial_eta': 16.0,
            'angular_eta': 8.0,
            'radial_dist_divisions': 16,
            'angular_dist_divisions': 4,
            'zeta': 32.0,
            'angle_sections': 8,
            'num_species': 4
        }
        self.aev_computer = AEVComputerForRepulsion.cover_linearly(
            **ani1x_values).double().to(self.device)
        self.converter = torchani.SpeciesConverter(['H', 'C', 'N', 'O'
                                                    ]).double().to(self.device)
        # fully symmetric methane
        self.coordinates = torch.tensor(
            [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [0.5, 0.5, 0.5]],
            dtype=torch.double,
            device=self.device).unsqueeze(0)
        self.species = torch.tensor([[1, 1, 1, 1, 6]],
                                    dtype=torch.long,
                                    device=self.device)
        # for the purposes of testing we use the exact same conversion factors
        # as the original DFTD3 code
        self.old_angstrom_to_bohr = units.ANGSTROM_TO_BOHR
        self.old_hartree_to_kcalmol = units.HARTREE_TO_KCALMOL
        units.ANGSTROM_TO_BOHR = 1 / 0.52917726
        units.HARTREE_TO_KCALMOL = 627.509541

    def tearDown(self):
        # reset conversion factors
        units.ANGSTROM_TO_BOHR = self.old_angstrom_to_bohr
        units.HARTREE_TO_KCALMOL = self.old_hartree_to_kcalmol

    def testConstructor(self):
        species_idx = self.converter((self.species, self.coordinates)).species
        species, aevs, atom_index12, distances =\
                self.aev_computer((species_idx, self.coordinates))
        disp = DispersionD3()
        self.assertTrue(disp.s6 == torch.tensor(1.))

    def testMethaneCoordinationNums(self):
        species_idx = self.converter((self.species, self.coordinates)).species
        _, _, atom_index12, distances =\
                self.aev_computer((species_idx, self.coordinates))
        disp = DispersionD3().to(self.device)
        distances = units.angstrom2bohr(distances)
        coordnums = disp._get_coordnums(self.coordinates.shape[1],
                                        self.species.flatten()[atom_index12],
                                        atom_index12, distances)
        # coordination numbers taken directly from DFTD3 Grimme et. al. code
        self.assertTrue(
            torch.isclose(
                coordnums.cpu(),
                torch.tensor(
                    [1.0052222, 1.0052222, 1.0052222, 1.0052222,
                     3.999873048]).double()).all())

    def testPrecomputedC6(self):
        species_idx = self.converter((self.species, self.coordinates)).species
        _, _, atom_index12, distances =\
                self.aev_computer((species_idx, self.coordinates))
        disp = DispersionD3().to(self.device)
        diags = {
            'H': torch.diag(disp.precalc_order6_coeffs[1, 1]),
            'C': torch.diag(disp.precalc_order6_coeffs[6, 6]),
            'O': torch.diag(disp.precalc_order6_coeffs[7, 7]),
            'N': torch.diag(disp.precalc_order6_coeffs[8, 8])
        }
        # values from DFTD3 Grimme et. al. code
        expect_diags = {
            'H':
            torch.tensor([
                3.0267000198, 7.5915999413, -1.0000000000, -1.0000000000,
                -1.0000000000
            ]),
            'C':
            torch.tensor([
                49.1129989624, 43.2452011108, 29.3602008820, 25.7808990479,
                18.2066993713
            ]),
            'O':
            torch.tensor([
                25.2684993744, 22.1240997314, 19.6767997742, 15.5817003250,
                -1.0000000000
            ]),
            'N':
            torch.tensor([
                15.5059003830, 12.8161001205, 10.3708000183, -1.0000000000,
                -1.0000000000
            ])
        }
        for k in diags.keys():
            self.assertTrue(
                torch.isclose(diags[k].cpu(), expect_diags[k]).all())

    def testAllCarbonConstants(self):
        c6_constants, _, _ = constants.get_c6_constants()
        expect_c6_carbon = torch.tensor(
            [[49.1130, 46.0681, 37.8419, 35.4129, 29.2830],
             [46.0681, 43.2452, 35.5219, 33.2540, 27.5206],
             [37.8419, 35.5219, 29.3602, 27.5063, 22.9517],
             [35.4129, 33.2540, 27.5063, 25.7809, 21.5377],
             [29.2830, 27.5206, 22.9517, 21.5377, 18.2067]])
        self.assertTrue(
            torch.isclose(expect_c6_carbon, c6_constants[6, 6]).all())

    def testMethaneC6(self):
        species_idx = self.converter((self.species, self.coordinates)).species
        _, _, atom_index12, distances =\
                self.aev_computer((species_idx, self.coordinates))
        disp = DispersionD3().to(self.device)
        distances = units.angstrom2bohr(distances)
        species12 = self.species.flatten()[atom_index12]
        coordnums = disp._get_coordnums(self.coordinates.shape[1], species12,
                                        atom_index12, distances)
        order6_coeffs = disp._interpolate_order6_coeffs(
            species12, coordnums, atom_index12)
        # C6 coefficients taken directly from DFTD3 Grimme et. al. code
        expect_order6 = torch.tensor([
            3.0882003, 3.0882003, 3.0882003, 7.4632792, 3.0882003, 3.0882003,
            7.4632792, 3.0882003, 7.4632792, 7.4632792
        ]).double()
        self.assertTrue(
            torch.isclose(order6_coeffs.cpu(), expect_order6).all())

    def testMethaneEnergy(self):
        species_idx = self.converter((self.species, self.coordinates)).species
        _, _, atom_index12, distances =\
                self.aev_computer((species_idx, self.coordinates))
        energy = torch.tensor([0.0],
                              dtype=self.coordinates.dtype,
                              device=self.device)
        disp = DispersionD3().to(self.device)
        energy = disp((self.species, energy), atom_index12, distances).energies
        energy = units.hartree2kcalmol(energy)
        self.assertTrue(
            torch.isclose(energy.cpu(),
                          torch.tensor([-1.251336]).double()))

    def testMethaneStandalone(self):
        disp = StandaloneDispersionD3(neighborlist_cutoff=8.0).to(self.device)
        energy = disp((self.species, self.coordinates)).energies
        energy = units.hartree2kcalmol(energy)
        self.assertTrue(
            torch.isclose(energy.cpu(),
                          torch.tensor([-1.251336]).double()))

    def testForce(self):
        species_idx = self.converter((self.species, self.coordinates)).species
        self.coordinates.requires_grad_(True)
        _, _, atom_index12, distances =\
                self.aev_computer((species_idx, self.coordinates))
        energy = torch.tensor([0.0],
                              dtype=self.coordinates.dtype,
                              device=self.device)
        disp = DispersionD3().to(self.device)
        energy = disp((self.species, energy), atom_index12, distances).energies
        gradient = torch.autograd.grad(energy, self.coordinates)[0]
        gradient /= units.ANGSTROM_TO_BOHR
        # compare with analytical gradient from Grimme's DFTD3 (DFTD3 gives
        # gradient in Bohr)
        expect_grad = torch.tensor([[
            -0.42656701194940E-05, -0.42656701194940E-05, -0.42656701194940E-05
        ], [
            -0.42656701194940E-05, 0.42656701194940E-05, 0.42656701194940E-05
        ], [
            0.42656701194940E-05, -0.42656701194940E-05, 0.42656701194940E-05
        ], [
            0.42656701194940E-05, 0.42656701194940E-05, -0.42656701194940E-05
        ], [0.00000000000000E+00, 0.00000000000000E+00,
            0.00000000000000E+00]]).double()
        self.assertTrue(torch.isclose(expect_grad, gradient.cpu()).all())


if __name__ == '__main__':
    unittest.main()
