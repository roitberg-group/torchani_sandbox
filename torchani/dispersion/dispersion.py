from typing import Tuple

import torch
from torch import Tensor

from .. import units
from ..nn import SpeciesEnergies
from ..aev import CutoffSmooth
from . import constants


class DampingFunction(torch.nn.Module):
    # modified D3M retains damping function, only modifies some parameters
    # D3M-BJ modifies parameters AND damping function
    # cutoff radii are used for damping functions
    def __init__(self, constants=None, modified=False):

        self.register_buffer("use_modified_damping", torch.tensor(modified, dtype=torch.bool))
        if constants is None:
            # by default constnats are for bj damping, for the B97D density functional
            functional = 'wB97X'
            if self.use_modified_damping:
                functional += '_modified'
            s6 = constants.bj_damping[functional]['s6']
            s8 = constants.bj_damping[functional]['s8']
            a1 = constants.bj_damping[functional]['a1']
            a2 = constants.bj_damping[functional]['a2']
        else:
            s6 = constants['s6']
            s8 = constants['s8']
            a1 = constants['a1']
            a2 = constants['a2']

        self.register_buffer('s6', torch.tensor(s6))
        self.register_buffer('s8', torch.tensor(s8))
        self.register_buffer('a1', torch.tensor(a1))
        self.register_buffer('a2', torch.tensor(a2))


class RationalDamping(DampingFunction):

    def __init__(*args, **kwargs):
        super().__init__(*args, **kwargs)
        sqrt_q = constants.get_sqrt_empirical_charge()
        # for becke-johnson damping the cutoff radii are calculated directly from the 
        # order 8 and order 6 coefficients
        # note that the cutoff radii is a matrix of T x T where T are the possible atom types
        # note that these cutoff radii are in AU (Bohr)
        self.register_buffer('cutoff_radii', torch.sqrt(3 * torch.outer(sqrt_q, sqrt_q)))

    def forward(self, species12: Tensor, distances: Tensor, order: int):
        cutoff_radii = self.cutoff_radii[species12[0], species12[1]]
        return distances.pow(order) + (self.a1 * cutoff_radii + self.a2).pow(order)


class ZeroDamping(DampingFunction):

    def __init__(*args, **kwargs):
        beta = kwargs.pop('beta', 0.0)
        super().__init__(*args, **kwargs)
        self.register_buffer("beta", torch.tensor(beta))
        cutoff_radii = constants.get_cutoff_radii()
        # note that these cutoff radii are in Angstrom
        self.register_buffer('cutoff_radii', cutoff_radii)

    def forward(self, species12: Tensor, distances: Tensor, order: int):
        cutoff_radii = self.cutoff_radii[species12[0], species12[1]]
        if order == 6:
            alpha = 14
            s = self.s6
        else:
            assert order == 8
            alpha = 16
            s = self.s8

        inner_term = distances / (self.s * cutoff_radii)

        if self.use_modified_damping:
            # this is only added for the D3M(BJ) modified damping
            inner_term += cutoff_radii * self.beta

        return distances.pow(order) * (1 + 6 * inner_term).pow(-self.alpha)


class DispersionD3(torch.nn.Module):
    """Calculates the DFT-D3 dispersion corrections repulsion energy terms for a given molecule"""

    def __init__(self, constants=None, damping='rational', modified_damping=False):

        super().__init__()
        # rational damping is becke-johnson 
        assert damping in ['rational', 'zero'], 'Unsupported damping'

        if damping == 'rational':
            self.damping_function = RationalDamping(constants, modified=modified_damping)
        else:
            self.damping_function = ZeroDamping(constants, modified=modified_damping)
            
        self.register_buffer('s6', self.damping_function.s6)
        self.register_buffer('s8', self.damping_function.s8)

        # get the order6 and order8 coeffs somehow
        order6_coeffs = torch.tensor(0)
        order8_coeffs = torch.tensor(0)
        c6_constants, c6_coordination_a, c6_coordination_b = get_c6_constants(supported_d3_elements)
        covalent_radii = constants.get_covalent_radii()
        sqrt_empirical_charge = constants.get_sqrt_empirical_charge()

        self.register_buffer('precomputed_order6_coeffs', c6_constants)
        self.register_buffer('precomputed_coordination_a', c6_coordination_a)
        self.register_buffer('precomputed_coordination_b', c6_coordination_b)
        self.register_buffer('covalent_radii', covalent_radii)
        # the product of the sqrt of the empirical q's is stored directly
        self.register_buffer('sqrt_empirical_charge_ab', torch.outer(sqrt_empirical_charge, sqrt_empirical_charge))

    def get_coordination_nums(self, num_atoms: int, species12: Tensor, atom_index12: Tensor, distances: Tensor):

        covalent_radii_sum = self.covalent_radii[species12[0]] + self.covalent_radii[species12[1]]
        # for coordination numbers covalent radii are used, not cutoff radii
        k1 = 16
        k2 = 4/3

        counting_function = 1/(1 + torch.exp(-k1 * (k2 * covalent_radii_sum / distances - 1)))

        # add terms corresponding to all neighbors
        coordination_nums = torch.zeros(num_atoms, device=distances.device, dtype=distances.dtype)
        coordination_nums.index_add_(0, atom_index12[0], counting_function)
        coordination_nums.index_add_(0, atom_index12[1], counting_function)
        # coordination nums shape is (A,), there is one per atom
        return coordination_nums

    def interpolate_c6_coefficients(self, species12: Tensor, coordination_nums: Tensor):
        assert coordination_nums.ndim == 1, 'coordination nums must be one dimensional'
        assert species12.ndim == 2, 'species12 must be 2 dimensional'

        # find pre-computed values for every species pair
        # the precomputed order6 coeff 
        # and the precomputed cn's have shape (A, A, 5, 5)
        precomputed_order6 = self.precomputed_order6_coeffs[species12[0], species12[1]]
        precomputed_cn_a = self.precomputed_coordination_a[species12[0], species12[1]]
        precomputed_cn_b = self.precomputed_coordination_b[species12[0], species12[1]]
        num_atoms = len(coordination_nums)
        k3 = 4
        gaussian_distance = (coordination_nums.view(num_atoms, 1, 1, 1) - precomputed_cn_a) ** 2
        gaussian_distance += (coordination_nums.view(1, num_atoms, 1, 1) - precomputed_cn_b) ** 2
        gaussian_distance = torch.exp(-k3 * gaussian_distance)
        # sum over references
        w_factor = gaussian_distance.view(num_atoms, num_atoms, -1).sum(-1) 

        z_factor = order6_coeffs 
        # here gaussian distance should be a tensor of shape (A, A, 5, 5)
        # where A is the number of atoms, and there are 5 x 5 possible references

    def forward(self, species_energies: Tensor, atom_index12: Tensor, distances: Tensor) -> Tuple[Tensor, Tensor]:

        species, energies = species_energies
        assert len(species) == 1, "Not implemented for batch calculations"
        assert distances.ndim == 1, "distances should be 1 dimensional"
        assert species.ndim == 2, "species_energies should be 2 dimensional"
        assert atom_index12.ndim == 2, "atom_index12 should be 2 dimensional"
        assert len(distances) == atom_index12.shape[1]

        # distances has all interaction pairs within a given cutoff,
        # for a molecule or set of molecules
        # and atom_index12 holds all pairs of indices
        # species is of shape (C x Atoms)
        num_atoms = species.shape[1]
        species12 = species.flatten()[atom_index12]

        coordination_nums = self.get_coordination_nums(num_atoms, species12, atom_index12, distances)
        interpolated_c6 = self.interpolate_c6_coefficients(species12, coordination_nums)

        distances_damp6 = self.damping_function(species12, distances, 6)
        distances_damp8 = self.damping_function(species12, distances, 8)

        two_body_dispersion = self.s6 * order6_coeffs / distances_damp6 + self.s8 * order8_coeffs / distances_damp8
        three_body_dispersion = 0.0
        
        # factor of 1/2 is not needed for two body since we only add the
        # interacting pairs once
        dispersion_correction = -(two_body_dispersion.sum() + (1 / 6) * three_body_dispersion.sum())

        energies += dispersion_correction
        return SpeciesEnergies(species, energies)
