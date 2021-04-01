from typing import Tuple

import torch
from torch import Tensor

from . import units
from .nn import SpeciesEnergies
from .aev import CutoffSmooth

# covalent radii are used for the calculation of coordination numbers
# covalent radii in angstrom taken directly from Grimme et. al. dftd3 source code, 
# in turn taken from Pyykko and Atsumi, Chem. Eur. J. 15, 2009, 188-197
# values for metals decreased by 10 %
covalent_radii = [0.32, 0.46, 1.20, 0.94, 0.77, 0.75, 0.71, 0.63, 0.64, 0.67
, 1.40, 1.25, 1.13, 1.04, 1.10, 1.02, 0.99, 0.96, 1.76, 1.54
, 1.33, 1.22, 1.21, 1.10, 1.07, 1.04, 1.00, 0.99, 1.01, 1.09
, 1.12, 1.09, 1.15, 1.10, 1.14, 1.17, 1.89, 1.67, 1.47, 1.39
, 1.32, 1.24, 1.15, 1.13, 1.13, 1.08, 1.15, 1.23, 1.28, 1.26
, 1.26, 1.23, 1.32, 1.31, 2.09, 1.76, 1.62, 1.47, 1.58, 1.57
, 1.56, 1.55, 1.51, 1.52, 1.51, 1.50, 1.49, 1.49, 1.48, 1.53
, 1.46, 1.37, 1.31, 1.23, 1.18, 1.16, 1.11, 1.12, 1.13, 1.32
, 1.30, 1.30, 1.36, 1.31, 1.38, 1.42, 2.01, 1.81, 1.67, 1.58
, 1.52, 1.53, 1.54, 1.55 ]
covalent_radii = torch.tensor(covalent_radii)

assert len(covalent_radii) == 94



# constants for the density functional from psi4 source code, citations:
#    A. Najib, L. Goerigk, J. Comput. Theory Chem., 14 5725, 2018)
#    N. Mardirossian, M. Head-Gordon, Phys. Chem. Chem. Phys, 16, 9904, 2014
constants_bj_damping = {'wB97X' : {'s6' : 1.000, 'a1': 0.0000, 's8': 0.2641, 'a2': 5.4959}}

# modified D3M retains damping function, only modifies some parameters
# D3M-BJ modifies parameters AND damping function

class DampingFunction(torch.nn.Module):
    def __init__(self, constants=None, modified=False):

        self.register_buffer("use_modified_damping", torch.tensor(modified, dtype=torch.bool))
        if constants is None:
            # by default constnats are for bj damping, for the B97D density functional
            functional = 'wB97X'
            if self.use_modified_damping:
                functional += '_modified'
            s6 = constants_bj_damping[functional]['s6']
            s8 = constants_bj_damping[functional]['s8']
            a1 = constants_bj_damping[functional]['a1']
            a2 = constants_bj_damping[functional]['a2']
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

    def forward(self, distances: Tensor, cutoff_radii: Tensor, order: int):
        # assumes that the input are cutoff radii
        return distances.pow(order) + (self.a1 * cutoff_radii + self.a2).pow(order)


class ZeroDamping(DampingFunction):

    def __init__(*args, **kwargs):
        beta = kwargs.pop('beta', 0.0)
        super().__init__(*args, **kwargs)
        self.register_buffer("beta", torch.tensor(beta))


    def forward(self, distances: Tensor, cutoff_radii: Tensor, order: int):
        # assumes that the input are distances
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

        if damping == 'rational':
            # for becke-johnson damping the cutoff radii are calculated directly from the 
            # order 8 and order 6 coefficients
            # note that the cutoff radii is a matrix of T x T where T are the possible atom types
            self.register_buffer('cutoff_radii', torch.sqrt(self.order8_coeffs/self.order6_coeffs))
        else:
            # get the cutoff radii somehow
            raise NotImplementedError('The cutoff radii for zero cutoff are not yet implemented')

        # get the order6 and order8 coeffs somehow
        order6_coeffs = torch.tensor(0)
        order8_coeffs = torch.tensor(0)

        self.register_buffer('order_6coeffs', order_6coeffs)
        self.register_buffer('order_8coeffs', order_8coeffs)

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
        species12 = species.flatten()[atom_index12]

        # find pre-computed values for every species pair
        order6_coeffs = self.order6_coeffs[species12[0], species12[1]]
        order8_coeffs = self.order8_coeffs[species12[0], species12[1]]

        distances_damp6 = self.damping_function(distances, cutoff_radii, 6)
        distances_damp8 = self.damping_function(distances, cutoff_radii, 8)

        two_body_dispersion = self.s6 * order_6coeffs / distances_damp6 + self.s8 * order8_coeffs / distances_damp8
        three_body_dispersion = 0.0

        dispersion_correction = -((1 / 2) * two_body_dispersion + (1 / 6) * three_body_dispersion).sum()

        energies += dispersion_correction
        return SpeciesEnergies(species, energies)
