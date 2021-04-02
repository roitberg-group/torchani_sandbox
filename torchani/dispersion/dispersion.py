from typing import Tuple

import torch
from torch import Tensor

from .. import units
from ..nn import SpeciesEnergies
from ..aev import CutoffSmooth
from . import constants


class DampFunction(torch.nn.Module):
    # D3M modifies parameters AND damp function for zero-damp and only
    # parameters for BJ damp cutoff radii are used for damp functions
    def __init__(self, df_constants=None, modified=False):
        modified = torch.tensor(modified, dtype=torch.bool)
        self.register_buffer("use_modified_damp", modified)
        if df_constants is None:
            # by default constants are for bj damp, for the B97D density
            # functional
            functional = 'wB97X'
            if self.use_modified_damp:
                functional += '_modified'
            df_constants = constants.bj_damp[functional]

        df_constants = {k: torch.tensor(v) for k, v in df_constants.items()}
        self.register_buffer('sr6', df_constants.get('sr6', None))
        self.register_buffer('sr8', df_constants.get('sr8', None))
        self.register_buffer('a1', df_constants.get('a1', None))
        self.register_buffer('a2', df_constants.get('a2', None))
        self.register_buffer('beta', df_constants.get('beta', None))


class RationalDamp(DampFunction):
    def __init__(*args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.a1 is not None
        assert self.a2 is not None
        sqrt_q = constants.get_sqrt_empirical_charge()
        # for becke-johnson (BJ) damp, the cutoff radii are calculated directly
        # from the order 8 and order 6 coeffs. Note that the cutoff radii is a
        # matrix of T x T where T are the possible atom types and that these
        # cutoff radii are in AU (Bohr)
        cutoff_radii = torch.sqrt(3 * torch.outer(sqrt_q, sqrt_q))
        self.register_buffer('cutoff_radii', cutoff_radii)
        assert cutoff_radii.shape[0] == constants.SUPPORTED_D3_ELEMENTS
        assert cutoff_radii.shape[1] == constants.SUPPORTED_D3_ELEMENTS
        assert cutoff_radii.ndim == 2

    def forward(self, species12: Tensor, distances: Tensor,
                order: int) -> Tensor:
        cutoff_radii = self.cutoff_radii[species12[0], species12[1]]
        damp_term = (self.a1 * cutoff_radii + self.a2).pow(order)
        return distances.pow(order) + damp_term


class ZeroDamp(DampFunction):
    def __init__(*args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.beta is None:
            assert not modified
        assert self.sr6 is not None
        assert self.sr8 is not None
        # cutoff radii is a matrix of T x T where T are the possible atom types
        # and these cutoff radii are in Angstrom, so we convert to Bohr before
        # using
        cutoff_radii = units.angstrom2bohr(constants.get_cutoff_radii())
        self.register_buffer('cutoff_radii', cutoff_radii)
        assert cutoff_radii.shape[0] == constants.SUPPORTED_D3_ELEMENTS
        assert cutoff_radii.shape[1] == constants.SUPPORTED_D3_ELEMENTS
        assert cutoff_radii.ndim == 2

    def forward(self, species12: Tensor, distances: Tensor,
                order: int) -> Tensor:
        cutoff_radii = self.cutoff_radii[species12[0], species12[1]]
        if order == 6:
            alpha = 14
            s = self.sr6
        else:
            assert order == 8
            alpha = 16
            s = self.sr8

        inner_term = distances / (s * cutoff_radii)
        if self.use_modified_damp:
            # this is only added for the D3M(BJ) modified damp
            inner_term += cutoff_radii * self.beta

        return distances.pow(order) * (1 + 6 * inner_term).pow(-alpha)


class DispersionD3(torch.nn.Module):
    r"""Calculates the DFT-D3 dispersion corrections"""
    def __init__(self,
                 df_constants=None,
                 damp='rational',
                 modified_damp=False, use_three_body=False):

        super().__init__()
        # rational damp is becke-johnson
        assert not use_three_body, "Not yet implemented"
        assert damp in ['rational', 'zero'], 'Unsupported damp'

        self.register_buffer('s6', torch.tensor(df_constants.pop('s6')))
        self.register_buffer('s8', torch.tensor(df_constants.pop('s8')))

        if damp == 'rational':
            self.damp_function = RationalDamp(df_constants, modified_damp)
        else:
            self.damp_function = ZeroDamp(df_constants, modified_damp)

        order6_constants, coordnums_a, coordnums_b = get_c6_constants()
        self.register_buffer('precalc_order6_coeffs', order6_constants)
        self.register_buffer('precalc_coordnums_a', coordnums_a)
        self.register_buffer('precalc_coordnums_b', coordnums_b)
        self.register_buffer('covalent_radii', constants.get_covalent_radii())
        # the product of the sqrt of the empirical q's is stored directly
        sqrt_empirical_charge = constants.get_sqrt_empirical_charge()
        charge_ab = torch.outer(sqrt_empirical_charge, sqrt_empirical_charge)
        self.register_buffer('sqrt_charge_ab', charge_ab)

    def _get_coordnums(self, num_atoms: int, species12: Tensor,
                       atom_index12: Tensor, distances: Tensor) -> Tensor:

        covalent_radii_sum = self.covalent_radii[species12[0]]
        covalent_radii_sum += self.covalent_radii[species12[1]]
        # for coordination numbers covalent radii are used, not cutoff radii
        k1 = 16
        k2 = 4 / 3
        denom = 1 + torch.exp(-k1 * (k2 * covalent_radii_sum / distances - 1))
        counting_function = 1 / denom

        # add terms corresponding to all neighbors
        coordnums = torch.zeros(num_atoms,
                                device=distances.device,
                                dtype=distances.dtype)
        coordnums.index_add_(0, atom_index12[0], counting_function)
        coordnums.index_add_(0, atom_index12[1], counting_function)
        # coordination nums shape is (A,), there is one per atom
        return coordnums

    def _interpolate_order6_coeffs(self, species12: Tensor,
                                   coordnums: Tensor) -> Tensor:
        assert coordnums.ndim == 1, 'coordnums must be one dimensional'
        assert species12.ndim == 2, 'species12 must be 2 dimensional'

        num_atoms = len(coordnums)
        # find pre-computed values for every species pair
        precalc_order6 = self.precalc_order6_coeffs[species12[0], species12[1]]
        precalc_cn_a = self.precalc_coordnums_a[species12[0], species12[1]]
        precalc_cn_b = self.precalc_coordnums_b[species12[0], species12[1]]
        # the precalc order6 coeff
        # and the precalc cn's have shape (A, A, 5, 5)
        for t in (precalc_order6, precalc_cn_a, precalc_cn_b):
            assert t.shape[0] == num_atoms
            assert t.shape[1] == num_atoms
            assert t.shape[2] == 5
            assert t.shape[3] == 5
            assert t.ndim == 4

        k3 = 4
        gauss_dist = (coordnums.view(num_atoms, 1, 1, 1) - precalc_cn_a)**2
        gauss_dist += (coordnums.view(1, num_atoms, 1, 1) - precalc_cn_b)**2
        gauss_dist = torch.exp(-k3 * gauss_dist)
        # sum over references for w factor and z factor
        w_factor = gauss_dist.view(num_atoms, num_atoms, -1).sum(-1)
        z_factor = precalc_order6 * gauss_dist
        z_factor = z_factor.view(num_atoms, num_atoms, -1).sum(-1)
        order6_coeffs = z_factor / w_factor

        assert order6_coeffs.shape[0] == num_atoms
        assert order6_coeffs.shape[1] == num_atoms
        assert order6_coeffs.ndim == 2

        return order6_coeffs

    def forward(self, species_energies: Tensor, atom_index12: Tensor,
                distances: Tensor) -> Tuple[Tensor, Tensor]:
        # internally this module works in AU, so first we convert distances
        distances = units.angstrom2bohr(distances)

        species, energies = species_energies
        assert len(species) == 1, "Not implemented for batch calcs"
        assert distances.ndim == 1, "distances should be 1 dim"
        assert species.ndim == 2, "species_energies should be 2 dim"
        assert atom_index12.ndim == 2, "atom_index12 should be 2 dim"
        assert len(distances) == atom_index12.shape[1]

        # distances has all interaction pairs within a given cutoff, for a
        # molecule or set of molecules and atom_index12 holds all pairs of
        # indices species is of shape (C x Atoms)
        num_atoms = species.shape[1]
        species12 = species.flatten()[atom_index12]

        # use the coordination numbers and the internal precalc C6's and
        # CNa's/CNb's to get interpolated C6 coeffs, C8 coeffs are obtained
        # from C6 directly
        coordnums = self._get_coordnums(num_atoms, species12, atom_index12,
                                        distances)
        order6_coeffs = self._interpolate_order6_coeffs(species12, coordnums)
        order8_coeffs = 3 * order6_coeffs
        order8_coeffs *= self.sqrt_charge_ab[species12[0], species12[1]]

        distances_damp6 = self.damp_function(species12, distances, 6)
        distances_damp8 = self.damp_function(species12, distances, 8)

        two_body_dispersion = self.s6 * order6_coeffs / distances_damp6
        two_body_dispersion += self.s8 * order8_coeffs / distances_damp8
        three_body_dispersion = 0.0

        # factor of 1/2 is not needed for two body since we only add the
        # interacting pairs once
        dispersion_correction = -two_body_dispersion.sum()
        dispersion_correction += -(1 / 6) * three_body_dispersion

        energies += dispersion_correction
        return SpeciesEnergies(species, energies)
