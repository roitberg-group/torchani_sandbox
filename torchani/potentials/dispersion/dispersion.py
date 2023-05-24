import warnings
import math
from typing import Union

import torch
from torch import Tensor

from torchani.standalone import StandaloneWrapper
from torchani.units import ANGSTROM_TO_BOHR
from torchani.utils import ATOMIC_NUMBERS
from torchani.dispersion import constants
from torchani.cutoffs import Cutoff
from torchani.neighbors import NeighborData
from torchani.potentials import PairwisePotential


def _init_df_constants(df_constants, functional):
    assert not (df_constants and functional)
    if (df_constants is None) and (functional is None):
        # By default constants are for BJ damp, for the wB97X density
        # functional
        return constants.get_df_constants()['wB97X']
    elif functional is not None:
        return constants.get_df_constants()[functional]
    else:
        return df_constants


# D3M modifies parameters AND damp function for zero-damp and only
# parameters for BJ damp cutoff radii are used for damp functions
class DampFunction(torch.nn.Module):
    a1: Union[Tensor, None]
    a2: Union[Tensor, None]
    sr6: Union[Tensor, None]
    sr8: Tensor
    beta: Union[Tensor, None]

    def __init__(self, functional=None, df_constants=None, modified=False):
        super().__init__()
        modified = torch.tensor(modified, dtype=torch.bool)
        self.register_buffer("use_modified_damp", modified)

        df_constants = _init_df_constants(df_constants, functional)
        df_constants = {k: torch.tensor(v) for k, v in df_constants.items()}

        self.register_buffer('sr6', df_constants.get('sr6', None))
        self.register_buffer('sr8', torch.tensor(1.0))
        self.register_buffer('a1', df_constants.get('a1', None))
        self.register_buffer('a2', df_constants.get('a2', None))
        self.register_buffer('beta', df_constants.get('beta', None))


# AKA becke-johnson damping scheme
class RationalDamp(DampFunction):
    cutoff_radii: Tensor

    def __init__(self, *args, **kwargs):
        elements = kwargs.pop('elements', ('H', 'C', 'N', 'O'))
        super().__init__(*args, **kwargs)
        supported_znumbers = torch.tensor([ATOMIC_NUMBERS[e] for e in elements], dtype=torch.long)
        assert self.a1 is not None
        assert self.a2 is not None
        sqrt_q = constants.get_sqrt_empirical_charge()
        # for becke-johnson (BJ) damp, the cutoff radii are calculated directly
        # from the order 8 and order 6 coeffs. Note that the cutoff radii is a
        # matrix of T x T where T are the possible atom types and that these
        # cutoff radii are in AU (Bohr)
        cutoff_radii = torch.sqrt(3 * torch.outer(sqrt_q, sqrt_q))
        self.register_buffer('cutoff_radii', cutoff_radii[:, supported_znumbers][supported_znumbers, :])
        assert cutoff_radii.shape[0] == constants.SUPPORTED_D3_ELEMENTS + 1
        assert cutoff_radii.shape[1] == constants.SUPPORTED_D3_ELEMENTS + 1
        assert cutoff_radii.ndim == 2

    def forward(self, species12: Tensor, distances: Tensor,
                order: int) -> Tensor:
        cutoff_radii = self.cutoff_radii[species12[0], species12[1]]
        assert cutoff_radii.ndim == 1
        assert len(cutoff_radii) == species12.shape[1]
        damp_term = (self.a1 * cutoff_radii + self.a2).pow(order)
        return distances.pow(order) + damp_term


class ZeroDamp(DampFunction):
    cutoff_radii: Tensor

    def __init__(self, *args, **kwargs):
        elements = kwargs.pop('elements', ('H', 'C', 'N', 'O'))
        super().__init__(*args, **kwargs)
        supported_znumbers = torch.tensor([ATOMIC_NUMBERS[e] for e in elements], dtype=torch.long)

        if self.beta is None:
            assert not kwargs['modified']
        assert self.sr6 is not None
        assert self.sr8 is not None
        assert self.sr8 == 1.0  # this is fixed for all functionals
        # cutoff radii is a matrix of T x T where T are the possible atom types
        # and these cutoff radii are in Angstrom, so we convert to Bohr before
        # using
        cutoff_radii = ANGSTROM_TO_BOHR * constants.get_cutoff_radii()
        self.register_buffer('cutoff_radii', cutoff_radii[:, supported_znumbers][supported_znumbers, :])
        assert cutoff_radii.shape[0] == constants.SUPPORTED_D3_ELEMENTS + 1
        assert cutoff_radii.shape[1] == constants.SUPPORTED_D3_ELEMENTS + 1
        assert cutoff_radii.ndim == 2

    def forward(self, species12: Tensor, distances: Tensor,
                order: int) -> Tensor:
        cutoff_radii = self.cutoff_radii[species12[0], species12[1]]
        assert cutoff_radii.ndim == 1
        assert len(cutoff_radii) == species12.shape[1]
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

        return distances.pow(order) * (1 + (6 * inner_term).pow(-alpha))


class TwoBodyDispersionD3(PairwisePotential):
    r"""Calculates the DFT-D3 dispersion corrections"""
    s6: Tensor
    s8: Tensor
    covalent_radii: Tensor
    precalc_coordnums_a: Tensor
    precalc_coordnums_b: Tensor
    sqrt_charge_ab: Tensor
    precalc_order6_coeffs: Tensor

    def __init__(
        self,
        df_constants=None,
        functional=None,
        damp='rational',
        modified_damp: bool = False,
        use_three_body: bool = False,
        cutoff_fn: Union[str, Cutoff] = "dummy",
        damp_fn: str = "rational",
        cutoff=math.inf,
        elements=('H', 'C', 'N', 'O'),
    ):
        super().__init__(cutoff=cutoff, cutoff_fn=cutoff_fn)
        supported_znumbers = torch.tensor([ATOMIC_NUMBERS[e] for e in elements], dtype=torch.long)
        # rational damp is becke-johnson
        assert not use_three_body, "Not yet implemented"
        assert damp in ['rational', 'zero'], 'Unsupported damp'
        df_constants = _init_df_constants(df_constants, functional)

        if damp_fn == "rational":
            self.register_buffer('s6', torch.tensor(df_constants['s6_bj']))
            self.register_buffer('s8', torch.tensor(df_constants['s8_bj']))
            self.damp_function = RationalDamp(df_constants=df_constants, modified=modified_damp, elements=elements)
        else:
            assert damp_fn == "zero"
            self.register_buffer('s6', torch.tensor(df_constants['s6_zero']))
            self.register_buffer('s8', torch.tensor(df_constants['s8_zero']))
            self.damp_function = ZeroDamp(df_constants=df_constants, modified=modified_damp, elements=elements)

        if self.s6 != 1.0:
            warnings.warn(
                "The s6 parameter is not set to 1 in D3, Are you sure this is what you want? "
                "Usually s6 should be set to 1.0 except for B2PLYP, where it is set to 0.5"
            )
        order6_constants, coordnums_a, coordnums_b = constants.get_c6_constants()
        self.register_buffer('precalc_order6_coeffs', order6_constants[supported_znumbers, :][:, supported_znumbers])
        self.register_buffer('precalc_coordnums_a', coordnums_a[supported_znumbers, :][:, supported_znumbers])
        self.register_buffer('precalc_coordnums_b', coordnums_b[supported_znumbers, :][:, supported_znumbers])

        # covalent radii are in angstrom so we first convert to bohr
        supported_znumbers = torch.tensor([ATOMIC_NUMBERS[e] for e in elements], dtype=torch.long)
        covalent_radii = ANGSTROM_TO_BOHR * constants.get_covalent_radii()
        self.register_buffer('covalent_radii', covalent_radii[supported_znumbers])

        # the product of the sqrt of the empirical q's is stored directly
        sqrt_empirical_charge = constants.get_sqrt_empirical_charge()
        charge_ab = torch.outer(sqrt_empirical_charge, sqrt_empirical_charge)
        self.register_buffer('sqrt_charge_ab', charge_ab[supported_znumbers, :][:, supported_znumbers])

    def _get_coordnums(self, num_molecules: int, num_atoms: int, species12: Tensor,
                       atom_index12: Tensor, distances: Tensor) -> Tensor:
        # fine for batches
        covalent_radii_sum = self.covalent_radii[species12[0]]
        covalent_radii_sum += self.covalent_radii[species12[1]]
        # for coordination numbers covalent radii are used, not cutoff radii
        k1 = 16
        k2 = 4 / 3
        # fine for batches
        denom = 1 + torch.exp(-k1 * (k2 * covalent_radii_sum / distances - 1))
        counting_function = 1 / denom

        # add terms corresponding to all neighbors
        coordnums = distances.new_zeros((num_molecules * num_atoms))
        coordnums.index_add_(0, atom_index12[0], counting_function)
        coordnums.index_add_(0, atom_index12[1], counting_function)
        # coordination nums shape is (A,), there is one per atom
        return coordnums

    def _interpolate_order6_coeffs(self, species12: Tensor,
            coordnums: Tensor, atom_index12: Tensor) -> Tensor:
        assert coordnums.ndim == 1, 'coordnums must be one dimensional'
        assert species12.ndim == 2, 'species12 must be 2 dimensional'
        num_pairs = species12.shape[1]
        # find pre-computed values for every species pair, and flatten over all references
        precalc_order6 = self.precalc_order6_coeffs[species12[0], species12[1]]
        precalc_cn_a = self.precalc_coordnums_a[species12[0], species12[1]]
        precalc_cn_b = self.precalc_coordnums_b[species12[0], species12[1]]
        for t in [precalc_order6, precalc_cn_a, precalc_cn_b]:
            # the precalc order6 coeffs
            # and the precalc cn's have shape (Nb, 5, 5)
            # where Nb is the number of neighbors
            assert t.shape[0] == num_pairs
            assert t.shape[1] == 5
            assert t.shape[2] == 5
            assert t.ndim == 3
        # flattened shapes are (Nb, 25)
        precalc_cn_a = precalc_cn_a.flatten(1, 2)
        precalc_cn_b = precalc_cn_b.flatten(1, 2)
        precalc_order6 = precalc_order6.flatten(1, 2)

        k3 = 4
        gauss_dist = (coordnums[atom_index12[0]].view(-1, 1) - precalc_cn_a)**2
        gauss_dist += (coordnums[atom_index12[1]].view(-1, 1) - precalc_cn_b)**2
        # Extra factor of gauss_dist.mean() and + 20 needed for numerical stability
        gauss_dist = torch.exp(-k3 * gauss_dist)
        # only consider C6 coefficients strictly greater than zero,
        # don't include -1 and 0.0 terms in the sums,
        # all missing parameters (with -1.0 values) are guaranteed to be the
        # same for precalc_cn_a/b and precalc_order6
        gauss_dist = gauss_dist.masked_fill(precalc_order6 <= 0.0, 0.0)
        # sum over references for w factor and z factor
        # This is needed for numerical stability, it will give 1 if W or Z are not
        # >> 1e-35 but those situations are rare in practice, and it avoids all potential
        # issues with NaN and exploding numbers / vanishing quantities
        w_factor = gauss_dist.sum(-1) + 1e-35
        z_factor = (precalc_order6 * gauss_dist).sum(-1) + 1e-35
        order6_coeffs = z_factor / w_factor
        return order6_coeffs

    def pair_energies(self, element_idxs: Tensor, neighbors: NeighborData) -> Tensor:
        # internally this module works in AU, so first we convert distances
        distances = units.angstrom2bohr(neighbors.distances)
        assert distances.ndim == 1, "distances should be 1 dim"
        assert element_idxs.ndim == 2, "species_energies should be 2 dim"
        assert neighbors.indices.ndim == 2, "atom_index12 should be 2 dim"
        assert distances.shape[0] == neighbors.indices.shape[1]

        # distances has all interaction pairs within a given cutoff, for a
        # molecule or set of molecules and atom_index12 holds all pairs of
        # indices. species is of shape (C x Atoms)
        species12 = element_idxs.flatten()[neighbors.indices]
        num_atoms = element_idxs.shape[1]
        num_molecules = element_idxs.shape[0]
        num_pairs = species12.shape[1]

        # use the coordination numbers and the internal precalc C6's and
        # CNa's/CNb's to get interpolated C6 coeffs, C8 coeffs are obtained
        # from C6 coeffs directly
        coordnums = self._get_coordnums(num_molecules, num_atoms, species12, neighbors.indices,
                                        distances)
        # coordnums is shape num_molecules * num_atoms
        order6_coeffs = self._interpolate_order6_coeffs(species12, coordnums, neighbors.indices)
        order8_coeffs = 3 * order6_coeffs
        order8_coeffs *= self.sqrt_charge_ab[species12[0], species12[1]]
        distances_damp6 = self.damp_function(species12, distances, 6)
        distances_damp8 = self.damp_function(species12, distances, 8)
        for t in [distances_damp6, distances_damp8, order6_coeffs, order8_coeffs]:
            assert t.ndim == 1
            assert t.shape[0] == num_pairs

        order6_energy = self.s6 * order6_coeffs / distances_damp6
        order8_energy = self.s8 * order8_coeffs / distances_damp8
        return -(order6_energy + order8_energy)


def StandaloneTwoBodyDispersionD3(
    cutoff: float = 5.2,
    alpha: Sequence[float] = None,
    y_eff: Sequence[float] = None,
    k_rep_ab: Optional[Tensor] = None,
    symbols: Sequence[str] = ('H', 'C', 'N', 'O'),
    cutoff_fn: Union[str, Cutoff] = 'smooth',
    **standalone_kwargs,
) -> StandaloneWrapper:

    module = TwoBodyDispersionD3(
        alpha=alpha,
        y_eff=y_eff,
        k_rep_ab=k_rep_ab,
        cutoff=cutoff,
        symbols=symbols,
        cutoff_fn=cutoff_fn
    )
    return StandaloneWrapper(module, **standalone_kwargs)
