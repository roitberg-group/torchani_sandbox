import math
from typing import Union, Optional, Sequence, Type

import torch
from torch import Tensor
from torch.jit import Final

from torchani.standalone import StandaloneWrapper
from torchani.units import ANGSTROM_TO_BOHR
from torchani.utils import ATOMIC_NUMBERS
from torchani.dispersion import constants
from torchani.cutoffs import Cutoff
from torchani.neighbors import NeighborData
from torchani.potentials import PairwisePotential


# D3M modifies parameters AND damp function for zero-damp and only
# parameters for BJ damp cutoff radii are used for damp functions
class Damp(torch.nn.Module):
    r"""Damp function interface

    Damp functions are like cutoff functions, but modulate potentials close to
    zero.

    For modulating potentials of different "order" (e.g. 1 / r ** 6 => order 6),
    different parameters may be needed.
    """

    _order: Final[int]
    atomic_numbers: Tensor

    def __init__(
        self,
        *args,
        symbols: Sequence[str] = ('H', 'C', 'N', 'O'),
        order: int = 6,
        **kwargs,
    ):
        super().__init__()
        self._order = order
        self.atomic_numbers = torch.tensor(
            [ATOMIC_NUMBERS[e] for e in symbols],
            dtype=torch.long
        )

    @classmethod
    def from_functional(cls, functional: str = "wB97X", **kwargs) -> "Damp":
        raise NotImplementedError()

    def forward(self, species12: Tensor, distances: Tensor) -> Tensor:
        raise NotImplementedError()


class BJDamp(Damp):
    r"""Implementation of Becke-Johnson style damping

    For this damping style, the cutoff radii are by default calculated directly
    from the order 8 and order 6 coeffs, via the square root of the effective
    charges. Note that the cutoff radii is a matrix of T x T where T are the
    possible atom types and that these cutoff radii are in AU (Bohr)
    """
    cutoff_radii: Tensor
    _a1: Final[float]
    _a2: Final[float]

    def __init__(
        self,
        a1: float,
        a2: float,
        *args,
        cutoff_radii: Optional[Tensor] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        sqrt_q = constants.get_sqrt_empirical_charge()

        znumbers = self.atomic_numbers
        if cutoff_radii is None:
            _cutoff_radii = torch.sqrt(
                3 * torch.outer(sqrt_q, sqrt_q)
            )[:, znumbers][znumbers, :]
        else:
            _cutoff_radii = cutoff_radii

        # Cutoff radii is a matrix of T x T where T are the supported elements.
        assert _cutoff_radii.shape == (len(znumbers), len(znumbers))

        self.register_buffer('cutoff_radii', _cutoff_radii)
        self._a1 = a1
        self._a2 = a2

    @classmethod
    def from_functional(
        cls,
        functional: str = "wB97X",
        modified_damp: bool = False,
        **kwargs,
    ) -> "BJDamp":
        if modified_damp:
            raise ValueError("Modified damp is not yet implemented")
        d = constants.get_functional_constants()[functional or 'wB97X']
        return cls(a1=d["a1"], a2=d["a2"], cutoff_radii=None, **kwargs)

    def forward(
        self,
        species12: Tensor,
        distances: Tensor,
    ) -> Tensor:
        cutoff_radii = self.cutoff_radii[species12[0], species12[1]]
        damp_term = (self._a1 * cutoff_radii + self._a2).pow(self._order)
        return distances.pow(self._order) + damp_term


class ZeroDamp(Damp):
    r"""Zero-style damping

    Sometimes this is useful, but it may have some artifacts.
    TODO: This damping is untested
    """

    cutoff_radii: Tensor
    _sr: Final[float]
    _beta: Final[float]
    _alpha: Final[int]

    def __init__(
        self,
        alpha: int,
        sr: float,
        *args,
        beta: float = 0.0,
        cutoff_radii: Optional[Tensor] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        znumbers = self.atomic_numbers
        if cutoff_radii is None:
            # These cutoff radii are in Angstrom, so we convert to Bohr.
            _cutoff_radii = ANGSTROM_TO_BOHR * constants.get_cutoff_radii()
            _cutoff_radii = _cutoff_radii[:, znumbers][znumbers, :]
        else:
            _cutoff_radii = cutoff_radii

        # Cutoff radii is a matrix of T x T where T are the supported elements.
        assert _cutoff_radii.shape == (len(znumbers), len(znumbers))

        self._sr = sr
        self._beta = beta
        self._alpha = alpha
        self.register_buffer('cutoff_radii', _cutoff_radii)

    @classmethod
    def from_functional(
        cls,
        functional: str = "wB97X",
        order: int = 6,
        modified_damp: bool = False,
        **kwargs
    ) -> "ZeroDamp":
        d = constants.get_functional_constants()[functional or 'wB97X']
        if modified_damp:
            raise ValueError("Modified damp is not yet implemented")

        if order == 6:
            sr = d["sr6"]
            alpha = 14

        if order == 8:
            sr = d["sr8"]
            alpha = 16

        return cls(sr=sr, alpha=alpha, beta=0.0, cutoff_radii=None, **kwargs)

    def forward(
        self,
        species12: Tensor,
        distances: Tensor,
    ) -> Tensor:
        cutoff_radii = self.cutoff_radii[species12[0], species12[1]]
        inner_term = distances / (self._srr * cutoff_radii) + cutoff_radii * self._beta
        return distances.pow(self._order) * (1 + (6 * inner_term).pow(-self._alpha))


class TwoBodyDispersionD3(PairwisePotential):
    r"""Calculates the DFT-D3 dispersion corrections

    Only calculates the 2-body part of the dispersion corrections. Requires a
    damping function for the order-6 and order-8 potential terms.
    """

    covalent_radii: Tensor
    precalc_coordnums_a: Tensor
    precalc_coordnums_b: Tensor
    precalc_coeff6: Tensor
    sqrt_charge_ab: Tensor

    _s6: Final[float]
    _s8: Final[float]
    _k1: Final[int]
    _k2: Final[float]
    _k3: Final[int]

    def __init__(
        self,
        *args,
        damp_fn_6: Damp,
        damp_fn_8: Damp,
        s6: float,
        s8: float,
        cutoff_fn: Union[str, Cutoff] = "dummy",
        cutoff=math.inf,
        **kwargs,
    ):
        super().__init__(*args, cutoff=cutoff, cutoff_fn=cutoff_fn, **kwargs)

        self._damp_fn_6 = damp_fn_6
        self._damp_fn_8 = damp_fn_8

        order6_constants, coordnums_a, coordnums_b = constants.get_c6_constants()
        self.register_buffer('precalc_coeff6', order6_constants[self.atomic_numbers, :][:, self.atomic_numbers])
        self.register_buffer('precalc_coordnums_a', coordnums_a[self.atomic_numbers, :][:, self.atomic_numbers])
        self.register_buffer('precalc_coordnums_b', coordnums_b[self.atomic_numbers, :][:, self.atomic_numbers])

        # Covalent radii are in angstrom so we first convert to bohr
        covalent_radii = ANGSTROM_TO_BOHR * constants.get_covalent_radii()
        self.register_buffer('covalent_radii', covalent_radii[self.atomic_numbers])

        # The product of the sqrt of the empirical q's is stored directly
        sqrt_empirical_charge = constants.get_sqrt_empirical_charge()
        charge_ab = torch.outer(sqrt_empirical_charge, sqrt_empirical_charge)
        self.register_buffer('sqrt_charge_ab', charge_ab[self.atomic_numbers, :][:, self.atomic_numbers])

        self.ANGSTROM_TO_BOHR = ANGSTROM_TO_BOHR

        self._s6 = s6
        self._s8 = s8

        # Hardcoded values from Grimme
        self._k1 = 16
        self._k2 = 4 / 3
        self._k3 = 4

        # Solves numerical issues
        self._eps = 1e-35

    @classmethod
    def from_functional(
        cls,
        symbols: Sequence[str] = ("H", "C", "N", "O"),
        functional: str = "wB97X",
        modified_damp: bool = False,
        damp_fn: str = "bj",
        **kwargs,
    ) -> "TwoBodyDispersionD3":
        if damp_fn not in {"bj", "zero"}:
            raise ValueError("Damp function should be one of 'bj' or 'zero'")

        d = constants.get_functional_constants()[functional]

        DampCls: Type[Damp]
        if damp_fn == "bj":
            DampCls = BJDamp
        else:
            DampCls = ZeroDamp

        return cls(
            s6=d[f"s6_{damp_fn}"],
            s8=d[f"s8_{damp_fn}"],
            damp_fn_6=DampCls.from_functional(functional, symbols=symbols, order=6),
            damp_fn_8=DampCls.from_functional(functional, symbols=symbols, order=8),
            **kwargs,
        )

    def pair_energies(
        self,
        element_idxs: Tensor,
        neighbors: NeighborData,
    ) -> Tensor:
        # Internally this module works in AU
        distances = self.ANGSTROM_TO_BOHR * neighbors.distances

        species12 = element_idxs.flatten()[neighbors.indices]
        num_atoms = element_idxs.shape[1]
        num_molecules = element_idxs.shape[0]

        # Coordnums shape is num_molecules * num_atoms
        coordnums = self._coordnums(
            num_molecules,
            num_atoms,
            species12,
            neighbors.indices,
            distances
        )

        # Order 6 and 8 coefs
        order6_coeffs = self._interpolate_coeff6(
            species12,
            coordnums,
            neighbors.indices
        )
        order8_coeffs = 3 * order6_coeffs * self.sqrt_charge_ab[species12[0], species12[1]]

        # Order 6 and 8 energies
        order6_energy = self.s6 * order6_coeffs / self._damp_fn_6(species12, distances)
        order8_energy = self.s8 * order8_coeffs / self._damp_fn_8(species12, distances)
        return -(order6_energy + order8_energy)

    # Use the coordination numbers and the internal precalc C6's and
    # CNa's/CNb's to get interpolated C6 coeffs, C8 coeffs are obtained from C6
    # coeffs directly Output shape is (A,)
    def _coordnums(
        self,
        num_molecules: int,
        num_atoms: int,
        species12: Tensor,
        atom_index12: Tensor,
        distances: Tensor
    ) -> Tensor:
        # For coordination numbers "covalent radii" are used, not "cutoff radii"
        covalent_radii_sum = (
            self.covalent_radii[species12[0]] + self.covalent_radii[species12[1]]
        )

        count_fn = 1 / (
            1 + torch.exp(
                -self._k1 * (self._k2 * covalent_radii_sum / distances - 1)
            )
        )

        # Add terms corresponding to all neighbors
        coordnums = distances.new_zeros((num_molecules * num_atoms))
        coordnums.index_add_(0, atom_index12[0], count_fn)
        coordnums.index_add_(0, atom_index12[1], count_fn)
        return coordnums

    def _interpolate_coeff6(
        self,
        species12: Tensor,
        coordnums: Tensor,
        atom_index12: Tensor
    ) -> Tensor:
        assert coordnums.ndim == 1, 'coordnums must be one dimensional'
        assert species12.ndim == 2, 'species12 must be 2 dimensional'

        # Find pre-computed values for every species pair, and flatten over all
        # references shape is (num_pairs, 5, 5) flat-> (num_pairs, 25)
        precalc_coeff6 = self.precalc_coeff6[species12[0], species12[1]].flatten(1, 2)
        precalc_cn_a = self.precalc_coordnums_a[species12[0], species12[1]].flatten(1, 2)
        precalc_cn_b = self.precalc_coordnums_b[species12[0], species12[1]].flatten(1, 2)

        gauss_dist = (
            (coordnums[atom_index12[0]].view(-1, 1) - precalc_cn_a) ** 2
            + (coordnums[atom_index12[1]].view(-1, 1) - precalc_cn_b) ** 2
        )
        # Extra factor of gauss_dist.mean() and + 20 needed for numerical stability
        gauss_dist = torch.exp(-self._k3 * gauss_dist)
        # only consider C6 coefficients strictly greater than zero,
        # don't include -1 and 0.0 terms in the sums,
        # all missing parameters (with -1.0 values) are guaranteed to be the
        # same for precalc_cn_a/b and precalc_order6
        gauss_dist = gauss_dist.masked_fill(precalc_coeff6 <= 0.0, 0.0)
        # sum over references for w factor and z factor
        # This is needed for numerical stability, it will give 1 if W or Z are not
        # >> 1e-35 but those situations are rare in practice, and it avoids all potential
        # issues with NaN and exploding numbers / vanishing quantities
        w_factor = gauss_dist.sum(-1) + self._eps
        z_factor = (precalc_coeff6 * gauss_dist).sum(-1) + self._eps
        return z_factor / w_factor


def StandaloneTwoBodyDispersionD3(
    functional: str = "wB97X",
    symbols: Sequence[str] = ('H', 'C', 'N', 'O'),
    cutoff_fn: Union[str, Cutoff] = 'dummy',
    damp_fn: str = "bj",
    cutoff: float = math.inf,
    **standalone_kwargs,
) -> StandaloneWrapper:

    module = TwoBodyDispersionD3.from_functional(
        functional=functional,
        cutoff=cutoff,
        cutoff_fn=cutoff_fn,
        damp_fn=damp_fn,
        symbols=symbols,
    )
    return StandaloneWrapper(module, **standalone_kwargs)
