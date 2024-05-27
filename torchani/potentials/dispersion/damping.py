import typing as tp

import torch
from torch import Tensor
from torch.jit import Final
import typing_extensions as tpx

from torchani.utils import ATOMIC_NUMBERS
from torchani.units import ANGSTROM_TO_BOHR
from torchani.potentials.dispersion import constants
from torchani.annotations import Device, FloatDType


# D3M modifies parameters AND damp function for zero-damp and only
# parameters for BJ damp cutoff radii are used for damp functions
class Damp(torch.nn.Module):
    r"""Damp function interface

    Damp functions modulate potentials close to zero.

    For modulating potentials of different "order" (e.g. 1 / r ** 6 => order
    6), different parameters may be needed.
    """

    order: Final[int]
    atomic_numbers: Tensor

    def __init__(
        self,
        symbols: tp.Sequence[str],
        order: int,
        *args,
        device: Device = "cpu",
        **kwargs,
    ):
        super().__init__()
        self.order = order
        self.atomic_numbers = torch.tensor(
            [ATOMIC_NUMBERS[e] for e in symbols],
            dtype=torch.long,
            device=device,
        )

    @classmethod
    def from_functional(
        cls,
        symbols: tp.Sequence[str],
        order: int,
        functional: str,
        device: Device = "cpu",
    ) -> tpx.Self:
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
        symbols: tp.Sequence[str],
        order: int,
        a1: float,
        a2: float,
        device: Device = "cpu",
        dtype: FloatDType = torch.float,
    ):
        super().__init__(symbols=symbols, order=order, device=device)
        sqrt_q = constants.get_sqrt_empirical_charge().do(device=device, dtype=dtype)

        znumbers = self.atomic_numbers
        outer_sqrt_q = torch.outer(sqrt_q, sqrt_q)
        cutoff_radii = torch.sqrt(3 * outer_sqrt_q)[:, znumbers][znumbers, :]

        # Cutoff radii is a matrix of S x S where S are the supported elements.
        assert cutoff_radii.shape == (len(znumbers), len(znumbers))

        self.register_buffer("cutoff_radii", cutoff_radii)
        self._a1 = a1
        self._a2 = a2

    @classmethod
    def from_functional(
        cls,
        symbols: tp.Sequence[str],
        order: int,
        functional: str,
        device: Device = "cpu",
    ) -> tpx.Self:
        d = constants.get_functional_constants()[functional.lower()]
        return cls(
            symbols=symbols,
            order=order,
            a1=d["a1"],
            a2=d["a2"],
            device=device,
        )

    def forward(
        self,
        species12: Tensor,
        distances: Tensor,
    ) -> Tensor:
        cutoff_radii = self.cutoff_radii[species12[0], species12[1]]
        damp_term = (self._a1 * cutoff_radii + self._a2).pow(self.order)
        return distances.pow(self.order) + damp_term


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
        symbols: tp.Sequence[str],
        order: int,
        alpha: int,
        sr: float,
        beta: float = 0.0,
        device: Device = "cpu",
        dtype: FloatDType = torch.float,
    ):
        super().__init__(symbols=symbols, order=order, device=device)

        znumbers = self.atomic_numbers
        # These cutoff radii are in Angstrom, so we convert to Bohr.
        cutoff_radii = constants.get_cutoff_radii().to(dtype=dtype, device=device)
        cutoff_radii = ANGSTROM_TO_BOHR * cutoff_radii[:, znumbers][znumbers, :]

        # Cutoff radii is a matrix of T x T where T are the supported elements.
        assert cutoff_radii.shape == (len(znumbers), len(znumbers))

        self._sr = sr
        self._beta = beta
        self._alpha = alpha
        self.register_buffer("cutoff_radii", cutoff_radii)

    @classmethod
    def from_functional(
        cls,
        symbols: tp.Sequence[str],
        order: int,
        functional: str,
        device: Device = "cpu",
    ) -> tpx.Self:
        d = constants.get_functional_constants()[functional.lower()]
        if order == 6:
            sr = d["sr6"]
            alpha = 14

        if order == 8:
            sr = d["sr8"]
            alpha = 16

        return cls(
            symbols=symbols,
            order=order,
            sr=sr,
            alpha=alpha,
            beta=0.0,
            device=device,
        )

    def forward(
        self,
        species12: Tensor,
        distances: Tensor,
    ) -> Tensor:
        cutoff_radii = self.cutoff_radii[species12[0], species12[1]]
        inner_term = distances / (self._srr * cutoff_radii) + cutoff_radii * self._beta
        return distances.pow(self.order) * (1 + (6 * inner_term).pow(-self._alpha))


def _parse_damp_fn_cls(kind: str) -> tp.Type[Damp]:
    if kind == "bj":
        return BJDamp
    elif kind == "zero":
        return ZeroDamp
    raise ValueError("Incorrect damp function class")
