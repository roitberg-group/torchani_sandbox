import itertools
import math
import typing as tp
import typing_extensions as tpx

import torch
from torch import Tensor

try:
    import matplotlib.pyplot as plt

    _MPL_AVAIL = True
except ImportError:
    _MPL_AVAIL = False

from torchani.constants import ATOMIC_NUMBER, PERIODIC_TABLE
from torchani.cutoffs import _parse_cutoff_fn, CutoffArg, CutoffDummy
from torchani.neighbors import Neighbors, adaptive_list, all_pairs
from torchani.utils import _validate_user_kwargs
from torchani.units import ANGSTROM_TO_BOHR, HARTREE_TO_EV, HARTREE_TO_KCALPERMOL


# TODO: The "_coordinates" input is only required due to a quirk of the
# implementation of the cuAEV
class Potential(torch.nn.Module):
    r"""Base class for all atomic potentials

    Potentials may be many-body (3-body, 4-body, ...) potentials or 2-body (pair)
    potentials. Subclasses must implement ``compute`` and may override ``__init__``.
    """

    ANGSTROM_TO_BOHR: float
    cutoff: float
    atomic_numbers: Tensor
    _enabled: bool

    def __init__(
        self,
        symbols: tp.Sequence[str],
        *,
        cutoff: float = math.inf,
    ):
        super().__init__()
        self.atomic_numbers = torch.tensor(
            [ATOMIC_NUMBER[e] for e in symbols], dtype=torch.long
        )
        self.cutoff = cutoff
        # First element is extra-pair and last element will always be -1
        conv_tensor = -torch.ones(118 + 2, dtype=torch.long)
        for i, znum in enumerate(self.atomic_numbers):
            conv_tensor[znum] = i
        self._conv_tensor = conv_tensor
        self.ANGSTROM_TO_BOHR = ANGSTROM_TO_BOHR
        self._enabled = True

    @torch.jit.export
    def set_enabled(self, val: bool = True) -> None:
        self._enabled = val

    @torch.jit.unused
    def set_enabled_(self, val: bool = True) -> tpx.Self:
        self._enabled = val
        return self

    @torch.jit.export
    def is_enabled(self, val: bool = True) -> bool:
        return self._enabled

    # Validate a seqof floats that must have the same len as the provided symbols
    # default must be a mapping or sequence that when indexed with
    # atomic numbers returns floats
    @torch.jit.unused
    def _validate_elem_seq(
        self, name: str, seq: tp.Sequence[float], default: tp.Sequence[float] = ()
    ) -> tp.Sequence[float]:
        if not seq and default:
            seq = [default[j] for j in self.atomic_numbers]
        if not all(isinstance(v, float) for v in seq):
            raise ValueError(f"Some values in {name} are not floats")
        num_elem = len(self.symbols)
        if not len(seq) == num_elem:
            raise ValueError(f"{name} and symbols should have the same len")
        return seq

    @property
    @torch.jit.unused
    def symbols(self) -> tp.Tuple[str, ...]:
        return tuple(PERIODIC_TABLE[z] for z in self.atomic_numbers)

    @torch.jit.unused
    def calc(
        self,
        species: Tensor,
        coordinates: Tensor,
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
        periodic_table_index: bool = True,
        atomic: bool = False,
    ) -> Tensor:
        r"""
        Outputs energy, as calculated by the potential

        Output shape depends on the value of ``atomic``, it is either
        ``(molecs, atoms)`` or ``(molecs,)``
        """
        if periodic_table_index:
            elem_idxs = self._conv_tensor.to(species.device)[species]
        else:
            elem_idxs = species
        # Check inputs
        assert elem_idxs.dim() == 2
        assert coordinates.shape == (elem_idxs.shape[0], elem_idxs.shape[1], 3)

        if self.cutoff > 0.0:
            if coordinates.shape[0] == 1:
                neighbors = adaptive_list(
                    elem_idxs, coordinates, self.cutoff, cell, pbc
                )
            else:
                neighbors = all_pairs(elem_idxs, coordinates, self.cutoff, cell, pbc)
        else:
            neighbors = Neighbors(torch.empty(0), torch.empty(0), torch.empty(0))
        return self(elem_idxs, neighbors, atomic=atomic)

    def forward(
        self,
        elem_idxs: Tensor,
        neighbors: Neighbors,
        _coordinates: tp.Optional[Tensor] = None,
        ghost_flags: tp.Optional[Tensor] = None,
        atomic: bool = False,
    ) -> Tensor:
        if not self._enabled:
            if atomic:
                return neighbors.distances.new_zeros(elem_idxs.shape)
            return neighbors.distances.new_zeros(elem_idxs.shape[0])
        return self.compute(elem_idxs, neighbors, _coordinates, ghost_flags, atomic)

    def compute(
        self,
        elem_idxs: Tensor,
        neighbors: Neighbors,
        _coordinates: tp.Optional[Tensor] = None,
        ghost_flags: tp.Optional[Tensor] = None,
        atomic: bool = False,
    ) -> Tensor:
        r"""Compute the energies associated with the potential

        Must be implemented by subclasses
        """
        raise NotImplementedError("Must be implemented by subclasses")


class BasePairPotential(Potential):
    r"""General base class for all pairwise potentials

    Subclasses must implement pair_energies, and override init using this template:

    .. code-block:: python

        def __init__(
            symbols,
            ..., # User args go here
            cutoff: float=math.inf,
            cutoff_fn="smooth",
            ..., # User kwargs go here
        )
            super().__init__(symbols, cutoff, cutoff_fn)
            ... # User code goes here
    """

    def __init__(
        self,
        symbols: tp.Sequence[str],
        *,
        cutoff: float = math.inf,
        cutoff_fn: CutoffArg = "smooth",
    ):
        super().__init__(symbols, cutoff=cutoff)
        if cutoff != math.inf:
            self.cutoff_fn = _parse_cutoff_fn(cutoff_fn)
        else:
            self.cutoff_fn = CutoffDummy()

    @staticmethod
    def clamp(distances: Tensor) -> Tensor:
        return distances.clamp(min=1e-7)

    def pair_energies(
        self,
        elem_idxs: Tensor,
        neighbors: Neighbors,
    ) -> Tensor:
        r"""Return energy of all pairs of neighbors

        Returns a Tensor of energies, of shape ('pairs',) where 'pairs' is
        the number of neighbor pairs.
        """
        raise NotImplementedError("Must be overriden by subclasses")

    # Modulate pair_energies by wrapping with the cutoff fn envelope,
    # and scale ghost pair energies by 0.5
    def _pair_energies_wrapper(
        self,
        elem_idxs: Tensor,
        neighbors: Neighbors,
        ghost_flags: tp.Optional[Tensor] = None,
    ) -> Tensor:
        # Input validation
        assert elem_idxs.ndim == 2, "species should be 2 dimensional"
        assert neighbors.distances.ndim == 1, "distances should be 1 dimensional"
        assert neighbors.indices.ndim == 2, "atom_index12 should be 2 dimensional"
        assert neighbors.distances.shape[0] == neighbors.indices.shape[1]

        pair_energies = self.pair_energies(elem_idxs, neighbors)
        pair_energies *= self.cutoff_fn(neighbors.distances, self.cutoff)

        if ghost_flags is not None:
            if not ghost_flags.numel() == elem_idxs.numel():
                raise ValueError(
                    "ghost_flags and species should have the same number of elements"
                )
            ghost12 = ghost_flags.flatten()[neighbors.indices]
            ghost_mask = torch.logical_or(ghost12[0], ghost12[1])
            pair_energies = torch.where(ghost_mask, pair_energies * 0.5, pair_energies)
        return pair_energies

    # Compute should not be modified by subclasses of BasePairPotential
    def compute(
        self,
        elem_idxs: Tensor,
        neighbors: Neighbors,
        _coordinates: tp.Optional[Tensor] = None,
        ghost_flags: tp.Optional[Tensor] = None,
        atomic: bool = False,
    ) -> Tensor:
        pair_energies = self._pair_energies_wrapper(elem_idxs, neighbors, ghost_flags)
        molecs_num, atoms_num = elem_idxs.shape
        if atomic:
            energies = neighbors.distances.new_zeros(molecs_num * atoms_num)
            energies.index_add_(0, neighbors.indices[0], pair_energies / 2)
            energies.index_add_(0, neighbors.indices[1], pair_energies / 2)
            energies = energies.view(molecs_num, atoms_num)
        else:
            energies = neighbors.distances.new_zeros(molecs_num)
            molecs_idxs = torch.div(
                neighbors.indices[0], elem_idxs.shape[1], rounding_mode="floor"
            )
            energies.index_add_(0, molecs_idxs, pair_energies)
        return energies

    @torch.jit.unused
    def plot(
        self,
        symbol_pairs: tp.Sequence[tp.Tuple[str, str]] = (),
        xmin: float = 0.1,
        xmax: tp.Optional[float] = None,
        ymin: tp.Optional[float] = None,
        ymax: tp.Optional[float] = None,
        steps: int = 1000,
        force: bool = False,
        units: str = "hartree",
        ylog: bool = False,
        block: bool = True,
    ) -> None:
        factors = {
            "ev": HARTREE_TO_EV,
            "kcalpermol": HARTREE_TO_KCALPERMOL,
            "hartree": 1,
        }
        factor = factors.get(units.lower(), None)
        if factor is None:
            raise ValueError(
                f"Unsupported unit {units}. Supported are {set(factors.keys())}"
            )
        if not _MPL_AVAIL:
            raise RuntimeError("Please install matplotlib to plot this potential")
        if not symbol_pairs:
            symbol_pairs = tuple(itertools.combinations(self.symbols, 2))
        fig, ax = plt.subplots()
        if xmax is None:
            xmax = self.cutoff if self.cutoff != math.inf else 10.0
        r = torch.linspace(xmin, xmax, steps)
        for pair in symbol_pairs:
            atomic_nums = torch.zeros((steps, 2), dtype=torch.long)
            atomic_nums[:, 0] = ATOMIC_NUMBER[pair[0]]
            atomic_nums[:, 1] = ATOMIC_NUMBER[pair[1]]
            coords = torch.zeros((steps, 2, 3))
            if force:
                r = r.detach().requires_grad_(True)
            coords[:, 0, 0] = r
            energies = self.calc(atomic_nums, coords)
            energies *= factor
            if force:
                forces = -torch.autograd.grad(energies.sum(), r)[0]
                r.detach_()
                energies.detach_()
                ax.plot(r, forces, label=f"{pair[0]}-{pair[1]}")
            else:
                ax.plot(r, energies, label=f"{pair[0]}-{pair[1]}")
        ax.legend()
        ax.set_title(f"{self.__class__.__name__}")
        ax.set_xlabel(r"Inter atomic distance, ($\AA$)")
        unit_symbol = {
            "hartree": r"$E_h$",
            "ev": r"$\mathrm{eV}$",
            "kcalpermol": r"$\frac{\text{kcal}}{\text{mol}}$",
        }[units.lower()]
        ax.set_ylabel(f"Energy, ({unit_symbol})")
        if force:
            ax.set_ylabel(f"Force, ({unit_symbol}/" r"$\AA$)")
        if ylog:
            ax.set_yscale("log")
        ax.set_ylim(ymin, ymax)
        plt.show(block=block)


class PairPotential(BasePairPotential):
    r"""User friendly, simple class for pairwise potentials

    Subclasses must implement ``pair_energies`` and, if they use
    any paramters or buffers, specify two list of strings:

    - 'tensors'
    - 'elem_tensors'

    An simple example would be:

    .. code-block:: python

        class Square(PairPotential)
            tensors = ['bias']  # shape should be (S,) or (1,)
            elem_tensors = ['k']  # shape should be (num-symbols,)

            def pair_energies(self, neighbor_idxs, neighbors):
                return  self.bias + self.k / 2 * neighbors.distances ** 2

        pot = Square(symbols=("H", "C"), k=(1., 2.), bias=0.1)

        # Or if the constants are trainable:
        pot = Square(symbols=("H", "C"), k=(1., 2.), bias=0.1, trainable="k")
    """

    tensors: tp.List[str] = []
    elem_tensors: tp.List[str] = []

    def __init__(
        self,
        symbols: tp.Sequence[str],
        *,
        trainable: tp.Union[str, tp.Sequence[str]] = (),
        cutoff: float = math.inf,
        cutoff_fn: CutoffArg = "smooth",
        **kwargs,
    ) -> None:
        super().__init__(symbols, cutoff=cutoff, cutoff_fn=cutoff_fn)
        if isinstance(trainable, str):
            trainable = [trainable]

        _validate_user_kwargs(
            self.__class__.__name__,
            {
                "tensors": self.tensors,
                "elem_tensors": self.elem_tensors,
            },
            kwargs,
            trainable,
        )
        for k, v in kwargs.items():
            tensor = torch.tensor(v)
            if k in trainable:
                self.register_parameter(k, torch.nn.Parameter(tensor))
            else:
                self.register_buffer(k, tensor)

        if self.elem_tensors and len(getattr(self, self.elem_tensors[0])) != len(
            symbols
        ):
            raise ValueError(
                "All tensors registered as elem_tensors"
                " must have the same length as the passed chemical symbols"
            )
