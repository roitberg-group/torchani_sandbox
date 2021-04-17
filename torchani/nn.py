import torch
from collections import OrderedDict
from torch import Tensor
from typing import Tuple, NamedTuple, Optional
from . import utils
from .compat import Final


class SpeciesEnergies(NamedTuple):
    species: Tensor
    energies: Tensor


class SpeciesCoordinates(NamedTuple):
    species: Tensor
    coordinates: Tensor


class ANIModel(torch.nn.ModuleDict):
    """ANI model that compute energies from species and AEVs.

    Different atom types might have different modules, when computing
    energies, for each atom, the module for its corresponding atom type will
    be applied to its AEV, after that, outputs of modules will be reduced along
    different atoms to obtain molecular energies.

    .. warning::

        The species must be indexed in 0, 1, 2, 3, ..., not the element
        index in periodic table. Check :class:`torchani.SpeciesConverter`
        if you want periodic table indexing.

    .. note:: The resulting energies are in Hartree.

    Arguments:
        modules (:class:`collections.abc.Sequence`): Modules for each atom
            types. Atom types are distinguished by their order in
            :attr:`modules`, which means, for example ``modules[i]`` must be
            the module for atom type ``i``. Different atom types can share a
            module by putting the same reference in :attr:`modules`.
    """

    size: Final[int]

    @staticmethod
    def ensureOrderedDict(modules):
        if isinstance(modules, OrderedDict):
            return modules
        od = OrderedDict()
        for i, m in enumerate(modules):
            od[str(i)] = m
        return od

    def __init__(self, modules):
        super().__init__(self.ensureOrderedDict(modules))
        self.size = 1

    def forward(self, species_aev: Tuple[Tensor, Tensor],  # type: ignore
                cell: Optional[Tensor] = None,
                pbc: Optional[Tensor] = None) -> SpeciesEnergies:
        species, aev = species_aev
        assert species.shape == aev.shape[:-1]

        atomic_energies = self._atomic_energies((species, aev))
        # shape of atomic energies is (C, A)
        return SpeciesEnergies(species, torch.sum(atomic_energies, dim=1))

    @torch.jit.export
    def _atomic_energies(self, species_aev: Tuple[Tensor, Tensor]) -> Tensor:
        # Obtain the atomic energies associated with a given tensor of AEV's
        species, aev = species_aev
        assert species.shape == aev.shape[:-1]
        species_ = species.flatten()
        aev = aev.flatten(0, 1)

        output = aev.new_zeros(species_.shape)

        for i, m in enumerate(self.values()):
            mask = (species_ == i)
            midx = mask.nonzero().flatten()
            if midx.shape[0] > 0:
                input_ = aev.index_select(0, midx)
                output.masked_scatter_(mask, m(input_).flatten())
        output = output.view_as(species)
        return output


class Ensemble(torch.nn.ModuleList):
    """Compute the average output of an ensemble of modules."""

    size: Final[int]

    def __init__(self, modules):
        super().__init__(modules)
        self.size = len(modules)

    def forward(self, species_input: Tuple[Tensor, Tensor],  # type: ignore
                cell: Optional[Tensor] = None,
                pbc: Optional[Tensor] = None) -> SpeciesEnergies:
        sum_ = 0
        for x in self:
            sum_ += x(species_input)[1]
        species, _ = species_input
        return SpeciesEnergies(species, sum_ / self.size)  # type: ignore

    @torch.jit.export
    def _atomic_energies(self, species_aev: Tuple[Tensor, Tensor]) -> Tensor:
        members_list = []
        for nnp in self:
            members_list.append(nnp._atomic_energies((species_aev)).unsqueeze(0))
        member_atomic_energies = torch.cat(members_list, dim=0)
        return member_atomic_energies


class Sequential(torch.nn.ModuleList):
    """Modified Sequential module that accept Tuple type as input"""

    def __init__(self, *modules):
        super().__init__(modules)

    def forward(self, input_: Tuple[Tensor, Tensor],  # type: ignore
                cell: Optional[Tensor] = None,
                pbc: Optional[Tensor] = None):
        for module in self:
            input_ = module(input_, cell=cell, pbc=pbc)
        return input_


class Gaussian(torch.nn.Module):
    """Gaussian activation"""
    def forward(self, x: Tensor) -> Tensor:
        return torch.exp(- x * x)


class SpeciesConverter(torch.nn.Module):
    """Converts tensors with species labeled as atomic numbers into tensors
    labeled with internal torchani indices according to a custom ordering
    scheme. It takes a custom species ordering as initialization parameter. If
    the class is initialized with ['H', 'C', 'N', 'O'] for example, it will
    convert a tensor [1, 1, 6, 7, 1, 8] into a tensor [0, 0, 1, 2, 0, 3]

    Arguments:
        species (:class:`collections.abc.Sequence` of :class:`str`):
        sequence of all supported species, in order (it is recommended to order
        according to atomic number).
    """
    conv_tensor: Tensor
    supported_atomic_numbers: Tensor

    def __init__(self, species):
        super().__init__()
        rev_idx = {s: k for k, s in enumerate(utils.PERIODIC_TABLE)}
        maxidx = max(rev_idx.values())
        self.register_buffer('conv_tensor', torch.full((maxidx + 2,), -1, dtype=torch.long))
        self.register_buffer('supported_atomic_numbers', torch.tensor([utils.PERIODIC_TABLE.index(s) for s in species], dtype=torch.long))

        for i, s in enumerate(species):
            self.conv_tensor[rev_idx[s]] = i

    def forward(self, input_: Tuple[Tensor, Tensor],
                cell: Optional[Tensor] = None,
                pbc: Optional[Tensor] = None):
        """Convert species from periodic table element index to 0, 1, 2, 3, ... indexing"""
        species, coordinates = input_
        converted_species = self.conv_tensor[species]

        # check if unknown species are included
        if converted_species[species.ne(-1)].lt(0).any():
            raise ValueError(f'Unknown species found in {species}')

        return SpeciesCoordinates(converted_species.to(species.device), coordinates)


class EnergyAdder(torch.nn.Module):
    """Helper class for adding and subtracting self atomic energies

    Arguments:
        self_energies (:class:`collections.abc.Sequence`): optional sequence of
            floating numbers for the self energy of each atom type. The numbers
            should be in order, i.e. ``self_energies[i]`` should be atom type
            ``i``, if not passed then GSAE's are used.
        elements: supported elements by the EnergyAdder.
        intercept: float,  intercept to add to the SAE's, only supported if self_energies are passed directly
        level_of_theory: Level of theory for the GSAE's
    """
    self_energies: Tensor
    supported_atomic_numbers: Tensor
    intercept: Tensor

    def __init__(self, elements, intercept: float = 0.0, self_energies=None, level_of_theory: str = 'RwB97X'):
        super().__init__()
        assert isinstance(elements[0], str), f'elements must be a tuple of atomic symbols but got {elements}, {type(elements)} of type {type(elements[0])}'

        if self_energies is not None:
            self_energies = torch.tensor(self_energies, dtype=torch.float)
        else:
            # if self_energies are not passed, we use the QM atomic energies for these elements
            self_energies = torch.tensor([utils.GSAEs[level_of_theory][e] for e in elements], dtype=torch.float)
            assert intercept == 0.0, "No physical meaning to an intercept if you are using GSAEs"

        self.register_buffer('intercept', torch.tensor(intercept, dtype=torch.float))
        self.register_buffer('self_energies', self_energies)
        self.register_buffer('supported_atomic_numbers', torch.tensor([utils.PERIODIC_TABLE.index(s) for s in elements], dtype=torch.long))

    def forward(self, species_energies: Tuple[Tensor, Tensor], cell: Optional[Tensor] = None, pbc: Optional[Tensor] = None) -> SpeciesEnergies:
        """(species, molecular energies) -> (species, molecular energies + sae)"""
        species, energies = species_energies
        energies += self.sae(species)
        return SpeciesEnergies(species, energies)

    @torch.jit.export
    def sae(self, species: Tensor) -> Tensor:
        """Compute self energies for molecules.

        Dummy atoms are automatically excluded.

        Arguments:
            species (:class:`torch.Tensor`): Long tensor in shape
                ``(conformations, atoms)``.

        Returns:
            :class:`torch.Tensor`: 1D vector in shape ``(conformations,)``
            with molecular self energies.
        """
        return self._calc_atomic_saes(species).sum(dim=1) + self.intercept

    @torch.jit.export
    def _calc_atomic_saes(self, species: Tensor) -> Tensor:
        # Compute atomic self energies for a set of species.
        self_atomic_energies = self.self_energies[species]
        self_atomic_energies = self_atomic_energies.masked_fill(species == -1, 0.0)
        return self_atomic_energies
