import torch
import math
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
        species_ = species.view(-1)
        aev = aev.view(-1, aev.shape[-1])

        output = aev.new_zeros(species_.shape)

        for i, m in enumerate(self.values()):
            midx = (species_ == i).nonzero().view(-1)
            if midx.shape[0] > 0:
                input_ = aev.index_select(0, midx)
                output.index_add_(0, midx, m(input_).view(-1))
        output = output.view_as(species)
        return output


class ANIModelLocalMessagePassing(torch.nn.ModuleDict):

    size: Final[int]
    internal_size: Final[int]
    cutoff: Final[float]
    decay_factor: Tensor
    decay_prefactor: Tensor

    @staticmethod
    def ensureOrderedDict(modules):
        if isinstance(modules, OrderedDict):
            return modules
        od = OrderedDict()
        for i, m in enumerate(modules):
            od[str(i)] = m
        return od

    def __init__(self, modules, cutoff_fn, second_pass_modules, transforms, cutoff=5.2, internal_size=96):
        super().__init__(self.ensureOrderedDict(modules))
        self.size = 1
        self.internal_size = internal_size
        self.register_parameter('decay_factor', torch.nn.Parameter(torch.tensor(1.0)))
        self.register_parameter('decay_prefactor', torch.nn.Parameter(torch.tensor(1.0)))

        self.second_pass = torch.nn.ModuleDict(self.ensureOrderedDict(second_pass_modules))
        self.cutoff_fn = cutoff_fn
        self.cutoff = cutoff
        self.transforms = transforms

    def forward(self, species_aev: Tuple[Tensor, Tensor],  # type: ignore
                atom_index12: Tensor, distances: Tensor) -> SpeciesEnergies:
        species, aev = species_aev
        assert species.shape == aev.shape[:-1]

        atomic_energies = self._atomic_energies((species, aev), atom_index12, distances)
        # shape of atomic energies is (C, A)
        return SpeciesEnergies(species, torch.sum(atomic_energies, dim=1))

    @torch.jit.export
    def _atomic_energies(self, species_aev: Tuple[Tensor, Tensor], atom_index12: Tensor, distances: Tensor) -> Tensor:
        # Obtain the atomic energies associated with a given tensor of AEV's
        species, aev = species_aev
        assert species.shape == aev.shape[:-1]
        species_ = species.view(-1)
        aev = aev.view(-1, aev.shape[-1])
        internal_rep = aev.new_zeros((species_.shape[0], self.internal_size))
        for i, m in enumerate(self.values()):
            midx = (species_ == i).nonzero().view(-1)
            if midx.shape[0] > 0:
                input_ = aev.index_select(0, midx)
                internal_rep.index_add_(0, midx, m(input_).view(-1, self.internal_size))
        # at this point output has a shape A', G
        # I want first to duplicate this and then
        # I'll add in each position a weighted sum of the aev's of all pairs of
        # each atom, instead of attention I use simple distance weighting
        neighbor_rep = internal_rep.new_zeros(size=(internal_rep.shape[0], self.internal_size // 2))
        neighbor_merged_rep = internal_rep.new_zeros(size=(internal_rep.shape[0], self.internal_size // 2))
        # this should probably be multiplied by a cutoff function
        decay = (self.decay_prefactor**2) * torch.exp(-(self.decay_factor**2) * distances)
        decay *= self.cutoff_fn(distances, self.cutoff)

        # there is a different transform for each atom type
        for i, m in enumerate(self.transforms):
            midx = (species_ == i).nonzero().view(-1)
            if midx.shape[0] > 0:
                input_ = internal_rep.index_select(0, midx)
                neighbor_rep.index_add_(0, midx, m(input_).view(-1, self.internal_size // 2))

        neighbor_rep = neighbor_rep[atom_index12] * decay.view(1, -1, 1)
        # output has shape (P, 96)
        neighbor_merged_rep.index_add_(0, atom_index12.view(-1), neighbor_rep.view(-1, self.internal_size // 2))
        combined_rep = torch.cat((internal_rep, neighbor_merged_rep), dim=-1)
        # now internal output has shape (A', 96), so now I do an extra NN pass

        final_output = combined_rep.new_zeros(species_.shape)
        for i, m in enumerate(self.second_pass.values()):
            midx = (species_ == i).nonzero().view(-1)
            if midx.shape[0] > 0:
                input_ = combined_rep.index_select(0, midx)
                final_output.index_add_(0, midx, m(input_).view(-1))
        return final_output.view_as(species)


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


class SSP(torch.nn.Module):
    # shifted softplus activation

    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.softplus(x) - math.log(2)


class FittedSoftplus(torch.nn.Module):
    """Softplus function parametrized to be equal to a CELU

    This allows keeping the good characteristics of CELU, while having an
    infinitely differentiable function.
    It is highly recommended to leave alpha and beta as their defaults,
    which match closely CELU with alpha = 0.1"""

    alpha: Final[float]
    beta: Final[float]

    def __init__(self, alpha=0.1, beta=20):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.softplus(x + self.alpha, beta=self.beta) - self.alpha


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

    def __init__(self, species):
        super().__init__()
        rev_idx = {s: k for k, s in enumerate(utils.PERIODIC_TABLE)}
        maxidx = max(rev_idx.values())
        self.register_buffer('conv_tensor', torch.full((maxidx + 2,), -1, dtype=torch.long))
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
