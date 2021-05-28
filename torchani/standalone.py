from .aev import FullPairwise, BaseNeighborlist
from .utils import map_to_central
import torch
from torch import Tensor
from typing import Tuple, Optional
from .compat import Final
from .nn import SpeciesConverter


class StandalonePairwiseWrapper(torch.nn.Module):
    # this helper class wraps modules so that they can function directly with
    # an input of species_coordinates, cell, pbc. This is useful for testing
    # purposes and for some special cases, it is specially useful for the
    # "repulsion" and "dispersion" computers
    # IMPORTANT NOTE: This should be inherited from FIRST (leftmost in inheritance list)
    # for the scheme to work properly
    periodic_table_index: Final[bool]

    def __init__(self, *args, **kwargs):
        supported_species = kwargs.get('elements', ('H', 'C', 'N', 'O'))
        self.periodic_table_index = kwargs.pop('periodic_table_index', False)
        neighborlist = kwargs.pop('neighborlist', FullPairwise)
        cutoff = kwargs.pop('neighborlist_cutoff', 5.2)
        super().__init__(*args, **kwargs)
        self.species_converter = SpeciesConverter(supported_species)
        # neighborlist uses radial cutoff only
        self.neighborlist = neighborlist(cutoff) if neighborlist is not None else BaseNeighborlist(cutoff)
        self.register_buffer('default_cell', torch.eye(3, dtype=torch.float))
        self.register_buffer('default_pbc', torch.zeros(3, dtype=torch.bool))

    def _validate_inputs(self, species_coordinates: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        species, coordinates = species_coordinates
        # check shapes for correctness
        assert species.dim() == 2
        assert coordinates.dim() == 3
        assert (species.shape == coordinates.shape[:2]) and (coordinates.shape[2] == 3)
        return species_coordinates

    def _perform_module_actions(self, species_coordinates: Tuple[Tensor, Tensor], atom_index12: Tensor,
            distances: Tensor) -> Tuple[Tensor, Tensor]:
        assert False, "This method should be overriden by subclasses"
        return species_coordinates

    def forward(self, species_coordinates: Tuple[Tensor, Tensor], cell: Optional[Tensor] = None,
                pbc: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:

        species_coordinates = self._validate_inputs(species_coordinates)

        if self.periodic_table_index:
            species_coordinates = self.species_converter(species_coordinates)
        species, coordinates = species_coordinates

        # the coordinates that are input into the neighborlist are **not** assumed to be
        # mapped into the central cell for pbc calculations,
        # and **in general are not**
        atom_index12, _, _, distances = self.neighborlist(species, coordinates, cell, pbc)

        # the coordinates that are input into the next module, on the other hand,
        # are always assumed to be mapped to the central cell
        if pbc is not None:
            if pbc.any():
                assert cell is not None
                coordinates = map_to_central(coordinates, cell, pbc)

        return self._perform_module_actions(species_coordinates, atom_index12, distances)
