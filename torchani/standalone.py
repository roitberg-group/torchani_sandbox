from typing import Tuple, Optional

import torch
from torch import Tensor

from .compat import Final
from .nn import SpeciesConverter
from .structs import SpeciesEnergies
from .aev import FullPairwise, BaseNeighborlist
from .utils import map_to_central


# This helper class wraps modules so that they can function directly with
# an input of species_coordinates, cell, pbc. This is useful for testing
# purposes and for some special cases, it is especially useful for the
# "repulsion" and "dispersion" computers
# IMPORTANT: This should be inherited from FIRST (leftmost in inheritance list)
# for the scheme to work properly
class StandaloneWrapper(torch.nn.Module):
    periodic_table_index: Final[bool]

    def __init__(self, *args, **kwargs):
        symbols = kwargs.get('symbols', ('H', 'C', 'N', 'O'))
        self.periodic_table_index = kwargs.pop('periodic_table_index', True)
        neighborlist = kwargs.pop('neighborlist', FullPairwise)
        cutoff = kwargs.pop('neighborlist_cutoff', 5.2)
        super().__init__(*args, **kwargs)
        # neighborlist uses radial cutoff only
        if neighborlist is not None:
            self.neighborlist = neighborlist(cutoff)
        else:
            self.neighborlist = BaseNeighborlist(cutoff)
        self.znumbers_to_idxs = SpeciesConverter(symbols)

    def _validate_inputs(
        self,
        species_coordinates: Tuple[Tensor, Tensor],
        cell: Optional[Tensor] = None,
        pbc: Optional[Tensor] = None
    ) -> None:
        species, coordinates = species_coordinates
        # check shapes for correctness
        assert species.dim() == 2
        assert coordinates.dim() == 3
        assert (species.shape == coordinates.shape[:2]) and (coordinates.shape[2] == 3)
        if pbc is not None and pbc.any():
            assert cell is not None

    def _calculate_energy(
        self,
        atomic_idxs: Tensor,
        coordinates: Tensor,
        neighbor_idxs: Tensor,
        distances: Tensor,
        diff_vectors: Optional[Tensor] = None,
    ) -> SpeciesEnergies:
        raise NotImplementedError("This method should be overriden by subclasses")
        return SpeciesEnergies(atomic_idxs, coordinates)

    def forward(
        self,
        species_coordinates: Tuple[Tensor, Tensor],
        cell: Optional[Tensor] = None,
        pbc: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:

        self._validate_inputs(species_coordinates, cell, pbc)
        species, coordinates = species_coordinates

        if self.periodic_table_index:
            atomic_idxs = self.znumbers_to_idxs(species_coordinates).species
        else:
            atomic_idxs = species

        # the coordinates that are input into the neighborlist are **not** assumed to be
        # mapped into the central cell for pbc calculations,
        # and **in general are not**
        neighbor_data = self.neighborlist(atomic_idxs, coordinates, cell, pbc)

        # the coordinates that are input into the next module, on the other hand,
        # are always assumed to be mapped to the central cell
        if pbc is not None and cell is not None:
            coordinates = map_to_central(coordinates, cell, pbc)

        energy = self._calculate_energy(
            (atomic_idxs, coordinates),
            neighbor_idxs=neighbor_data.indices,
            distances=neighbor_data.distances,
            diff_vectors=neighbor_data.diff_vectors,
        )
        return SpeciesEnergies(species, torch.zeros(species.shape[0], dtype=energy.dtype, device=energy.device) + energy)
