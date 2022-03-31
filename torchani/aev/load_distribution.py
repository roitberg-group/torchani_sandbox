from typing import Tuple, Optional

import torch
from torch import Tensor
from .neighbors import BaseNeighborlist


class LoadPartitioner(BaseNeighborlist):
    """This class will run before the neighborlist, and its responsibility is to assign
       different parts of the system to different groups, which will then be split into
       GPU's, to distribute the calculation load"""

    def __init__(self, cutoff: float, constant_volume: bool = True):
        super().__init__(cutoff)
        self.constant_volume = constant_volume

    def forward(self, coordinates: Tensor,
                      cell: Optional[Tensor] = None,
                      pbc: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:

        # This class is only for inference
        assert coordinates.shape[0] == 1, "Load partitioning doesn't support batches"

        # PBC can be an input but it will just be disregarded
        if pbc is None:
            pbc = self.default_pbc
        assert not pbc.any(), "PBC is not yet supported with load partitioning"

        # If not provided we calculate a bounding box
        if cell is None:
            # Displaced coordinates are only used for computation if PBC is not required
            coordinates_displaced, cell = self._compute_bounding_cell(coordinates.detach(), eps=1e-3)
        else:
            coordinates_displaced = coordinates.detach()
        print(coordinates_displaced)

        group_partition_list = torch.tensor([])
        return group_partition_list, cell, pbc
