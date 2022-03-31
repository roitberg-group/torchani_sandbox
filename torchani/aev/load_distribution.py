from typing import Tuple, Optional

import torch
from torch import Tensor
from .neighbors import BaseNeighborlist


class LoadPartitioner(BaseNeighborlist):
    """This class will run before the neighborlist, and its responsibility is to assign
       different parts of the system to different groups, which will then be split into
       GPU's, to distribute the calculation load"""

    def __init__(self, cutoff: float, constant_volume: bool = True, spatial_divisions: Tuple[int, int, int] = (2, 0, 0)):
        super().__init__(cutoff)
        # divisions holds the number of divisions the load partitioner will create in the
        # X, Y, and Z dimensions
        self.constant_volume = constant_volume
        self.register_buffer('_spatial_divisions', torch.tensor(spatial_divisions, dtype=torch.long))
        assert self._spatial_divisions == torch.tensor([2, 0, 0]), "only partitioning the X dimension is currently supported"

    @classmethod
    def from_gpu_number(cls, cutoff, constant_volume: bool = True, gpu_num: int = 2):
        return cls(cutoff, constant_volume, (2, 0, 0))

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
        # displaced coordinates will be used for all calculation purposes

        print(coordinates_displaced)

        group_partition_list = torch.tensor([])
        # group partition list will have some format of distribution of atoms over different GPU's
        return group_partition_list, cell, pbc
