from typing import Tuple, Optional

import torch
from torch import Tensor
from .neighbors import BaseNeighborlist, CellList


class LoadPartitioner(BaseNeighborlist):
    """This class will run before the neighborlist, and its responsibility is to assign
       different parts of the system to different groups, which will then be split into
       GPU's, to distribute the calculation load"""

    def __init__(self, cutoff: float, constant_volume: bool = True, spatial_divisions: Tuple[int, int, int] = (2, 1, 1)):
        super().__init__(cutoff)
        # spatial_divisions holds the number of divisions the load partitioner will create in the
        # X, Y, and Z dimensions, it should be 1 if that dimension is not split
        self.constant_volume = constant_volume
        self.register_buffer('_spatial_divisions', torch.tensor(spatial_divisions, dtype=torch.long))
        assert (self._spatial_divisions == torch.tensor([2, 1, 1])).all(), "only partitioning the X dimension is currently supported"

    @classmethod
    def from_gpu_number(cls, cutoff, constant_volume: bool = True, gpu_num: int = 2):
        return cls(cutoff, constant_volume, (2, 1, 1))

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
            # displaced coordinates will be used for all calculation purposes
            coordinates_displaced = coordinates.detach()

        # coordinates must first be fractionalized, (0-1) and then sent into different domains
        # depending on the place in the cell they are located in
        fractional_coordinates = CellList._fractionalize_coordinates(coordinates_displaced, torch.inverse(cell))
        # load balancing is currently done homogeneously, with every dimension
        # split evenly, so floor can be used to assign to different groups
        group_assignment = torch.floor(fractional_coordinates * self._spatial_divisions.view(1, 1, -1)).long()
        print(group_assignment)
        # group partition list will have some format of distribution of atoms over different GPU's
        return group_assignment, cell, pbc
