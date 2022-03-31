import torch
import torchani
from torchani.aev.neighbors import BaseNeighborlist
import numpy as np
from numpy.random import rand

sim_box = rand(3,3)

num_atoms = 100
random_coordinates = torch.randn(1, num_atoms, 3) * 10
bn = BaseNeighborlist(5.2)
bounded_coordinates, cell = bn._compute_bounding_cell(random_coordinates, 1e-2)
print(cell)
