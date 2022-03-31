import torch
import torchani
from torchani.aev.neighbors import BaseNeighborlist, FullPairwise
import numpy as np
from numpy.random import choice
from torchani.utils import ATOMIC_NUMBERS

_symbols_to_numbers = np.vectorize(lambda x: ATOMIC_NUMBERS[x])


#Generating simulation box with random coordinates within
#Generating species that coorelate with random coordinates 
num_atoms = 100


random_coordinates = torch.randn(1, num_atoms, 3) * 10
bn = BaseNeighborlist(5.2)
bounded_coordinates, cell = bn._compute_bounding_cell(random_coordinates, 1e-2)


species_opt = ['C', 'H', 'N', 'O']
species = choice(species_opt, (1, bounded_coordinates.shape[1]))
species = torch.Tensor(_symbols_to_numbers(species))


#Generating the Neighbor List, the diff_vector and distance vector
FP = FullPairwise(5.2)
nl, sv, diff, dist = FP(species, bounded_coordinates)

