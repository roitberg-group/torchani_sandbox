import torch
import torchani
from torchani.aev.neighbors import BaseNeighborlist, FullPairwise
import numpy as np
from numpy.random import choice
from torchani.utils import ATOMIC_NUMBERS

_symbols_to_numbers = np.vectorize(lambda x: ATOMIC_NUMBERS[x])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


#Split cell in half along x axis
#cell[0]/2
#if < cell[0]/2 + 5.2 angstroms compute, but only compute aev for atoms in the extra 5.2 angstroms 

def split_cell(coordinates: torch.Tensor, 
        cell: torch.Tensor,
        species: torch.Tensor):
    print(cell[0])
    print('half:', cell[0][0]/2)
    split_dat={'species':[], 'coordinates':[]}
    for s, atom in zip(species, coordinates):
        if atom[0] <= cell[0][0]/2 + 5.2:
            split_dat['species'].append(s.item())
            split_dat['coordinates'].append(atom)
    split_dat['species'] = torch.Tensor(split_dat['species'])
    split_species = split_dat['species'].type(torch.long)
    split_coordinates= torch.stack((split_dat['coordinates']))
    return split_species, split_coordinates

def flag_overlap(species,
        coordinates, 
        cell: torch.Tensor, 
        model, 
        device=device):
    coordinates = torch.tensor(coordinates, requires_grad=True, device=device).unsqueeze(0)
    species = species.unsqueeze(0).to(device)
    species_coordinates = (species,coordinates)
    species, aev = model.aev_computer(species_coordinates)
    print(aev)
model = torchani.models.ANI1x().to(device)

split_species, split_coordinates = split_cell(bounded_coordinates.squeeze(0), cell, species.squeeze(0))


flag_overlap(split_species, split_coordinates, cell, model, device)
