import torch
import torchani
from torchani.tuples import SpeciesCoordinates
from torchani.AL_protocol.isolator import Isolator
from torchani.datasets import ANIDataset

input_file = '/home/nick/test_mol_dist.xyz'
ds = ANIDataset(
    '/home/nick/First_DSs/ANI-1x-first-conformers.h5'
    )
ch4_species = ds['CH4']['species']
ch4_coord = ds['CH4']['coordinates']
ch4_species_coord = (
    ch4_species,
    ch4_coord
    )

def main():
    isolator = Isolator()
    isolator.run_the_thing(
        input_file,
        is_file=True
        )
"""
    isolator.run_the_thing(
        ch4_species_coord,
        is_file=False
        )
"""
if __name__ == "__main__":
    main()