import torch
import torchani
from torchani.tuples import SpeciesCoordinates
from torchani.AL_protocol.isolator import Isolator
from torchani.datasets import ANIDataset

import timeit

model = torchani.models.ANIdr().double()

input_file = '/home/nick/test_mol_dist.xyz'
ds = ANIDataset(
    '/home/nick/First_DSs/ANI-1x-first-conformers.h5'
    )
ch4_species = ds['CH4']['species']
ch4_coord = ds['CH4']['coordinates']
ch4_species_coord = (ch4_species, ch4_coord)

def run_file(model):
    isolator = Isolator(model=model)
    isolator.execute(input_file, is_file=True)

def run_data(model):
    isolator = Isolator(model=model)
    isolator.execute(ch4_species_coord, is_file=False)

def main():
    global model
    file_time = timeit.timeit(lambda: run_file(model), number=1)
    print(f"Time taken for processing file: {file_time} seconds")

    data_time = timeit.timeit(lambda: run_data(model), number=1)
    print(f"Time taken for processing data: {data_time} seconds")

if __name__ == "__main__":
    main()