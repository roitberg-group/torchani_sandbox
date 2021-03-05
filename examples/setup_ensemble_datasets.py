# -*- coding: utf-8 -*-
"""
.. _setup-ensemble-dataset:

Break up a dataset into folds for ensemble training
=======================================

"""

###############################################################################
# To begin with, let's first import the modules and setup devices we will use:

import torch
import torchani
from pathlib import Path

# helper function to convert energy unit from Hartree to kcal/mol

# device to run the training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    dspath = Path(__file__).resolve().parent
except NameError:
    dspath = Path.cwd().resolve()

# this dataset path contains the full dataset, which we want to break apart into
# folds in order to train an ensemble
dspath = dspath.joinpath('../dataset/ani1-up_to_gdb4/ani_gdb_s01.h5').resolve()
energy_shifter = torchani.EnergyShifter(None)
batch_size = 2560

# this is the dataset path we we want to load, in this case we want to load dataset_0
pickled_dataset_path = Path('./datasets/dataset_0.pkl').resolve()

# We pickle the dataset after loading to ensure we use the same validation set
# each time we restart training, otherwise we risk mixing the validation and
# training sets on each restart.
if not pickled_dataset_path.is_file():
    print(f'Processing dataset in {dspath}')
    training_sets, validation_sets = torchani.data.load(dspath)\
                                        .subtract_self_energies(energy_shifter)\
                                        .species_to_indices("periodic_table")\
                                        .shuffle()\
                                        .split_for_cross_validation()
    torchani.data.pickle_cross_validation_datasets(training_sets, validation_sets, energy_shifter.self_energies.cpu(), batch_size=batch_size)
    del training_sets
    del validation_sets

# here we load the training and validation sets for dataset_0
training, validation, self_energies = torchani.data.load_pickled_dataset(pickled_dataset_path)
energy_shifter.self_energies = self_energies.to(device)
print('Self atomic energies: ', energy_shifter.self_energies)
