from pathlib import Path

import torch
import torchani
import pkbar
from torchani.data.dataset import ANIBatchedDataset, save_batched_dataset

# Explanation of the Batched Dataset API for ANI, which is a dataset that
# consumes minimal memory since it lives on disk, and batches are fetched on
# the fly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dspath = Path('../dataset/ani1-up_to_gdb4/ani_gdb_s01.h5').resolve()
batch_size = 2560

# We are going to load a "batched dataset" which is a directory
# with serialized files containing pre-padded batches
path_to_batched = Path('./batched_dataset_npz').resolve()

# If this directory doesn't already exist we create a "batched dataset"
# by loading it and batching it, and then calling "save_batched_datset".
# It is important to specify the split (training or validation)
if not path_to_batched.exists():
    file_format = 'numpy'
    training, validation = torchani.data.load(dspath.as_posix())\
                                        .species_to_indices()\
                                        .shuffle()\
                                        .split(0.8, None)
    training = training.collate(batch_size).cache()
    validation = validation.collate(batch_size).cache()
    save_batched_dataset(training, path_to_batched, file_format='numpy', split='training')
    save_batched_dataset(validation, path_to_batched, file_format='numpy', split='validation')

# Once we have created the batched dataset we instance it using the class
# ANIBatchedDataset, which subclasses torch.utils.data.Dataset
training = ANIBatchedDataset(path_to_batched, split='training')
validation = ANIBatchedDataset(path_to_batched, split='validation')

# This batched dataset can be directly iterated upon, but it is more practical
# to wrap it with a torch dataloader to obtain automatic shuffling evey epoch,
# multiprocessing and memory pinning
# Note: it is very important here to pass batch_size = None since the dataset is
# already batched!
training_loader = torch.utils.data.DataLoader(training, num_workers=2, prefetch_factor=2, pin_memory=True, shuffle=True, batch_size=None)
validation_loader = torch.utils.data.DataLoader(validation, num_workers=2, prefetch_factor=2, pin_memory=True, shuffle=True, batch_size=None)

progbar = pkbar.Kbar(target=len(training_loader) - 1, width=8)
for i, batch in enumerate(training_loader):
    species = batch['species'].long().to(device, non_blocking=True)
    coordinates = batch['coordinates'].float().to(device, non_blocking=True)
    energies = batch['energies'].float().to(device, non_blocking=True)
    progbar.update(i)
    torch.cuda.synchronize()  # only needed for timing measurement

# The batched dataset lives in disk, not in memory, so iterating is a bit
# slower than holding all the dataset in memory since reads from disk are
# significantly slower than reads from memory, however, the difference may not
# be that much for your case, it is recommended to try not caching first.

# If you want some extra speedup you can cache the dataset before passing it to
# the dataloader, so that it will live in memory, but this may occupy a lot of
# memory, so be careful!!!, this would be done with:
training = training.cache()
validation = validation.cache()

progbar = pkbar.Kbar(target=len(training) - 1, width=8)
for i, batch in enumerate(training):
    species = batch['species'].long().to(device, non_blocking=True)
    coordinates = batch['coordinates'].float().to(device, non_blocking=True)
    energies = batch['energies'].float().to(device, non_blocking=True)
    progbar.update(i)
    torch.cuda.synchronize()  # only needed for timing measurement
