import time
from pathlib import Path

import torch
import torchani
from torchani.data.dataset import ANIBatchedDataset, save_batched_dataset

# Explanation of the Batched Dataset API for ANI, which is a dataset that
# consumes minimal memory since it lives on disk, and batches are fetched on
# the fly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# dspath = Path('../dataset/ani1-up_to_gdb4/ani_gdb_s01.h5').resolve()
dspath = Path('/home/ignacio/Datasets/ani1x_release_wb97x_dz.h5')
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


path_to_batched = Path('./batched_dataset_npz').resolve()
# Once we have created the batched dataset we instance it using the class
# ANIBatchedDataset, which subclasses torch.utils.data.Dataset
training = ANIBatchedDataset(path_to_batched, split='training'),
validation = ANIBatchedDataset(path_to_batched, split='validation')

# This batched dataset can be directly iterated upon, but it is more practical
# to wrap it with a torch dataloader to obtain automatic shuffling evey epoch,
# multiprocessing and memory pinning
training = torch.utils.data.DataLoader(training, num_workers=0, pin_memory=True, shuffle=True)
validation = torch.utils.data.DataLoader(validation, num_workers=0, pin_memory=True, shuffle=True)

# The batched dataset lives in disk, not in memory, so iterating is a bit
# slower than holding all the dataset in memory since reads from disk are
# significantly slower than reads from memory, however, the difference may not
# be that much for your case, it is recommended to try not caching first.

# If you want some extra speedup you can cache the dataset before passing it to
# the dataloader, so that it will live in memory, but this may occupy a lot of
# memory, so be careful!!!, this would be done with:
# training = torch.utils.data.DataLoader(training.cache(), num_workers=0, pin_memory=True, shuffle=True)
# validation = torch.utils.data.DataLoader(validation.cache(), num_workers=0, pin_memory=True, shuffle=True)

start = time.time()
overhead = 0.0
for batch in training:
    start_overhead = time.time()
    torch.cuda.synchronize()
    species = batch['species'].long().to(device)
    coordinates = batch['coordinates'].float().to(device)
    energies = batch['energies'].float().to(device)
    torch.cuda.synchronize()
    end_overhead = time.time()
    overhead += end_overhead - start_overhead
end = time.time()
total = end - start

print('total:', total)
print('casting overhead:', overhead)
print('loading from disk', total - overhead)
