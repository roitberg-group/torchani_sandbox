import torchani  # noqa
from torchani.datasets import ANIDataset
# Example of simple usage of the ANIDataset class, which supersedes the obsolete
# anidataloader

# ANIDataset accepts a path to an h5 file or a list of paths (optionally with names)
dataset = ANIDataset('/path/to/h5/file.h5')
dataset = ANIDataset(locations=('/path/to/h5file1.h5', '/path/to/h5file2.h5'), names=('only_HCNO', 'heavy'))

# There are also builtin datasets that live in moria, and they can be directly
# downloaded through torchani.

# Downloading the builtin datasets performs a checksum to make sure the files
# are correct. If the function is called again and the dataset is already on
# the path, only the checksum is performed, the data is not downloaded. The
# output is an ANIDataset class
# Uncomment the following code to download (watch out, it may take some time):

# ds_1x = torchani.datasets.ANI1x('./datasets/ani1x/', download=True)
# ds_comp6 = torchani.datasets.COMP6v1('./datasets/comp6v1/', download=True)
# ds_2x = torchani.datasets.ANI2x('./datasets/ani2x/', download=True)

# ############## Conformer groups:  ###########################
# To access groups of conformers we can just use the dataset as an ordered
# dictionary
group = dataset['C10H10']
# if you have many files you can access by using the store names you passed:
group = dataset['name1/C10H10']
# by default the store names are just numbers if not passed
group0 = dataset['0/C10H10']
group1 = dataset['1/C10H8']
print(group)

# items(), values() and keys() work as expected for groups of conformers
for k, v in dataset.items():
    print(k, v)

for k in dataset.keys():
    print(k)

for v in dataset.values():
    print(v)

# To get the number of groups of conformers we can use len(), or also
# dataset.num_conformer_groups
num_groups = len(dataset)
print(num_groups)

# ############## Conformers:  ###########################
# To access individual conformers or subsets of conformers we use *_conformer
# methods, get_conformers and iter_conformers
conformer = dataset.get_conformers('C10H10', 0)
print(conformer)
conformer = dataset.get_conformers('C10H10', 1)
print(conformer)

# A tensor can also be passed for indexing, to fetch multiple conformers
# from the same group, which is faster.
# Since we copy the data for simplicity, this allows all fancy indexing
# operations (directly indexing using h5py does not).
# a numpy array or an int / list of ints can also be used.
conformers = dataset.get_conformers('C10H10', [0, 1])
print(conformers)

# We can also access all the group, same as with [] if we don't pass an index
conformer = dataset.get_conformers('C10H10')
print(conformer)

# Finally, it is possible to specify which properties we want using 'include_properties'
conformer = dataset.get_conformers('C10H10', include_properties=('species', 'energies'))
print(conformer)

conformer = dataset.get_conformers('C10H10', [0, 3], include_properties=('species', 'energies'))
print(conformer)

# We can iterate over all conformers sequentially by calling iter_conformer,
# (this is faster than doing it manually since it caches each conformer group
# previous to starting the iteration)
for c in dataset.iter_conformers():
    print(c)

# To get the number of conformers we can use num_conformers
num_conformers = dataset.num_conformers
print(num_conformers)
