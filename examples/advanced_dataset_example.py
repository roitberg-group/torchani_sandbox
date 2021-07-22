from torchani.datasets import ANIDataset

# Example of some more advanced usage of the ANIDataset class
ds = ANIDataset(('/path/to/h5file.h5', '/path/to/h5file2.h5'))

# Grouping:
# You can query whether your dataset is in a legacy format by interrogating the
# dataset grouping attribute
if ds.grouping == 'legacy':
    print("Dataset uses a legacy format")

# Legacy format is the format used by Justin when storing his datasets.  In the
# legacy format there can be groups arbitrarily nested in the hierarchical tree
# inside the h5 files, and the "species" property does not have a batch
# dimension.  this means all properties with an "atomic" dimension must be
# ordered the same way within a group (don't worry too much if you don't
# understand what this means, it basically means this is difficult to deal
# with)

# We can convert to a less error prone and easier to parse format by calling
# "regroup_by_formula" or "regroup_by_num_atoms"

ds = ds.regroup_by_formula()
assert ds.grouping == 'by_formula'
ds = ds.regroup_by_num_atoms()
assert ds.grouping == 'by_num_atoms'

# In these formats all of the first dimensions of all properties are the same in
# all groups, and groups can only have depth one.
# in other words the tree structure is just:
#
# /C10H22/coordinates, shape (10, 32, 3)
#        /species, shape (10, 32)
#        /energies, shape (10,)
# /C8H22N2/coordinates, shape (10, 32, 3)
#        /species, shape (10, 32)
#        /energies, shape (10,)
# /C12H22/coordinates, shape (5, 34, 3)
#        /species, shape (5, 34)
#        /energies, shape (5,)
# ...
# for "by_formula" and
# /032/coordinates, shape (20, 32, 3)
#      /species, shape (20, 32)
#      /energies, shape (20,)
# /034/coordinates, shape (5, 34, 3)
#      /species, shape (5, 34)
#      /energies, shape (5,)
# ...
# for "by_num_atoms"

# ######### Manipulating properties #################
# (Watch out: dataset mutability modifies the underlying stores)
# All of the molecules in the dataset have the same properties
# (energies, coordinates, etc...)
# You can query which are these properties like this
print(ds.properties)
# it is possible to rename the properties by passing a dict of old-new names
# (The class assumes at least one of "species" or "numbers" is always present,
# so **Don't rename those please**)
ds.rename_properties({'energies': 'my_renamed_energies'})
# it is also possible to delete unwanted / unnedded properties
ds.delete_properties(('useless_property', 'bad_property'))

# Sometimes it may be useful to just create one placeholder property for some
# purpose.
# You can pass make the second dimension equal to the number of atoms in
# the group by passing is_atomic, and you can add also extra dims, for example:
# This creates a property with shape (N, A)
ds.create_full_property('my_new_property', is_atomic=True, fill_value=0.0, dtype=float)
# for more examples see docstring of the function

# It may be very useful in some cases to have only atomic numbers instead of
# chemical symbols, you can easily create this properties. If the dataset
# has atomic numbers and you want chemical symbols you can call:
ds.create_species_from_numbers()
# if the dataset has chemical symbols and you want atomic numbers:
ds.create_numbers_from_species()
# (I know "numbers" and "species" are bad names, don't blame me for that please)

# ######### Manipulating conformers #################
# (Watch out: dataset mutability modifies the underlying stores)
# All of the molecules in the dataset have the same properties
# Conformers as tensors can be appended by calling:
import torch  # noqa
conformers = {'numbers': torch.tensor([[1, 1, 6, 6], [1, 1, 6, 6]]),
              'species': torch.tensor([[1, 1, 6, 6], [1, 1, 6, 6]]),
              'coordinates': torch.randn(2, 4, 3),
              'energies': torch.randn(2)}
# Here I put random numbers as species and coordinates but you should put
# something that makes sense, if you have only one store you can pass
# "group_name" directly, also note that species should have the atomic numbers
ds.append_conformers('store_name/group_name', conformers)

# It is also possible to append conformers as numpy arrays, in this case
# "species" can hold the chemical symbols
import numpy as np  # noqa
numpy_conformers = {'numbers': np.array([[1, 1, 6, 6], [1, 1, 6, 6]]),
                    'species': np.array([['H', 'H', 'C', 'C'], ['H', 'H', 'C', 'C']]),
                    'coordinates': np.random.standard_normal((2, 4, 3)),
                    'energies': np.random.standard_normal(2)}
ds.append_conformers('store_name/group_name', numpy_conformers)
# Conformers can also be deleted from the dataset.
# This will delete conformers 0 and 2 only:
ds.delete_conformers('store_name/group_name', [0, 2])
# This will delete all conformers in the group:
ds.delete_conformers('store_name/group_name')

# NOTE: currently, when appending the class checks:
# - that the first dimension of all your properties is the same
# - that you are appending a set of conformers with the same properties that already exist in the dataset
# - that all your formulas are correct when the grouping type is "by_formula",
# - that your group name does not contain illegal "/" characters
# - that species is consistent with numbers if both are present
# It does NOT check:
# - That the number of atoms is the same in all properties that are atomic
#   (It doesn't know which properties are atomic so it can't)
# - That the name of the group is consistent with the formula / num atoms
#   (It allows arbitrary names)
# It is the responsibility of the user to make sure of that

# ###### Utilities #########
# Multiple datasets can be concatenated into one h5 file, optionally deleting the
# original h5 files if the concatenation is successful
from torchani.datasets.utils import concatenate  # noqa
concat_ds = concatenate(ds, '/path/to/concatenated/ds/', delete_originals=True)
# Another useful feature is deleting inplace all conformers with force magnitude
# above a given threshold
from torchani.datasets.utils import filter_by_high_force  # noqa
bad_conformations = filter_by_high_force(ds, delete_inplace=True)
print(bad_conformations)

# ######### Context manager usage #######################
# If you need to perform a lot of read/write operations in the dataset it can
# be useful to keep all the underlying stores open, you can do this
# by using a "keep_open" context:
with ds.keep_open('r+') as open_ds:
    for c in open_ds.iter_conformers():
        print(c)
    for group in ('bad_group1', 'bad_group2', 'bad_group3'):
        open_ds.delete_conformers(group)

# #### Creating a dataset from scratch ####
# It is possible to create an ANIDataset from scratch by calling:
ds = ANIDataset('/path/to/my/dataset.h5', create=True)
# By defalt the grouping is "by_formula"
# The first set of conformers you append will determine what properties this
# dataset will support
numpy_conformers = {'numbers': np.array([[1, 1, 6, 6], [1, 1, 6, 6]]),
                    'species': np.array([['H', 'H', 'C', 'C'], ['H', 'H', 'C', 'C']]),
                    'coordinates': np.random.standard_normal((2, 4, 3)),
                    'energies': np.random.standard_normal(2)}
ds.append_conformers('C2H2', numpy_conformers)
