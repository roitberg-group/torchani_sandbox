from pathlib import Path
import pickle
import warnings
import torch
import h5py
from typing import Union, Optional, List
import numpy as np
from collections.abc import Mapping
from functools import partial


class ANIBatchedDataset(torch.utils.data.Dataset):

    def __init__(self, store_dir: Union[str, Path], file_format: Optional[str] = None):
        if isinstance(store_dir, str):
            store_dir = Path(store_dir).resolve()
        assert store_dir.is_dir(), f"The directory {store_dir.as_posix()} could not be found"
        self.store_dir = store_dir
        self.batch_paths = [f for f in self.store_dir.iterdir()]
        assert self.batch_paths, "The path provided has no files"
        assert all([f.is_file() for f in self.batch_paths]), "Subdirectories in path not supported"
        suffix = self.batch_paths[0].suffix
        assert all([f.suffix == suffix for f in self.batch_paths]), "Different file extensions in same path not supported"

        def numpy_extractor(idx, paths):
            return {k: torch.as_tensor(v) for k, v in np.load(paths[idx]).items()}

        def pickle_extractor(idx, paths):
            with open(paths[idx], 'rb') as f:
                return {k: torch.as_tensor(v) for k, v in pickle.load(f).items()}

        if suffix == '.npz' or file_format == 'numpy':
            self.extractor = partial(numpy_extractor, paths=self.batch_paths)
        elif suffix == '.pkl' or file_format == 'pickle':
            self.extractor = partial(pickle_extractor, paths=self.batch_paths)
        else:
            msg = f'Format for file with extension {suffix} could not be infered, please specify explicitly'
            raise RuntimeError(msg)

    def cache(self):
        warnings.warn("Caching the dataset may a lot of memory, be careful")

        def memory_extractor(idx, ds):
            return ds._data[idx]

        self._data = [self.extractor(idx) for idx in range(len(self))]
        self.extractor = partial(memory_extractor, ds=self)

    def __getitem__(self, idx: int):
        # integral indices must be provided for compatibility with pytorch
        # dataloader API
        return self.extractor(idx)

    def __len__(self):
        return len(self.batch_paths)


class H5Dataset(Mapping):

    def __init__(self, store_file: Union[str, Path], flag_key: Optional[str] = None):
        if isinstance(store_file, str):
            store_file = Path(store_file).resolve()
        assert isinstance(store_file, Path)
        if not store_file.is_file():
            raise RuntimeError(f"The h5 file in {store_file.as_posix()} could not be found")

        self._store_file = store_file
        # flag key is used to infer size of molecule groups
        # when iterating over the dataset
        self._flag_key = flag_key
        self._groups = dict()
        self._cache_group_paths_and_sizes()
        self.num_conformers = sum(self._groups.values())
        self.num_conformer_groups = len(self._groups.keys())

    # API
    def __getitem__(self, key: str):
        return self._get_group(key, include_properties=None)

    def __len__(self):
        return self.num_conformer_groups

    def __iter__(self):
        # Iterating over groups and yield the associated molecule groups as
        # dictionaries of numpy arrays (except for species, which is a list of
        # strings)
        return iter(self._groups.keys())

    def get_conformers(self, key: str, idx: Optional[Union[int, np.ndarray]] = None, **kwargs):
        # fetching a conformer actually copies all the group into memory first,
        # so it is faster to fetch all the indices we need at the same time
        # using an array for indexing
        include_properties = kwargs.pop('include_properties', None)
        strict = kwargs.pop('strict', False)
        molecule_group = self._get_group(key, include_properties, strict)
        if idx is None:
            return molecule_group
        return self._extract_from_molecule_group(molecule_group, idx, **kwargs)

    def iter_conformers(self, **kwargs):
        # Iterating sequentially over conformers is also supported
        include_properties = kwargs.pop('include_properties', None)
        strict = kwargs.pop('strict', False)
        for k, size in self._groups.items():
            conformer_group = self.get_group(k, include_properties, strict)
            for j in range(size):
                yield self._extract_from_molecule_group(conformer_group, j, **kwargs)
    # end API

    def _cache_group_paths_and_sizes(self):
        # cache paths of all molecule groups into a list
        def visitor_fn(name, object_, ds: H5Dataset):
            # validate format of the dataset
            if isinstance(object_, h5py.Dataset):
                molecule_group = object_.parent
                for k, v in molecule_group.items():
                    if not isinstance(v, h5py.Dataset):
                        msg = "Invalid dataset format, there shouldn't be Groups inside Groups that have Datasets"
                        raise RuntimeError(msg)
                # If the format is correct cache the molecule group path, if it
                # hasn't been cached already
                if molecule_group.name not in ds._groups.keys():
                    size = ds._get_group_size(molecule_group)
                    ds._groups.update({molecule_group.name: size})

        with h5py.File(self._store_file, 'r') as f:
            f.visititems(partial(visitor_fn, ds=self))

    def _get_group_size(self, molecule_group):
        if self._flag_key is not None:
            try:
                size = len(molecule_group[self._flag_key])
            except KeyError:
                print(f'The flag key provided {self._flag_key} is not in {molecule_group.name}')
                raise
        else:
            molecule_keys = list(molecule_group.keys())
            if 'coordinates' in molecule_keys:
                size = len(molecule_group['coordinates'])
            elif 'energies' in molecule_keys:
                size = len(molecule_group['coordinates'])
            elif 'forces' in molecule_keys:
                size = len(molecule_group['coordinates'])
            else:
                msg = """Could not infer number of molecules in molecule
                         group since 'coordinates', 'forces' and 'energies' dont
                         exist, please provide a key that holds a dataset with the
                         moleucule size as its first axis / dim"""
                raise RuntimeError(msg)
        return size

    def _get_group(self, key: str, include_properties: Optional[List[str]] = None, strict: bool = False):
        # note that if include_properties are not found then this returns an
        # empty dict silently, unless strict is passed
        if include_properties is None:
            with h5py.File(self._store_file, 'r') as f:
                molecules = {k: self._parse_species(v[()]) for k, v in f[key].items()}
        else:
            if strict:
                msg = f"Some of the requested properties could not be found in group {key}"
                assert all([p in f[key].keys() for p in include_properties]), msg
            with h5py.File(self._store_file, 'r') as f:
                molecules = {k: self._parse_species(v[()]) for k, v in f[key].items() if k in include_properties}
        return molecules

    @staticmethod
    def _extract_from_molecule_group(molecule_group,
                                     idx: Optional[Union[int, np.ndarray]],
                                     element_keys=('smiles', 'species', 'numbers', 'atomic_numbers')):
        # this extraction procedure will fail if there are other keys in the
        # dataset besides "species", "numbers" and "atomic_numbers" that don't
        # have group_size as the 0th shape, in this case those keys have to be
        # specified

        if isinstance(idx, np.ndarray):
            assert idx.dtype == np.int, "The index has to be an integer"
            assert idx.ndim == 1, "Only vector indices are supported"

        conformer = {k: v[idx] for k, v in molecule_group.items() if k not in element_keys}
        # only one of these keys per molecule group exists
        for k in element_keys:
            try:
                conformer.update({k: molecule_group[k]})
            except KeyError:
                pass
        return conformer

    @staticmethod
    def _parse_species(v: np.ndarray):
        if v.dtype == np.bytes_ or v.dtype == np.str or v.dtype.name == 'bytes8':
            v = [s.decode('ascii') for s in v]
        return v


if __name__ == '__main__':
    dataset = H5Dataset('/home/ignacio/Datasets/ani1x_release_wb97x_dz.h5')

    # ############## Conformer groups:  ###########################
    # To access groups of conformers we can just use the dataset as a dictionary
    group = dataset['C10H10']
    print(group)

    # items(), values() and keys() work as expected for groups of conformers
    for k, v in dataset.items():
        print(k, v)

    for k in dataset.keys():
        print(k)

    for v in dataset.values():
        print(v)

    # to get the number of groups of conformers we can use len(), or num_conformer_groups
    num_groups = len(dataset)
    print(num_groups)
    num_groups = dataset.num_conformer_groups
    print(num_groups)

    # ############## Conformers:  ###########################
    # To access individual conformers or subsets of conformers we use *_conformer methods, get_conformers and iter_conformers
    conformer = dataset.get_conformers('C10H10', 0)
    print(conformer)
    conformer = dataset.get_conformers('C10H10', 1)
    print(conformer)

    # a numpy array can also be passed for indexing, to fetch multiple conformers
    # from the same group, which is faster. Since I copy the data for simplicity,
    # this allows all of numpy fancy indexing operations (directly indexing using
    # h5py does not
    conformers = dataset.get_conformers('C10H10', np.array([0, 1]))
    print(conformers)

    # We can also access all the group if we don't pass an index
    conformer = dataset.get_conformers('C10H10')
    print(conformer)

    # finally, it is possible to specify which properties we want using include_properties
    conformer = dataset.get_conformers('C10H10', include_properties=('species', 'energies'))
    print(conformer)

    conformer = dataset.get_conformers('C10H10', np.array([0, 3]), include_properties=('species', 'energies'))
    print(conformer)

    # we can iterate over all conformers sequentially by calling iter_conformer,
    # this is faster than doing it manually since it caches each conformer group
    # previous to starting the iteration
    for c in dataset.iter_conformers():
        print(c)

    # to get the number of conformers we can use num_conformers
    num_conformers = dataset.num_conformers
