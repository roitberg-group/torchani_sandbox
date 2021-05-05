from pathlib import Path
import pickle
import warnings
import torch
import h5py
import importlib
from typing import Union, Optional, List, Dict, Any
import numpy as np
from collections.abc import Mapping
from functools import partial


PKBAR_INSTALLED = importlib.util.find_spec('pkbar') is not None  # type: ignore
if PKBAR_INSTALLED:
    import pkbar


def save_batched_dataset(dataset, path: Union[str, Path], file_format='numpy', verbose=False, split='training'):
    if isinstance(path, str):
        path = Path(path).resolve()
    path = path.joinpath(split)
    if path.is_dir():
        subdirs = [d for d in path.iterdir()]
        assert not subdirs, "The path provided already has files or directories, please provide a different path"
    else:
        assert not path.is_file(), "The path provided is a file, not a directory"
        path.mkdir(parents=True)

    use_pbar = PKBAR_INSTALLED and verbose
    pbar = pkbar.Pbar(f'=> saving dataset to {path}, in format {file_format}, total molecules: {len(dataset)}', len(dataset))
    for j, batch in enumerate(dataset):
        if file_format == 'pickle':
            with open(path.joinpath(f'batch{j}.pkl'), 'wb') as batch_file:
                pickle.dump({'species': batch['species'].numpy(),
                        'coordinates': batch['coordinates'].numpy(),
                        'energies': batch['energies'].numpy()}, batch_file)
        elif file_format == 'numpy':
            np.savez(path.joinpath(f'batch{j}'),
                    species=batch['species'].numpy(),
                    coordinates=batch['coordinates'].numpy(),
                    energies=batch['energies'].numpy())
        if use_pbar:
            pbar.update(j)


class ANIBatchedDataset(torch.utils.data.Dataset):

    def __init__(self, store_dir: Union[str, Path], file_format: Optional[str] = None, split: str = 'training'):
        if isinstance(store_dir, str):
            store_dir = Path(store_dir).resolve()
        assert store_dir.is_dir(), f"The directory {store_dir.as_posix()} could not be found"
        self.split = split
        self.store_dir = store_dir.joinpath(split)
        msg = f"The directory {store_dir.as_posix()} exists, but the split {split} could not be found"
        assert self.store_dir.is_dir(), msg
        self.batch_paths = [f for f in self.store_dir.iterdir()]
        assert self.batch_paths, "The path provided has no files"
        assert all([f.is_file() for f in self.batch_paths]), "Subdirectories in path not supported"
        suffix = self.batch_paths[0].suffix
        assert all([f.suffix == suffix for f in self.batch_paths]), "Different file extensions in same path not supported"

        def numpy_extractor(idx, paths, pin_memory=False):
            return {
                k: torch.as_tensor(v).pin_memory() if pin_memory else torch.as_tensor(v)
                for k, v in np.load(paths[idx]).items()
            }

        def pickle_extractor(idx, paths, pin_memory=False):
            with open(paths[idx], 'rb') as f:
                return {
                    k: torch.as_tensor(v).pin_memory if pin_memory else torch.as_tensor(v)
                    for k, v in pickle.load(f).items()
                }

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

        self._data = [self.extractor(idx, pin_memory=True) for idx in range(len(self))]
        self.extractor = partial(memory_extractor, ds=self)
        return self

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
        self._groups: Dict[str, Any] = dict()
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
        if v.dtype == np.bytes_ or v.dtype == np.str_ or v.dtype.name == 'bytes8':
            v = [s.decode('ascii') for s in v]  # type: ignore
        return v
