from typing import (Union, Optional, Dict, Sequence, Iterator, Tuple, List, Set,
                    Mapping, Any, Iterable, Callable, TypeVar, ContextManager)
import json
import re
import pickle
import warnings
import shutil
from copy import deepcopy
from pathlib import Path
from functools import partial, wraps
from contextlib import ExitStack, contextmanager
from collections import OrderedDict

import torch
from torch import Tensor
import numpy as np

from ._backends import _H5PY_AVAILABLE
from ._annotations import (Transform, Conformers, NumpyConformers, MaybeNumpyConformers, PathLike, DTypeLike,
                           PathLikeODict, H5Group, H5File, H5Dataset)
from ..utils import species_to_formula, PERIODIC_TABLE, ATOMIC_NUMBERS, tqdm

if _H5PY_AVAILABLE:
    import h5py


Extractor = Callable[[int], Conformers]
_T = TypeVar('_T')

# IMPORTANT

# ANIDataset is a mapping, The mapping has keys "group_names" or "names" and
# values "conformers" or "conformer_group". Each group of conformers is also a
# mapping, where keys are "properties" and values are numpy arrays / torch
# tensors (they are just referred to as "values" or "data").
#
# In the current HDF5 datasets the group names are formulas (in some
# CCCCHHH.... etc, in others C2H4, etc) groups could also be smiles or number
# of atoms. Since HDF5 is hierarchical this grouping is essentially hardcoded
# into the dataset format.
#
# To parse all current HDF5 dataset types it is necessary to first determine
# where all the conformer groups are. HDF5 has directory structure, and in
# principle they could be arbitrarily located. One would think that there is
# some sort of standarization between the datasets, but unfortunately there is
# none (!!), and the legacy reader, anidataloader, just scans all the groups
# recursively...
#
# Cache update part 1:
# --------------------
# Since scanning recursively is super slow we just do this once and cache the
# location of all the groups, and the sizes of all the groups inside
# "groups_sizes". After this, it is not necessary to do the recursion again
# unless some modification to the dataset happens, in which case we need a
# cache update, to get "group_sizes" and "properties" again. Methods that
# modify the dataset are decorated so that the internal cache is updated.
#
# If the dataset has some semblance of standarization (it is a tree with depth
# 1, where all groups are directly joined to the root) then it is much faster
# to traverse the dataset. This can be assumed passing "assume_standarized". In
# any case after the first recursion if this structure is detected the flag is
# set internally so we never do the recursion again. This speeds up cache
# updates and lookup x30
#
# Cache update part 2:
# --------------------
# There is in principle no guarantee that all conformer groups have the same
# properties. Due to this we have to first traverse the dataset and check that
# this is the case. We store the properties the dataset supports inside an
# internal variable _properties (e.g. it may happen that one molecule has
# forces but not coordinates, if this happens then ANIDataset raises an error)

# Multiple files:
# ---------------
# Current datasets need ANIDataset to be able to manage multiple files, this is
# achieved by delegating execution of the methods to one of the _ANISubdataset
# instances that ANIDataset
# contains. Basically any method is either:
# 1 - delegated to a subdataset: If you ask for the conformer group "ANI1x/CH4"
#     in the "full ANI2x" dataset (ANI1x + ANI2x_FSCl + dimers),
#     then this will be delegated to the "ANI1x" subdataset.
# 2 - broadcasted to all subdatasets: If you want to rename a property or
#     delete a property it will be deleted / renamed in all subdatasets.
#
# This mechanism is currently achieved by decorating dummy methods but there
# may be some more elegant way.
#
# ContextManager usage:
# ----------------
# You can turn the dataset into a context manager that keeps all HDF5 files
# open simultaneously by using with ds.keep_open('r') as ro_ds:, for example.
# It seems that HDF5 is quite slow when opening files, it has to aqcuire locks,
# and do other things, so this speeds up iteration by 12 - 13 % usually. Since
# many files may need to be opened at the same time then ExitStack is needed to
# properly clean up everything. Each time a method needs to open a file it first
# checks if it is already open (i.e. we are inside a 'keep_open' context) in that
# case it just fetches the already opened file.


def _get_dim_size(conformers: Union[H5Group, Conformers, NumpyConformers], *,
                  common_keys: Set[str],
                  dim: int) -> int:
    # Calculates the dimension size in a conformer group. It tries to get it
    # from one of a number of the "common keys" that have the dimension
    present_keys = common_keys.intersection(set(conformers.keys()))
    if present_keys:
        size = conformers[tuple(present_keys)[0]].shape[dim]
    else:
        msg = (f'Could not infer dimension size of dim {dim} in properties '
               f' since {common_keys} are missing')
        raise RuntimeError(msg)
    return size


# calculates number of atoms in a conformer group
_get_num_atoms = partial(_get_dim_size, common_keys={'coordinates', 'coord', 'forces'}, dim=1)
# calculates number of conformers in a conformer group
_get_num_conformers = partial(_get_dim_size, common_keys={'coordinates', 'coord', 'forces', 'energies'}, dim=0)


class ANIBatchedDataset(torch.utils.data.Dataset[Conformers]):

    _SUFFIXES_AND_FORMATS = {'.npz': 'numpy', '.h5': 'hdf5', '.pkl': 'pickle'}
    batch_size: int

    def __init__(self, store_dir: PathLike,
                       file_format: Optional[str] = None,
                       split: str = 'training',
                       transform: Transform = lambda x: x,
                       properties: Optional[Sequence[str]] = ('coordinates', 'species', 'energies'),
                       drop_last: bool = False):

        self.split = split
        self.store_dir = Path(store_dir).resolve().joinpath(self.split)
        if not self.store_dir.is_dir():
            raise ValueError(f'The directory {self.store_dir.as_posix()} exists, '
                             f'but the split {split} could not be found')

        self.batch_paths = [f for f in self.store_dir.iterdir()]

        if not self.batch_paths:
            raise RuntimeError("The path provided has no files")
        if not all(f.is_file() for f in self.batch_paths):
            raise RuntimeError("Subdirectories were found in path, this is not supported")

        # sort batches according to batch numbers, batches are assumed to have a name
        # '<chars><number><chars>.suffix' where <chars> has only non numeric characters
        # by default batches are named batch<number>.suffix by create_batched_dataset
        batch_numbers: List[int] = []
        for b in self.batch_paths:
            matches = re.findall(r'\d+', b.with_suffix('').name)
            if not len(matches) == 1:
                raise ValueError(f"Batches must have one and only one number but found {matches} for {b.name}")
            batch_numbers.append(int(matches[0]))
        if not len(set(batch_numbers)) == len(batch_numbers):
            raise ValueError(f"Batch numbers must be unique but found {batch_numbers}")
        self.batch_paths = [p for _, p in sorted(zip(batch_numbers, self.batch_paths))]

        suffix = self.batch_paths[0].suffix
        if not all(f.suffix == suffix for f in self.batch_paths):
            raise RuntimeError("Different file extensions were found in path, not supported")

        self.transform = transform

        # We use pickle or numpy or hdf5 since saving in
        # pytorch format is extremely slow
        if file_format is None:
            file_format = self._SUFFIXES_AND_FORMATS[suffix]
        elif file_format not in self._SUFFIXES_AND_FORMATS.values():
            raise ValueError(f"The file format {file_format} is not one of the"
                             f"supported formats {self._SUFFIXES_AND_FORMATS.values()}")
        if file_format == 'hdf5' and not _H5PY_AVAILABLE:
            raise ValueError("File format hdf5 was specified but h5py could not"
                             " be found, please install h5py or specify a "
                             " different file format")

        def numpy_extractor(idx: int, paths: List[Path], properties: Optional[Sequence[str]]) -> Conformers:
            return {k: torch.as_tensor(v) for k, v in np.load(paths[idx]).items() if properties is None or k in properties}

        def pickle_extractor(idx: int, paths: List[Path], properties: Optional[Sequence[str]]) -> Conformers:
            with open(paths[idx], 'rb') as f:
                return {k: torch.as_tensor(v) for k, v in pickle.load(f).items() if properties is None or k in properties}

        def hdf5_extractor(idx: int, paths: List[Path], properties: Optional[Sequence[str]]) -> Conformers:
            with h5py.File(paths[idx], 'r') as f:
                return {k: torch.as_tensor(v[()]) for k, v in f['/'].items() if properties is None or k in properties}

        self._extractor: Extractor = {'numpy': partial(numpy_extractor, paths=self.batch_paths, properties=properties),
                                      'pickle': partial(pickle_extractor, paths=self.batch_paths, properties=properties),
                                      'hdf5': partial(hdf5_extractor, paths=self.batch_paths, properties=properties)}[file_format]

        try:
            with open(self.store_dir.parent.joinpath('creation_log.json'), 'r') as logfile:
                creation_log = json.load(logfile)
            self._is_inplace_transformed = creation_log['is_inplace_transformed']
            self.batch_size = creation_log['batch_size']
        except FileNotFoundError:
            warnings.warn("No creation log found, is_inplace_transformed assumed False")
            self._is_inplace_transformed = False
            # All batches except the last are assumed to have the same length
            first_batch = self[-1]
            self.batch_size = _get_num_conformers(first_batch)

        if drop_last:
            # Drops last batch only if it is smallest than the rest
            last_batch = self[-1]
            last_batch_size = _get_num_conformers(last_batch)
            if last_batch_size < self.batch_size:
                self.batch_paths.pop()
        self._len = len(self.batch_paths)

    def cache(self, pin_memory: bool = True,
              verbose: bool = True,
              apply_transform: bool = True) -> 'ANIBatchedDataset':
        r"""Saves the full dataset into RAM"""
        desc = f'Cacheing {self.split}, Warning: this may use a lot of RAM!'
        self._data = [self._extractor(idx) for idx in tqdm(range(len(self)),
                                                          total=len(self),
                                                          disable=not verbose,
                                                          desc=desc)]
        if apply_transform:
            desc = "Applying transforms once and discarding"
            with torch.no_grad():
                self._data = [self.transform(p) for p in tqdm(self._data,
                                                              total=len(self),
                                                              disable=not verbose,
                                                              desc=desc)]
            self.transform = lambda x: x
        if pin_memory:
            desc = 'Pinning memory; dont pin memory in torch DataLoader!'
            self._data = [{k: v.pin_memory()
                           for k, v in batch.items()}
                           for batch in tqdm(self._data,
                                                  total=len(self),
                                                  disable=not verbose,
                                                  desc=desc)]
        self._extractor = lambda idx: self._data[idx]
        return self

    def __getitem__(self, idx: int) -> Conformers:
        # integral indices must be provided for compatibility with pytorch
        # DataLoader API
        batch = self._extractor(idx)
        with torch.no_grad():
            batch = self.transform(batch)
        return batch

    def __len__(self) -> int:
        return self._len


class _ANIDatasetBase(Mapping[str, Conformers]):
    # Base class for ANIDataset and _ANISubdataset

    def __init__(self, *args, **kwargs) -> None:
        self.group_sizes: 'OrderedDict[str, int]' = OrderedDict()
        # "properties" is a persistent attribute needed for validation of
        # inputs, and may change if a property in the dataset is renamed or is
        # deleted.
        #
        # "properties" is read only and backed by _properties.
        # The variables _batch_properties, _nonbatch_properties, num_conformers
        # and num_conformer_groups are all calculated on the fly to guarantee
        # synchronization with "properties".
        # nonbatch and batch properties distinction is needed due to the legacy
        # format having some properties with no "batch" dimension
        self._properties: Set[str] = set()
        self._possible_nonbatch_properties: Set[str] = set()

    @property
    def properties(self) -> Set[str]:
        # read-only, can be called by user code
        return self._properties

    @property
    def _batch_properties(self) -> Set[str]:
        batch_properties = {k for k in self.properties
                            if k not in self._possible_nonbatch_properties}
        return batch_properties

    @property
    def _nonbatch_properties(self) -> Set[str]:
        batch_properties = {k for k in self.properties
                            if k in self._possible_nonbatch_properties}
        return batch_properties

    @property
    def num_conformers(self):
        return sum(self.group_sizes.values())

    @property
    def num_conformer_groups(self):
        return len(self.group_sizes.keys())

    @property
    def grouping(self) -> str:
        raise NotImplementedError

    def _set_grouping(self, grouping: str) -> None:
        raise NotImplementedError

    def __getitem__(self, key: str) -> Conformers:
        return self.get_conformers(key)

    def __len__(self) -> int:
        return self.num_conformer_groups

    def __iter__(self) -> Iterator[str]:
        return iter(self.group_sizes.keys())

    def get_conformers(self,
                       group_name: str,
                       idx: Optional[Tensor] = None, *,
                       properties: Optional[Iterable[str]] = None) -> Conformers:
        raise NotImplementedError

    def get_numpy_conformers(self,
                             group_name: str,
                             idx: Optional[Tensor] = None, *,
                             properties: Optional[Iterable[str]] = None) -> NumpyConformers:
        raise NotImplementedError

    def numpy_items(self, **kwargs) -> Iterator[Tuple[str, NumpyConformers]]:
        for group_name in self.keys():
            yield group_name, self.get_numpy_conformers(group_name, **kwargs)

    def numpy_values(self, **kwargs) -> Iterator[NumpyConformers]:
        for group_name in self.keys():
            yield self.get_numpy_conformers(group_name, **kwargs)

    def iter_key_idx_conformers(self, **kwargs) -> Iterator[Tuple[str, int, Conformers]]:
        for k, size in self.group_sizes.items():
            conformers = self.get_conformers(k, **kwargs)
            for idx in range(size):
                single_conformer = {k: conformers[k][idx] for k in conformers.keys()}
                yield k, idx, single_conformer

    def iter_conformers(self, **kwargs) -> Iterator[Conformers]:
        for _, _, c in self.iter_key_idx_conformers(**kwargs):
            yield c


# Decorators for ANIDataset
def _broadcast(method: Callable[..., 'ANIDataset']) -> Callable[..., 'ANIDataset']:
    # Decorator that wraps functions from ANIDataset that should be
    # delegated to all of its "_ANISubdataset" members in a loop.
    @wraps(method)
    def delegated_method_call(self: 'ANIDataset', *args, **kwargs) -> 'ANIDataset':
        for name in self._datasets.keys():
            self._datasets[name] = getattr(self._datasets[name], method.__name__)(*args, **kwargs)
        return self._update_internal_cache()
    return delegated_method_call


def _delegate(method: Callable[..., 'ANIDataset']) -> Callable[..., 'ANIDataset']:
    # Decorator that wraps functions from ANIDataset that should be
    # delegated to one of its "_ANISubdataset" members,
    @wraps(method)
    def delegated_method_call(self: 'ANIDataset', group_name: str, *args, **kwargs) -> 'ANIDataset':
        name, k = self._parse_key(group_name)
        getattr(self._datasets[name], method.__name__)(k, *args, **kwargs)
        return self._update_internal_cache()
    return delegated_method_call


def _delegate_with_return(method: Callable[..., _T]) -> Callable[..., _T]:
    # Decorator that wraps functions from ANIDataset that should be
    # delegated to one of its "_ANISubdataset" members.
    @wraps(method)
    def delegated_method_call(self: 'ANIDataset', group_name: str, *args, **kwargs) -> Any:
        name, k = self._parse_key(group_name)
        return getattr(self._datasets[name], method.__name__)(k, *args, **kwargs)
    return delegated_method_call


class ANIDataset(_ANIDatasetBase):
    # Essentially a container of _ANISubdataset instances that forwards
    # calls to the corresponding files in an appropriate way. Methods are
    # decorated depending on the forward manner, "_delegate" just calls the
    # method in one specific Subdataset "_broadcast" calls method in all the
    # Subdatasets and "_delegate_with_return" is like "_delegate" but has a
    # return value other than self, so it can't be chained
    def __init__(self, dataset_paths: Union[PathLike, PathLikeODict, Sequence[PathLike]], **kwargs):
        super().__init__()

        if isinstance(dataset_paths, (Path, str)):
            dataset_paths = [Path(dataset_paths).resolve()]

        if isinstance(dataset_paths, OrderedDict):
            od = [(k, _ANISubdataset(v, **kwargs)) for k, v in dataset_paths.items()]
        else:
            od = [(str(j), _ANISubdataset(v, **kwargs)) for j, v in enumerate(dataset_paths)]
        self._datasets = OrderedDict(od)
        self._num_subds = len(self._datasets)

        # save pointer to the first subds as attr, useful for code clarity
        self._first_name, self._first_subds = next(iter(self._datasets.items()))
        self._update_internal_cache()

    @contextmanager
    def keep_open(self, mode: str = 'r') -> Iterator['ANIDataset']:
        with ExitStack() as stack:
            for k in self._datasets.keys():
                self._datasets[k] = stack.enter_context(self._datasets[k].keep_open(mode))
            yield self

    def present_species(self) -> Tuple[str, ...]:
        present_species = {s for ds in self._datasets.values() for s in ds.present_species()}
        return tuple(sorted(present_species))

    # conformer getters/setters/deleters
    @_delegate_with_return
    def get_conformers(self, key: str, **kwargs) -> Conformers: ...  # noqa E704

    @_delegate_with_return
    def get_numpy_conformers(self, key: str, **kwargs) -> NumpyConformers: ...  # noqa E704

    @_delegate
    def append_conformers(self, group_name: str, *args, **kwargs) -> 'ANIDataset': ...  # noqa E704

    @_delegate
    def append_numpy_conformers(self, group_name: str, *args, **kwargs) -> 'ANIDataset': ...  # noqa E704

    @_delegate
    def delete_conformers(self, group_name: str, *args, **kwargs) -> 'ANIDataset': ...  # noqa E704

    # regrouping
    @_broadcast
    def regroup_by_formula(self, *args, **kwargs) -> 'ANIDataset': ...  # noqa E704

    @_broadcast
    def regroup_by_num_atoms(self, *args, **kwargs) -> 'ANIDataset': ...  # noqa E704

    @property
    def grouping(self) -> str:
        return self._first_subds.grouping

    # property manipulation ("columnwise" in relational ds)
    @_broadcast
    def create_species_from_numbers(self, *args, **kwargs) -> 'ANIDataset': ...  # noqa E704

    @_broadcast
    def create_numbers_from_species(self, *args, **kwargs) -> 'ANIDataset': ...  # noqa E704

    @_broadcast
    def extract_slice_as_new_group(self, *args, **kwargs) -> 'ANIDataset': ...  # noqa E704

    @_broadcast
    def create_full_scalar_property(self, *args, **kwargs) -> 'ANIDataset': ...  # noqa E704

    @_broadcast
    def delete_properties(self, *args, **kwargs) -> 'ANIDataset': ...  # noqa E704

    @_broadcast
    def rename_properties(self, *args, **kwargs) -> 'ANIDataset': ...  # noqa E704

    @_broadcast
    def set_aliases(self, *args, **kwargs) -> 'ANIDataset': ...  # noqa E704

    @_broadcast
    def repack(self, *args, **kwargs) -> 'ANIDataset': ... # noqa E704

    def __str__(self) -> str:
        str_ = ''
        for ds in self._datasets.values():
            str_ += f'{ds}\n'
        return str_

    def _update_internal_cache(self) -> 'ANIDataset':
        if self._num_subds > 1:
            od_args = [(f'{name}/{k}', v) for name, ds in self._datasets.items() for k, v in ds.group_sizes.items()]
        else:
            od_args = [(k, v) for ds in self._datasets.values() for k, v in ds.group_sizes.items()]
        self.group_sizes = OrderedDict(od_args)
        for name, ds in self._datasets.items():
            if not ds.grouping == self._first_subds.grouping:
                raise ValueError("Datasets have incompatible groupings,"
                                 f" got {self._first_subds.grouping} for {self._first_name}"
                                 f" and {ds.grouping} for {name}")

            if not ds.properties == self._first_subds.properties:
                raise ValueError('Supported properties are different for the'
                                 ' component subdatasets, got'
                                 f' {self._first_subds.properties} for {self._first_name}'
                                 f' and {ds.properties} for {name}')
        self._properties = self._first_subds.properties
        return self

    def _parse_key(self, key: str) -> Tuple[str, str]:
        tokens = key.split('/')
        if self._num_subds > 1:
            return tokens[0], '/'.join(tokens[1:])
        else:
            return self._first_name, '/'.join(tokens)


class AniH5Dataset(ANIDataset):

    def __init__(self, *args, **kwargs) -> None:
        warnings.warn("AniH5Dataset has been renamed to ANIDataset, please use ANIDataset instead")
        super().__init__(*args, **kwargs)


class AniBatchedDataset(ANIBatchedDataset):

    def __init__(self, *args, **kwargs) -> None:
        warnings.warn("AniBatchedDataset has been renamed to ANIBatchedDataset, please use ANIBatchedDataset instead")
        super().__init__(*args, **kwargs)


# Decorator for ANISubdataset
def _needs_cache_update(method: Callable[..., '_ANISubdataset']) -> Callable[..., '_ANISubdataset']:
    # Decorator that wraps functions that modify the dataset in place. Makes
    # sure that cache updating happens after dataset modification
    @wraps(method)
    def method_with_cache_update(ds: '_ANISubdataset', *args, **kwargs) -> '_ANISubdataset':
        ds = method(ds, *args, **kwargs)
        ds._update_internal_cache()
        return ds

    return method_with_cache_update


class _ANISubdataset(_ANIDatasetBase):

    _SUPPORTED_STORES = ('.h5',)

    def __init__(self,
                 store_location: PathLike, *,
                 assume_standard: bool = False,
                 create: bool = False,
                 grouping: Optional[str] = None,
                 property_aliases: Optional[Dict[str, str]] = None,
                 verbose: bool = True):
        super().__init__()
        # Private wrapper over backing storage, with some modifications it
        # could be used for directories with npz files. It should never ever be
        # used directly by user code.

        self._store_location = Path(store_location).resolve()
        if self._store_location.suffix == '.h5' or self._store_location.suffix not in self._SUPPORTED_STORES:
            if not _H5PY_AVAILABLE:
                raise ValueError('h5py backend was specified but h5py could not be found, please install h5py')
            self._backend = 'h5py'  # the only backend allowed currently, so we default to it

        self._open_store: Optional['_DatasetStoreAdaptor'] = None
        self._verbose = verbose
        self._symbols_to_numbers = np.vectorize(lambda x: ATOMIC_NUMBERS[x])
        self._numbers_to_symbols = np.vectorize(lambda x: PERIODIC_TABLE[x])

        # "storename" are the names of properties in the store file, and
        # "aliases" are the names users see when manipulating
        self._storename_to_alias = dict() if property_aliases is None else property_aliases
        self._alias_to_storename = {v: k for k, v in self._storename_to_alias.items()}

        # group_sizes and properties are needed as internal cache
        # variables, this cache is updated if something in the dataset changes
        if create:
            # Create the file, since there is nothing in it homogeneous
            # properties and standard format are assumed. Default grouping is
            # "by formula"
            open(self._store_location, 'x').close()
            self._has_standard_format = True
            self._checked_homogeneous_properties = True
            self._set_grouping('by_formula' if grouping is None else grouping)
        else:
            # In general all supported properties of the dataset should be
            # equal for all groups, this is not a problem for relational databases
            # but it can be an issue for HDF5. First we check that this is the case inside
            # _update_internal_cache, if it isn't then we raise an error, if
            # the check passes we set _checked_homogeneous_properties = True,
            # so that we don't run the costly check again
            # default grouping is "unspecified"
            if not self._store_location.is_file():
                raise FileNotFoundError(f"The h5 file in {self._store_location.as_posix()} could not be found")
            self._has_standard_format = assume_standard
            self._checked_homogeneous_properties = False
            self._set_grouping(self.grouping if grouping is None else grouping)
            self._update_internal_cache()

    @contextmanager
    def keep_open(self, mode: str = 'r') -> Iterator['_ANISubdataset']:
        r"""Context manager to keep dataset open while iterating over it

        Usage:
        with ds.keep_open('r') as ro_ds:
            c = ro_ds.get_conformers('CH4')
        etc
        this speeds up access in the context of many operations in a block,
        e.g.
        with ds.keep_open('r') as ro_ds:
            for c in ro_ds.iter_conformers():
                print(c)
        may be much faster than directly iterating over conformers
        """
        if self._backend == 'h5py':
            context_manager = h5py.File(self._store_location, mode)
        else:
            raise RuntimeError(f"Bad backend {self._backend}")
        self._open_store = _DatasetStoreAdaptor(context_manager, self._alias_to_storename)
        try:
            yield self
        finally:
            assert self._open_store is not None
            self._open_store.close()

    def _get_open_store(self, stack: ExitStack, mode: str = 'r') -> '_DatasetStoreAdaptor':
        # This trick makes methods fetch the open file directly
        # if they are being called from inside a "keep_open" context
        if self._open_store is None:
            if self._backend == 'h5py':
                context_manager = h5py.File(self._store_location, mode)
            else:
                raise RuntimeError(f"Bad backend {self._backend}")
            return stack.enter_context(_DatasetStoreAdaptor(context_manager, self._alias_to_storename))
        else:
            current_mode = self._open_store.mode
            assert mode in ['r+', 'r'], f"Unsupported mode {mode}"
            if mode == 'r+' and current_mode == 'r':
                msg = ('Tried to open a file with mode "r+" but the dataset is'
                       ' currently keeping its store file open with mode "r"')
                raise RuntimeError(msg)
            return self._open_store

    def _update_internal_cache(self) -> None:
        self.group_sizes = OrderedDict()
        self._properties = set()

        if self._backend == 'h5py':
            # detect datasets with _created / _meta standarization, or which
            # were regrouped. These have standard format.
            if not self._has_standard_format:
                try:
                    with ExitStack() as stack:
                        # get the raw hdf5 file object for manipulations inside this function
                        raw_hdf5_store = self._get_open_store(stack, 'r')._store_obj
                        raw_hdf5_store['/_created']
                        self._has_standard_format = True
                except KeyError:
                    pass

            if self._has_standard_format:
                # This is much faster (x30) than a visitor function but it assumes
                # the format is somewhat standard which means that all Groups have
                # depth 1, and all Datasets have depth 2.
                with tqdm(desc=f'Scanning {self._store_location.name} assuming standard format',
                          disable=not self._verbose) as pbar:
                    with ExitStack() as stack:
                        # get the raw hdf5 file object for manipulations inside this function
                        raw_hdf5_store = self._get_open_store(stack, 'r')._store_obj
                        for k, g in raw_hdf5_store.items():
                            pbar.update()
                            if g.name in ['/_created', '/_meta']:
                                continue
                            self._update_properties_cache_h5py(g)
                            self._update_groups_cache_h5py(g)
            else:
                def visitor_fn(name: str,
                               object_: Union[H5Dataset, H5Group],
                               dataset: '_ANISubdataset',
                               pbar: Any) -> None:
                    pbar.update()
                    # We make sure the node is a Dataset, and We avoid Datasets
                    # called _meta or _created since if present these store units
                    # or other metadata. We also check if we already visited this
                    # group via one of its children.
                    if not isinstance(object_, H5Dataset) or\
                           object_.name in ['/_created', '/_meta'] or\
                           object_.parent.name in dataset.group_sizes.keys():
                        return
                    g = object_.parent
                    # Check for format correctness
                    for v in g.values():
                        if isinstance(v, h5py.Group):
                            msg = (f"Invalid dataset format, there shouldn't be "
                                    "Groups inside Groups that have Datasets, "
                                    f"but {g.name}, parent of the dataset "
                                    f"{object_.name}, has group {v.name} as a "
                                    "child")
                            raise RuntimeError(msg)
                    dataset._update_properties_cache_h5py(g)
                    dataset._update_groups_cache_h5py(g)

                with ExitStack() as stack:
                    raw_hdf5_store = self._get_open_store(stack, 'r')._store_obj
                    with tqdm(desc='Verifying format correctness',
                              disable=not self._verbose) as pbar:
                        raw_hdf5_store.visititems(partial(visitor_fn, dataset=self, pbar=pbar))

                # If the visitor function succeeded and this condition is met the
                # dataset must be in standard format
                self._has_standard_format = not any('/' in k[1:] for k in self.group_sizes.keys())
        else:
            raise RuntimeError(f"Bad backend {self._backend}")

        self._checked_homogeneous_properties = True

        # By default iteration of HDF5 should be alphanumeric in which case
        # sorting should not be necessary, this internal assert ensures the
        # groups were not created with 'track_order=True', and that the visitor
        # function worked properly.
        assert list(self.group_sizes) == sorted(self.group_sizes), "Groups were not iterated upon alphanumerically"

    def _update_properties_cache_h5py(self, conformers: H5Group) -> None:
        # Updates the "_properties", variables. "_nonbatch_properties" are keys that
        # don't have a batch dimension, their shape must be (atoms,), they only make sense
        # if ordering by formula or smiles
        if not self.properties:
            self._properties = {self._storename_to_alias.get(p, p) for p in set(conformers.keys())}
        elif not self._checked_homogeneous_properties:
            found_properties = {self._storename_to_alias.get(p, p) for p in set(conformers.keys())}
            if not found_properties == self.properties:
                msg = (f"Group {conformers.name} has bad keys, "
                       f"found {found_properties}, but expected "
                       f"{self.properties}")
                raise RuntimeError(msg)

    def _update_groups_cache_h5py(self, conformers: H5Group) -> None:
        # updates "group_sizes" which holds the batch dimension (number of
        # molecules) of all grups in the dataset.
        self.group_sizes.update({conformers.name[1:]: _get_num_conformers(conformers)})

    def __str__(self) -> str:
        str_ = "ANI HDF5 File:\n"
        str_ += f"Supported properties: {self.properties}\n"
        str_ += f"Number of conformers: {self.num_conformers}\n"
        str_ += f"Number of conformer groups: {self.num_conformer_groups}\n"
        try:
            str_ += f"Present elements: {self.present_species()}\n"
        except ValueError:
            str_ += "Present elements: Unknown\n"
        return str_

    def present_species(self) -> Tuple[str, ...]:
        r"""Get an ordered tuple with all species present in the dataset"""
        if 'species' in self.properties:
            element_key = 'species'
            parser = lambda s: set(s['species'].ravel())  # noqa E731
        elif 'numbers' in self.properties:
            element_key = 'numbers'
            parser = lambda s: set(self._numbers_to_symbols(s['numbers']).ravel())  # noqa E731
        else:
            raise ValueError('"species" or "numbers" must be present to parse symbols')
        present_species: Set[str] = set()
        for group_name in self.keys():
            species = self.get_numpy_conformers(group_name, properties=element_key)
            present_species.update(parser(species))
        return tuple(sorted(present_species))

    def get_conformers(self,
                       group_name: str,
                       idx: Optional[Tensor] = None, *,
                       properties: Optional[Iterable[str]] = None) -> Conformers:
        r"""Get conformers in a given group in the dataset, with specified
        indices, and including only specified properties.  conformers are dict
        of the form {property: Tensor}, where properties are strings"""
        if isinstance(properties, str):
            properties = {properties}
        if properties is None:
            requested_properties = self.properties
        else:
            requested_properties = set(properties)
        # The tensor counterpart of get_numpy_conformers
        numpy_conformers = self.get_numpy_conformers(group_name, idx, properties=requested_properties)
        conformers = {k: torch.tensor(numpy_conformers[k]) for k in requested_properties.difference({'species', '_id'})}

        if 'species' in requested_properties:
            species = self._symbols_to_numbers(numpy_conformers['species'])
            conformers.update({'species': torch.from_numpy(species)})
        return conformers

    def get_numpy_conformers(self,
                             group_name: str,
                             idx: Optional[Tensor] = None, *,
                             properties: Optional[Iterable[str]] = None) -> NumpyConformers:
        r"""Same as get_conformers but conformers are a dict {property: ndarray}"""
        if isinstance(properties, str):
            properties = {properties}
        if properties is None:
            requested_properties = self.properties
        else:
            requested_properties = set(properties)
        self._check_properties_are_present(requested_properties)

        # Determine which of the properties passed are batched and which are nonbatch
        requested_nonbatch_properties = self._nonbatch_properties.intersection(requested_properties)
        requested_batch_properties = self._batch_properties.intersection(requested_properties)

        with ExitStack() as stack:
            f = self._get_open_store(stack, 'r')
            numpy_conformers = {p: f[group_name][p] for p in requested_properties}
            if idx is not None:
                assert idx.dim() <= 1, "index must be a 0 or 1 dim tensor"
                numpy_conformers.update({k: numpy_conformers[k][idx.cpu().numpy()]
                                        for k in requested_batch_properties})

        if requested_nonbatch_properties:
            tile_shape = (_get_num_conformers(numpy_conformers), 1) if idx is None or idx.dim() == 1 else (1,)
            numpy_conformers.update({k: np.tile(numpy_conformers[k], tile_shape)
                                     for k in requested_nonbatch_properties})

        if 'species' in requested_properties:
            numpy_conformers['species'] = numpy_conformers['species'].astype(str)
        if '_id' in requested_properties:
            numpy_conformers['_id'] = numpy_conformers['_id'].astype(str)

        return numpy_conformers

    def append_conformers(self, group_name: str, conformers: Conformers) -> '_ANISubdataset':
        r"""Attach a new set of conformers to the dataset. Conformers must be
        a dict {property: Tensor}, and they must have the same properties that the dataset
        supports. Appending is only supported for grouping 'by_formula' or
        'by_num_atoms'"""
        conformers = deepcopy(conformers)
        group_name, conformers = self._check_append_input(group_name, conformers)
        numpy_conformers = {k: conformers[k].detach().cpu().numpy() for k in self.properties.difference({'species'})}

        if 'species' in self.properties:
            if (conformers['species'] <= 0).any():
                raise ValueError('Species are atomic numbers, must be positive')
            species = self._numbers_to_symbols(conformers['species'].detach().cpu().numpy())
            numpy_conformers.update({'species': species})

        return self.append_numpy_conformers(group_name, numpy_conformers)

    @_needs_cache_update
    def append_numpy_conformers(self, group_name: str, conformers: NumpyConformers) -> '_ANISubdataset':
        r"""Same as append_conformers but conformers must be a dict {property: ndarray}"""
        group_name, conformers = self._check_append_input(group_name, conformers)
        with ExitStack() as stack:
            f = self._get_open_store(stack, 'r+')
            try:
                group = f.create_conformer_group(group_name)
                group.create_numpy_values(conformers)
            except ValueError:
                group = f[group_name]
                if not group.is_resizable():
                    raise RuntimeError("Dataset must be resizable to allow appending")
                group.append_numpy_values(conformers)
        return self

    def _check_append_input(self, group_name: str, conformers: MaybeNumpyConformers) -> Tuple[str, MaybeNumpyConformers]:
        if self.grouping not in ['by_formula', 'by_num_atoms']:
            raise ValueError("Can't append if the grouping is not by_formula or"
                             " by_num_atoms, please regroup your dataset")

        # check that all formulas are the same
        if self.grouping == 'by_formula':
            formulas = np.asarray(self._get_conformer_formulas(conformers))
            if not np.all(formulas[0] == formulas):
                raise ValueError("All appended conformers must have the same formula")

        # If this is the first conformer added update the dataset to support
        # these properties, otherwise check that all properties are present
        if not self.properties:
            self._properties = set(conformers.keys())
        elif not set(conformers.keys()) == self.properties:
            raise ValueError(f'Expected {self.properties} but got {set(conformers.keys())}')

        if '/' in group_name:
            raise ValueError('Character "/" not supported in group_name')

        # All properties must have the same batch dimension
        size = _get_num_conformers(conformers)
        if not all(conformers[k].shape[0] == size for k in self.properties):
            raise ValueError(f"All batch keys {self.properties} must have the same batch dimension")

        return group_name, conformers

    def _get_conformer_formulas(self, conformers: MaybeNumpyConformers) -> List[str]:
        if 'species' in conformers.keys():
            if isinstance(conformers['species'], Tensor):
                symbols = self._numbers_to_symbols(conformers['species'].detach().cpu().numpy())
            else:
                symbols = conformers['species']
        elif 'numbers' in conformers.keys():
            symbols = self._numbers_to_symbols(conformers['numbers'])
        else:
            raise ValueError("Either species or numbers must be present to parse formulas")

        return species_to_formula(symbols)

    @_needs_cache_update
    def delete_conformers(self, group_name: str, idx: Optional[Tensor] = None) -> '_ANISubdataset':
        r"""Delete a given set of conformers by passing their group name and
        indices within that group
        """
        if group_name not in self.keys():
            raise KeyError(group_name)
        if self.grouping not in ['by_formula', 'by_num_atoms']:
            raise ValueError("Can't delete if the grouping is not by_formula or"
                             " by_num_atoms, please regroup your dataset")
        all_conformers = self.get_numpy_conformers(group_name)
        with ExitStack() as stack:
            f = self._get_open_store(stack, 'r+')
            del f[group_name]
            # if no index was specified delete everything
            if idx is None:
                return self
            good_conformers = {k: np.delete(all_conformers[k], obj=idx.cpu().numpy(), axis=0)
                               for k in self.properties}
            if all(v.shape[0] == 0 for v in good_conformers.values()):
                # if we deleted everything in the group then just return,
                # otherwise we recreate the group using the good conformers
                return self
            group = f.create_conformer_group(group_name)
            group.create_numpy_values(good_conformers)
        return self

    @_needs_cache_update
    def create_species_from_numbers(self, source_key: str = 'numbers', dest_key: str = 'species') -> '_ANISubdataset':
        r"""Creates a 'species' property if a 'numbers' property exists"""
        self._check_properties_are_present(source_key)
        self._check_properties_are_not_present(dest_key)
        with ExitStack() as stack:
            f = self._get_open_store(stack, 'r+')
            for group_name in self.keys():
                symbols = self._numbers_to_symbols(f[group_name][source_key])
                f[group_name].create_numpy_values({dest_key: symbols})
        return self

    @_needs_cache_update
    def create_numbers_from_species(self, source_key: str = 'species', dest_key: str = 'numbers') -> '_ANISubdataset':
        r"""Creates a 'numbers' property if a 'species' property exists"""
        self._check_properties_are_present(source_key)
        self._check_properties_are_not_present(dest_key)
        with ExitStack() as stack:
            f = self._get_open_store(stack, 'r+')
            for group_name in self.keys():
                numbers = self._symbols_to_numbers(f[group_name][source_key].astype(str))
                f[group_name].create_numpy_values({dest_key: numbers})
        return self

    @_needs_cache_update
    def extract_slice_as_new_group(self,
                                   source_key: str,
                                   dest_key: str,
                                   idx_to_slice: int,
                                   dim_to_slice: int,
                                   squeeze_dest_key: bool = True) -> '_ANISubdataset':
        self._check_properties_are_present(source_key)
        self._check_properties_are_not_present(dest_key)
        # Annoyingly some properties are sometimes in this format:
        # "atomic_charges" with shape (C, A + 1), where charges[:, -1] is
        # actually the sum of the charges over all atoms.
        # This function solves the problem of dividing these properties as:
        # "atomic_charges (C, A + 1) -> "atomic_charges (C, A)", "charges (C,
        # )"
        with ExitStack() as stack:
            f = self._get_open_store(stack, 'r+')
            for group_name, conformers in self.numpy_items(properties='source_key'):
                to_slice = conformers[source_key]
                if to_slice.shape[dim_to_slice] <= 1:
                    raise ValueError("You can't slice the property if "
                                     "dim_to_slice has size 1 or smaller")
                # np.take automatically squeezes the output along the slice but
                # delete does not squeeze even if the resulting dim has size 1
                # so we sqeeze manually
                slice_ = np.take(to_slice, indices=idx_to_slice, axis=dim_to_slice)
                with_slice_deleted = np.delete(to_slice, obj=idx_to_slice, axis=dim_to_slice)
                if squeeze_dest_key:
                    with_slice_deleted = np.squeeze(with_slice_deleted, axis=dim_to_slice)

                del f[group_name][source_key]
                f[group_name].create_numpy_values({source_key: with_slice_deleted})
                f[group_name].create_numpy_values({dest_key: slice_})
        return self

    @_needs_cache_update
    def create_full_scalar_property(self,
                                    dest_key: str,
                                    fill_value: int = 0,
                                    dtype: DTypeLike = np.int64) -> '_ANISubdataset':
        r"""Creates one property with shape (num_conformers,), and with a
        specified dtype and value for all conformers in the dataset. Useful for
        creating 'charge' or 'spin_multiplicity' properties, which are usually
        the same for all conformers
        """
        self._check_properties_are_not_present(dest_key)
        with ExitStack() as stack:
            f = self._get_open_store(stack, 'r+')
            for group_name in self.keys():
                data = np.full(_get_num_conformers(f[group_name]), fill_value=fill_value, dtype=dtype)
                f[group_name].create_numpy_values({dest_key: data})
        return self

    def _make_empty_temporary_copy(self, grouping: Optional[str] = None) -> '_ANISubdataset':
        new_ds = _ANISubdataset(self._store_location.parent.joinpath('tmp_dataset.h5'),
                                create=True,
                                grouping=grouping if grouping is not None else self.grouping,
                                property_aliases=self._storename_to_alias,
                                assume_standard=self._has_standard_format,
                                verbose=False)
        return new_ds

    def _move_store_location_to_dataset(self, other_dataset: '_ANISubdataset') -> None:
        self._store_location.unlink()
        shutil.move(other_dataset._store_location.as_posix(), self._store_location.as_posix())
        other_dataset._store_location = self._store_location

    @_needs_cache_update
    def repack(self, verbose: bool = True) -> '_ANISubdataset':
        new_ds = self._make_empty_temporary_copy()
        for group_name, conformers in tqdm(self.numpy_items(),
                                           total=self.num_conformer_groups,
                                           desc='Repacking HDF5 file',
                                           disable=not verbose):
            new_ds.append_numpy_conformers.__wrapped__(new_ds, group_name, conformers)  # type: ignore
        self._move_store_location_to_dataset(new_ds)
        new_ds._verbose = self._verbose
        self = new_ds
        return self

    @_needs_cache_update
    def regroup_by_formula(self, repack: bool = True, verbose: bool = True) -> '_ANISubdataset':
        r"""Regroup dataset by formula (all conformers are extracted and
        redistributed in groups named 'C8H5N7', 'C10O3' etc, depending on the
        formula)
        """
        new_ds = self._make_empty_temporary_copy(grouping='by_formula')
        for group_name, conformers in tqdm(self.numpy_items(),
                                           total=self.num_conformer_groups,
                                           desc='Regrouping by formulas',
                                           disable=not verbose):
            # Get all formulas in the group to discriminate conformers by
            # formula and then attach conformers with the same formula to the
            # same groups
            formulas = np.asarray(self._get_conformer_formulas(conformers))
            unique_formulas = np.unique(formulas)
            formula_idxs = ((formulas == el).nonzero()[0] for el in unique_formulas)

            for formula, idx in zip(unique_formulas, formula_idxs):
                selected_conformers = {k: v[idx] for k, v in conformers.items()}
                new_ds.append_numpy_conformers.__wrapped__(new_ds, formula, selected_conformers)  # type: ignore
        self._move_store_location_to_dataset(new_ds)
        new_ds._verbose = self._verbose
        self = new_ds
        if repack:
            self._update_internal_cache()
            return self.repack.__wrapped__(self, verbose)
        return self

    @_needs_cache_update
    def regroup_by_num_atoms(self, repack: bool = True, verbose: bool = True) -> '_ANISubdataset':
        r"""Regroup dataset by number of atoms (all conformers are extracted
        and redistributed in groups named 'num_atoms_10', 'num_atoms_8' etc,
        depending on the number of atoms)
        """
        new_ds = self._make_empty_temporary_copy(grouping='by_num_atoms')
        for group_name, conformers in tqdm(self.numpy_items(),
                                           total=self.num_conformer_groups,
                                           desc='Regrouping by number of atoms',
                                           disable=not verbose):
            new_name = f'num_atoms_{_get_num_atoms(conformers)}'
            new_ds.append_numpy_conformers.__wrapped__(new_ds, new_name, conformers)  # type: ignore
        self._move_store_location_to_dataset(new_ds)
        new_ds._verbose = self._verbose
        self = new_ds
        if repack:
            self._update_internal_cache()
            return self.repack.__wrapped__(self, verbose)
        return self

    @_needs_cache_update
    def delete_properties(self, properties: Sequence[str], verbose: bool = True) -> '_ANISubdataset':
        r"""Delete some properties from the dataset"""
        self._check_properties_are_present(properties)
        with ExitStack() as stack:
            f = self._get_open_store(stack, 'r+')
            for group_key in tqdm(self.keys(),
                                  total=self.num_conformer_groups,
                                  desc='Deleting properties',
                                  disable=not verbose):
                for property_ in properties:
                    del f[group_key][property_]
                if not f[group_key].keys():
                    del f[group_key]
        return self

    @_needs_cache_update
    def rename_properties(self, old_new_dict: Dict[str, str]) -> '_ANISubdataset':
        r"""Rename some properties from the dataset, expects a dictionary of
        the form: {old_name: new_name}
        """
        # This can generate some counterintuitive results if the values are
        # aliases (renaming can be a no-op in this case) so we disallow it
        if set(old_new_dict.values()).issubset(set(self._storename_to_alias.keys())):
            raise ValueError("Cant rename to an alias")
        self._check_properties_are_present(old_new_dict.keys())
        self._check_properties_are_not_present(old_new_dict.values())
        with ExitStack() as stack:
            f = self._get_open_store(stack, 'r+')
            for k in self.keys():
                for old_name, new_name in old_new_dict.items():
                    f[k].move(old_name, new_name)
        return self

    @_needs_cache_update
    def set_aliases(self, property_aliases: Optional[Dict[str, str]] = None) -> '_ANISubdataset':
        r"""Set aliases for some properties from the dataset, expects a
        dictionary of the form: {old_name: new_name}. The properties are
        **not** renamed in the backing store, but the class will convert
        old_name to new_name internally when any method is called
        """
        self._storename_to_alias = dict() if property_aliases is None else property_aliases
        self._alias_to_storename = {v: k for k, v in self._storename_to_alias.items()}
        return self

    def _set_grouping(self, grouping: str) -> None:
        with ExitStack() as stack:
            f = self._get_open_store(stack, 'r+')
            f._set_grouping(grouping)

        if grouping in ['by_formula', 'by_num_atoms']:
            self._has_standard_format = True
            self._possible_nonbatch_properties = set()
        else:
            # other groupings are assumed to be "legacy", i.e. to have nonbatch
            # keys
            self._possible_nonbatch_properties = {'species', 'numbers'}

    @property
    def grouping(self) -> str:
        r"""Get the dataset grouping, one of 'by_formula', 'by_num_atoms' or an
        empty string for unspecified grouping
        """
        with ExitStack() as stack:
            # NOTE: r+ is needed due to HDF5 which disallows opening empty
            # files with r
            f = self._get_open_store(stack, 'r+')
            data = f.grouping
        return data

    def _check_properties_are_present(self, requested_properties: Iterable[str], raise_: bool = True) -> None:
        if isinstance(requested_properties, str):
            requested_properties = {requested_properties}
        else:
            requested_properties = set(requested_properties)
        if not requested_properties.issubset(self.properties):
            raise ValueError(f"Some of the properties requested {requested_properties} are not"
                             f" in the dataset, which has properties {self.properties}")

    def _check_properties_are_not_present(self, requested_properties: Iterable[str], raise_: bool = True) -> None:
        if isinstance(requested_properties, str):
            requested_properties = (requested_properties,)
        if set(requested_properties).issubset(self.properties):
            raise ValueError(f"Some of the properties requested {requested_properties} are"
                             f" in the dataset, which has properties {self.properties}, but they should not be")


class _DatasetStoreAdaptor(ContextManager['_DatasetStoreAdaptor'], Mapping[str, '_ConformerGroupAdaptor']):
    # wrapper around an open hdf5 file object that
    # returns ConformerGroup facades which renames properties on access and
    # creation
    def __init__(self, store_obj: H5File, alias_to_storename: Optional[Dict[str, str]] = None):
        # an open h5py file object
        self._store_obj = store_obj
        self._alias_to_storename = alias_to_storename if alias_to_storename is not None else dict()
        self.mode = self._store_obj.mode

    def _set_grouping(self, grouping: str) -> None:
        self._store_obj.attrs['grouping'] = grouping

    @property
    def grouping(self) -> str:
        try:
            data = self._store_obj.attrs['grouping']
        except (KeyError, OSError):
            data = ''
        assert isinstance(data, str)
        return data

    def __delitem__(self, k: str) -> None:
        del self._store_obj[k]

    def create_conformer_group(self, name) -> '_ConformerGroupAdaptor':
        # this wraps create_group
        self._store_obj.create_group(name)
        return self[name]

    def close(self) -> None:
        self._store_obj.close()

    def __enter__(self) -> '_DatasetStoreAdaptor':
        self._store_obj.__enter__()
        return self

    def __exit__(self, *args) -> None:
        self._store_obj.__exit__(*args)

    def __getitem__(self, name) -> '_ConformerGroupAdaptor':
        return _ConformerGroupAdaptor(self._store_obj[name], self._alias_to_storename)

    def __len__(self) -> int:
        return len(self._store_obj)

    def __iter__(self) -> Iterator[str]:
        return iter(self._store_obj)


class _ConformerGroupAdaptor(Mapping[str, np.ndarray]):
    def __init__(self, group_obj: H5Group, alias_to_storename: Optional[Dict[str, str]] = None):
        self._group_obj = group_obj
        self._alias_to_storename = alias_to_storename if alias_to_storename is not None else dict()

    def create_numpy_values(self, conformers: NumpyConformers) -> None:
        for p, v in conformers.items():
            self._create_property_with_data(p, v)

    def append_numpy_values(self, conformers: NumpyConformers) -> None:
        for p, v in conformers.items():
            self._append_property_with_data(p, v)

    def is_resizable(self) -> bool:
        return all(ds.maxshape[0] is None for ds in self._group_obj.values())

    def _append_property_with_data(self, p: str, data: np.ndarray) -> None:
        # resize and append to the dataset
        p = self._alias_to_storename.get(p, p)
        h5_dataset = self._group_obj[p]
        h5_dataset.resize(h5_dataset.shape[0] + data.shape[0], axis=0)
        try:
            h5_dataset[-data.shape[0]:] = data
        except TypeError:
            h5_dataset[-data.shape[0]:] = data.astype(bytes)

    def _create_property_with_data(self, p: str, data: np.ndarray) -> None:
        # this correctly handles strings (species and _id) and
        # key aliases
        p = self._alias_to_storename.get(p, p)

        # make the first axis resizable
        maxshape = (None,) + data.shape[1:]
        try:
            self._group_obj.create_dataset(name=p, data=data, maxshape=maxshape)
        except TypeError:
            self._group_obj.create_dataset(name=p, data=data.astype(bytes), maxshape=maxshape)

    def move(self, src: str, dest: str) -> None:
        src = self._alias_to_storename.get(src, src)
        dest = self._alias_to_storename.get(dest, dest)
        self._group_obj.move(src, dest)

    def __delitem__(self, k: str) -> None:
        del self._group_obj[k]

    def __getitem__(self, p: str) -> np.ndarray:
        p = self._alias_to_storename.get(p, p)
        array = self._group_obj[p][()]
        assert isinstance(array, np.ndarray)
        return array

    def __len__(self) -> int:
        return len(self._group_obj)

    def __iter__(self) -> Iterator[str]:
        for k in self._group_obj.keys():
            yield self._alias_to_storename.get(k, k)
