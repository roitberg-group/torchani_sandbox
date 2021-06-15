from typing import (Union, Optional, Dict, Sequence, Iterator, Tuple, List, Set,
                    overload, Mapping, Any, Iterable, Callable)
import json
import re
import pickle
import warnings
import itertools
from copy import deepcopy
from pprint import pformat
from pathlib import Path
from functools import partial, wraps
from collections import OrderedDict

import h5py
import torch
from torch import Tensor
import numpy as np

from ._annotations import Transform, Properties, NumpyProperties, MaybeNumpyProperties, PathLike, DTypeLike
from ..utils import species_to_formula, ChemicalSymbolsToAtomicNumbers, PERIODIC_TABLE, tqdm


DatasetWithFlag = Tuple['AniH5Dataset', bool]


def _may_need_cache_update(method: Callable[..., DatasetWithFlag]) -> Callable[..., 'AniH5Dataset']:
    # Decorator that wraps functions that modify the dataset in place.
    # After dataset modification cache updating may be needed.

    @wraps(method)
    def method_with_cache_update(ds: 'AniH5Dataset', *args: Any, **kwargs: Any) -> 'AniH5Dataset':
        _, update_cache = method(ds, *args, **kwargs)
        if update_cache:
            ds._update_internal_cache()
        return ds

    return method_with_cache_update


def _create_numpy_properties_handle_str(group: h5py.Group, numpy_properties: NumpyProperties) -> None:
    # creates a dataset with dtype bytes if the array is a string array
    for k, v in numpy_properties.items():
        try:
            group.create_dataset(name=k, data=v)
        except TypeError:
            group.create_dataset(name=k, data=v.astype(bytes))


def _get_num_atoms(molecule_group: Union[h5py.Group, Properties, NumpyProperties],
                   flag_property: Optional[str] = None,
                   supported_properties: Optional[Set[str]] = None) -> int:
    if flag_property is not None:
        size = molecule_group[flag_property].shape[1]
    else:
        assert supported_properties is not None
        if 'coordinates' in supported_properties:
            size = molecule_group['coordinates'].shape[1]
        elif 'coord' in supported_properties:
            size = molecule_group['coord'].shape[1]
        elif 'forces' in supported_properties:
            size = molecule_group['forces'].shape[1]
        elif 'dipoles' in supported_properties:
            size = molecule_group['dipoles'].shape[1]
        else:
            msg = ('Could not infer number of atoms in properties since '
                    '"coordinates", "forces" and "dipoles" dont exist, please '
                    'provide a key that holds an array/tensor with number of '
                    'atoms in the axis/dim 1')
            raise RuntimeError(msg)
    return size


def _get_properties_size(molecule_group: Union[h5py.Group, Properties, NumpyProperties],
                        flag_property: Optional[str] = None,
                        supported_properties: Optional[Set[str]] = None) -> int:
    if flag_property is not None:
        size = molecule_group[flag_property].shape[0]
    else:
        assert supported_properties is not None
        if 'coordinates' in supported_properties:
            size = molecule_group['coordinates'].shape[0]
        elif 'coord' in supported_properties:
            size = molecule_group['coord'].shape[0]
        elif 'energies' in supported_properties:
            size = molecule_group['energies'].shape[0]
        elif 'forces' in supported_properties:
            size = molecule_group['forces'].shape[0]
        else:
            msg = ('Could not infer number of molecules in properties since '
                    '"coordinates", "forces" and "energies" dont exist, please '
                    'provide a key that holds an array/tensor with number of '
                    'molecules in the axis/dim 0')
            raise RuntimeError(msg)
    return size


class AniBatchedDataset(torch.utils.data.Dataset[Properties]):

    SUPPORTED_FILE_FORMATS = ('numpy', 'hdf5', 'single_hdf5', 'pickle')
    batch_size: int

    def __init__(self, store_dir: PathLike,
                       file_format: Optional[str] = None,
                       split: str = 'training',
                       transform: Transform = lambda x: x,
                       flag_property: Optional[str] = None,
                       drop_last: bool = False):

        self.split = split
        self.store_dir = Path(store_dir).resolve().joinpath(self.split)
        if not self.store_dir.is_dir():
            raise ValueError(f'The directory {self.store_dir.as_posix()} exists, '
                             f'but the split {split} could not be found')

        self.batch_paths = [f for f in self.store_dir.iterdir()]

        if not self.batch_paths:
            raise RuntimeError("The path provided has no files")
        if not all([f.is_file() for f in self.batch_paths]):
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
        if not all([f.suffix == suffix for f in self.batch_paths]):
            raise RuntimeError("Different file extensions were found in path, not supported")

        self.transform = transform

        def numpy_extractor(idx: int, paths: List[Path]) -> Properties:
            return {k: torch.as_tensor(v) for k, v in np.load(paths[idx]).items()}

        def pickle_extractor(idx: int, paths: List[Path]) -> Properties:
            with open(paths[idx], 'rb') as f:
                return {k: torch.as_tensor(v) for k, v in pickle.load(f).items()}

        def hdf5_extractor(idx: int, paths: List[Path]) -> Properties:
            with h5py.File(paths[idx], 'r') as f:
                return {k: torch.as_tensor(v[()]) for k, v in f['/'].items()}

        def single_hdf5_extractor(idx: int, group_keys: List[str], path: Path) -> Properties:
            k = group_keys[idx]
            with h5py.File(path, 'r') as f:
                return {k: torch.as_tensor(v[()]) for k, v in f[k].items()}

        # We use pickle or numpy or hdf5 since saving in
        # pytorch format is extremely slow
        if file_format is None:
            format_suffix_map = {'.npz': 'numpy', '.pkl': 'pickle', '.h5': 'hdf5'}
            file_format = format_suffix_map[suffix]
            if file_format == 'hdf5' and ('single' in self.batch_paths[0].name):
                file_format = 'single_hdf5'

        if file_format not in self.SUPPORTED_FILE_FORMATS:
            raise ValueError(f"The file format {file_format} is not in the"
                             f"supported formats {self.SUPPORTED_FILE_FORMATS}")

        if file_format == 'numpy':
            self.extractor = partial(numpy_extractor, paths=self.batch_paths)
        elif file_format == 'pickle':
            self.extractor = partial(pickle_extractor, paths=self.batch_paths)
        elif file_format == 'hdf5':
            self.extractor = partial(hdf5_extractor, paths=self.batch_paths)
        elif file_format == 'single_hdf5':
            warnings.warn('Depending on the implementation, a single HDF5 file '
                          'may not support parallel reads, so using num_workers > 1 '
                          'may have a detrimental effect on performance')
            with h5py.File(self.batch_paths[0], 'r') as f:
                keys = list(f.keys())
                self._len = len(keys)
                self.extractor = partial(single_hdf5_extractor, group_keys=keys, path=self.batch_paths[0])
        else:
            raise RuntimeError(f'Format for file with extension {suffix} '
                                'could not be inferred, please specify explicitly')

        self._flag_property = flag_property

        try:
            with open(self.store_dir.parent.joinpath('creation_log.json'), 'r') as logfile:
                creation_log = json.load(logfile)
            self.is_inplace_transformed = creation_log['is_inplace_transformed']
            self.batch_size = creation_log['batch_size']
        except Exception:
            warnings.warn("No creation log found, is_inplace_transformed assumed False")
            self.is_inplace_transformed = False

            # all batches except the last are assumed to have the same length
            first_batch = self[-1]
            self.batch_size = _get_properties_size(first_batch, self._flag_property, set(first_batch.keys()))

        if drop_last:
            # drops last batch only if it is smallest than the rest
            last_batch = self[-1]
            last_batch_size = _get_properties_size(last_batch, self._flag_property, set(last_batch.keys()))
            if last_batch_size < self.batch_size:
                self.batch_paths.pop()

        self._len = len(self.batch_paths)

    def cache(self,
              pin_memory: bool = True,
              verbose: bool = True,
              apply_transform: bool = True) -> 'AniBatchedDataset':
        if verbose:
            print(f"Cacheing split {self.split} of dataset, this may take some time...\n"
                   "Important: Cacheing the dataset may use a lot of memory, be careful!")

        self._data = [self.extractor(idx)
                      for idx in tqdm(range(len(self)),
                                      total=len(self),
                                      disable=not verbose,
                                      desc='Loading data into memory')]

        if apply_transform:
            if verbose:
                print("Important: Transformations, if there are any present,\n"
                      "will be applied once during cacheing and then discarded.\n"
                      "If you want a different behavior pass apply_transform=False")
            with torch.no_grad():
                self._data = [self.transform(properties)
                              for properties in tqdm(self._data,
                                                     total=len(self),
                                                     disable=not verbose,
                                                     desc="Applying transforms if present")]
            # discard transform after aplication
            self.transform = lambda x: x

        # When the dataset is cached memory pinning is done here. When the
        # dataset is not cached memory pinning is done by the torch DataLoader.
        if pin_memory:
            if verbose:
                print("Important: Cacheing pins memory automatically.\n"
                      "Do **not** use pin_memory=True in torch.utils.data.DataLoader")
            self._data = [{k: v.pin_memory()
                           for k, v in properties.items()}
                           for properties in tqdm(self._data,
                                                  total=len(self),
                                                  disable=not verbose,
                                                  desc='Pinning memory')]

        def memory_extractor(idx: int, ds: 'AniBatchedDataset') -> Properties:
            return ds._data[idx]

        self.extractor = partial(memory_extractor, ds=self)
        return self

    def __getitem__(self, idx: int) -> Properties:
        # integral indices must be provided for compatibility with pytorch
        # DataLoader API
        properties = self.extractor(idx)
        with torch.no_grad():
            properties = self.transform(properties)
        return properties

    def __iter__(self) -> Iterator[Properties]:
        j = 0
        try:
            while True:
                yield self[j]
                j += 1
        except IndexError:
            return

    def __len__(self) -> int:
        return self._len


class AniH5DatasetList(Sequence['AniH5Dataset']):

    # essentially a wrapper around a list of AniH5Dataset instances
    # to avoid boilerplate code to chain iterations over the datasets
    def __init__(self, dataset_paths: Sequence[PathLike], **h5_dataset_kwargs: Any):

        self._datasets = [AniH5Dataset(p, **h5_dataset_kwargs) for p in dataset_paths]
        self._dataset_paths = [Path(p).resolve() for p in dataset_paths]
        self.num_conformer_groups = sum(d.num_conformer_groups for d in self._datasets)
        self.num_conformers = sum(d.num_conformers for d in self._datasets)

    @overload
    def __getitem__(self, idx: int) -> 'AniH5Dataset':
        ...

    @overload
    def __getitem__(self, s: slice) -> Sequence['AniH5Dataset']:
        ...

    def __getitem__(self, idx: Union[int, slice]) -> Union['AniH5Dataset', Sequence['AniH5Dataset']]:
        return self._datasets[idx]

    def __len__(self) -> int:
        return len(self._datasets)

    def get_conformers(self, file_idx: int, *args: Any, **kwargs: Any) -> Properties:
        return self[file_idx].get_conformers(*args, **kwargs)

    def get_numpy_conformers(self, file_idx: int, *args: Any, **kwargs: Any) -> NumpyProperties:
        return self[file_idx].get_numpy_conformers(*args, **kwargs)

    def iter_file_key_idx_conformers(self, include_properties: Optional[Sequence[str]] = None,
                                yield_file_idx: bool = True) -> Iterator[Tuple[Union[int, Path], str, int, Properties]]:

        # chain yields key, idx, conformers
        k_i_c_chain = itertools.chain.from_iterable(d.iter_key_idx_conformers(include_properties)
                                              for d in self._datasets)
        repeats: Iterator[Union[int, Path]]
        if yield_file_idx:
            repeats = itertools.chain.from_iterable(itertools.repeat(j, d.num_conformers)
                                                    for j, d in enumerate(self._datasets))
        else:
            repeats = itertools.chain.from_iterable(itertools.repeat(self._dataset_paths[j], d.num_conformers)
                                                    for j, d in enumerate(self._datasets))
        yield from ((f, k, i, c) for f, (k, i, c) in zip(repeats, k_i_c_chain))

    def iter_conformers(self, include_properties: Optional[Sequence[str]] = None) -> Iterator[Properties]:
        for _, _, _, c in self.iter_file_key_idx_conformers(include_properties):
            yield c

    def iter_file_key(self) -> Iterator[Tuple[int, str]]:
        yield from ((j, k) for j, d in enumerate(self._datasets) for k in d.group_sizes.keys())


class AniH5Dataset(Mapping[str, Properties]):

    def __init__(self,
                 store_file: PathLike,
                 flag_property: Optional[str] = None,
                 nonbatch_keys: Sequence[str] = ('species', 'numbers'),
                 assume_standard: bool = False,
                 validate_metadata: bool = False,
                 create: bool = False,
                 supported_properties: Optional[Iterable[str]] = None,
                 verbose: bool = True):

        self._all_nonbatch_keys = set(nonbatch_keys)
        self._verbose = verbose
        self._store_file = Path(store_file).resolve()
        self._symbols_to_numbers = ChemicalSymbolsToAtomicNumbers()

        # flag property is used to infer size of molecule groups when iterating
        # over the dataset
        self._flag_property = flag_property

        # group_sizes, supported properties and num_conformers(_groups) are
        # needed as internal cache variables, this cache is updated if
        # something in the dataset changes, and when it is initialized
        self.group_sizes: 'OrderedDict[str, int]' = OrderedDict()
        self.num_conformers = 0
        self.num_conformer_groups = 0
        self.supported_properties: Set[str] = set()
        self._supported_batch_keys: Set[str] = set()

        # this variable only has an effect if gorup_kind == 'formula' or None
        self._supported_nonbatch_keys: Set[str] = set()

        if create:
            if supported_properties is None:
                raise ValueError("Please provide supported properties to create the dataset")
            open(self._store_file, 'x').close()
            self._has_homogeneous_properties = True
            self._has_standard_format = True
            self.supported_properties = set(supported_properties)
            self._update_private_sets()
        else:
            # In general all supported properties of the dataset should be equal
            # for all groups.
            if not self._store_file.is_file():
                raise FileNotFoundError(f"The h5 file in {self._store_file.as_posix()} could not be found")
            self._has_standard_format = assume_standard
            self._has_homogeneous_properties = False
            self._update_internal_cache()

        if validate_metadata:
            self.validate_metadata()

    def _reset_internal_cache(self) -> None:
        self.group_sizes = OrderedDict()
        self.num_conformers = 0
        self.num_conformer_groups = 0
        self.supported_properties = set()
        self._supported_nonbatch_keys = set()
        self._supported_batch_keys = set()

    def _update_internal_cache(self) -> None:
        self._reset_internal_cache()
        if self._has_standard_format:
            # This is much faster (x30) than a visitor function but it assumes
            # the format is somewhat standard which means that all Groups have
            # depth 1, and all Datasets have depth 2. "Meta" datasets don't
            # bother in this case
            with tqdm(desc=f'Scanning {self._store_file.name} assuming standard format',
                      disable=not self._verbose) as pbar:
                with h5py.File(self._store_file, 'r') as f:
                    for k, g in f.items():
                        pbar.update()
                        self._update_supported_properties_cache(g)
                        self._update_group_sizes_cache(g)
        else:
            def visitor_fn(name: str,
                           object_: Union[h5py.Dataset, h5py.Group],
                           dataset: 'AniH5Dataset',
                           pbar: Any) -> None:
                pbar.update()
                # We make sure the node is a Dataset, and We avoid Datasets
                # called _meta or _created since if present these store units
                # or other metadata. We also check if we already visited this
                # group via one of its children.
                if not isinstance(object_, h5py.Dataset) or\
                       object_.name.lower() in ['/_created', '/_meta'] or\
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
                dataset._update_supported_properties_cache(g)
                dataset._update_group_sizes_cache(g)

            with h5py.File(self._store_file, 'r') as f:
                with tqdm(desc='Verifying format correctness',
                          disable=not self._verbose) as pbar:
                    f.visititems(partial(visitor_fn, dataset=self, pbar=pbar))

            # If the visitor function succeeded and this condition is met the
            # dataset must be in standard format
            self._has_standard_format = not any(['/' in k[1:] for k in self.group_sizes.keys()])

        self._has_homogeneous_properties = True
        self._update_counters_cache()

        # By default iteration of HDF5 should be alphanumeric in which case
        # sorting should not be necessary, this internal assert ensures the
        # groups were not created with 'track_order=True', and that the visitor
        # function worked properly.
        assert list(self.group_sizes) == sorted(self.group_sizes), "Groups were not iterated upon alphanumerically"

    def _update_supported_properties_cache(self, molecule_group: h5py.Group) -> None:
        # Updates the "supported_properties", "_supported_nonbatch_keys" and
        # "_supported_batch_keys" variables. "element keys" are keys that
        # don't have a batch dimension, their shape must be (atoms,)
        if not self.supported_properties:
            self.supported_properties = set(molecule_group.keys())
            if self._flag_property is not None and self._flag_property not in self.supported_properties:
                msg = f"Flag property {self._flag_property} not found in {self.supported_properties}"
                raise RuntimeError(msg)
        elif not self._has_homogeneous_properties:
            found_properties = set(molecule_group.keys())
            if not found_properties == self.supported_properties:
                msg = (f"Group {molecule_group.name} has bad keys, "
                       f"found {found_properties}, but expected "
                       f"{self.supported_properties}")
                raise RuntimeError(msg)
        self._update_private_sets()

    def _update_private_sets(self) -> None:
        self._supported_nonbatch_keys = {k for k in self.supported_properties
                                        if k in self._all_nonbatch_keys}
        self._supported_batch_keys = {k for k in self.supported_properties
                                      if k not in self._all_nonbatch_keys}

    def _update_group_sizes_cache(self, molecule_group: h5py.Group) -> None:
        # updates "group_sizes" which holds the batch dimension (number of
        # molecules) of all grups in the dataset.
        group_size = _get_properties_size(molecule_group,
                                          self._flag_property,
                                          self.supported_properties)
        self.group_sizes.update({molecule_group.name[1:]: group_size})

    def _update_counters_cache(self) -> None:
        self.num_conformers = sum(self.group_sizes.values())
        self.num_conformer_groups = len(self.group_sizes.keys())

    def __getitem__(self, key: str) -> Properties:
        return self.get_conformers(key)

    def __len__(self) -> int:
        return self.num_conformer_groups

    def __iter__(self) -> Iterator[str]:
        return iter(self.group_sizes.keys())

    def _check_append_input(self, group_name: str, properties: MaybeNumpyProperties) -> Tuple[str, MaybeNumpyProperties]:
        if '/' in group_name:
            raise ValueError('Character "/" not supported in group_name')
        if not set(properties.keys()) == self.supported_properties:
            raise ValueError(f'Expected {self.supported_properties} but got {set(properties.keys())}')

        for k in self._supported_nonbatch_keys:
            if properties[k].ndim == 2:
                if not (properties[k] == properties[k][0]).all():
                    raise ValueError(f'All {k} must be equal in the batch dimension')
                else:
                    properties[k] = properties[k][0]
            if properties[k].ndim != 1:
                raise ValueError(f'{k} must have 1 or 2 dimensions')

        any_batch_key = tuple(self._supported_batch_keys)[0]
        for k in self._supported_batch_keys:
            if not properties[k].shape[0] == properties[any_batch_key].shape[0]:
                raise ValueError(f"All batch keys {self._supported_batch_keys} must have the same batch dimension")

        return group_name, properties

    @_may_need_cache_update
    def append_numpy_conformers(self,
                                group_name: str,
                                properties: NumpyProperties,
                                check_input: bool = True,
                                allow_arbitrary_keys: bool = False) -> DatasetWithFlag:
        if check_input:
            properties = deepcopy(properties)
            # After check input nonbatch keys are correctly turned into batch
            # keys
            group_name, properties = self._check_append_input(group_name, properties)

        if group_name is None:
            if 'species' in self._supported_nonbatch_keys:
                group_name = species_to_formula(properties['species'])
            else:
                raise ValueError("Cant determine default name for the group, species is missing")

        # NOTE: Appending to datasets is actually allowed in HDF5 but only if
        # the dataset is created with "resizable" format, since this is not the
        # default  for simplicity we just rebuild the whole group with the new
        # properties appended
        with h5py.File(self._store_file, 'r+') as f:
            try:
                f.create_group(group_name)
                group_exists = False
            except ValueError:
                group_exists = True
            if not group_exists:
                _create_numpy_properties_handle_str(f[group_name], properties)
            else:
                for k in self._supported_nonbatch_keys:
                    if not f[group_name][k] == properties[k]:
                        raise ValueError(f"Attempted to combine groups with different nonbatch key {k}")
                cated_properties = dict()
                for k, v in properties.items():
                    if k in self._supported_nonbatch_keys:
                        cat_value = v
                    elif k in self._supported_batch_keys:
                        new_properties = self.get_numpy_conformers(group_name, repeat_nonbatch_keys=False)
                        cat_value = np.concatenate((new_properties[k], v), axis=0)
                    cated_properties[k] = cat_value
                del f[group_name]
                f.create_group(group_name)
                _create_numpy_properties_handle_str(f[group_name], cated_properties)
        return self, True

    def append_conformers(self,
                          group_name: str,
                          properties: Properties,
                          allow_arbitrary_keys: bool = False) -> 'AniH5Dataset':
        properties = deepcopy(properties)
        group_name, properties = self._check_append_input(group_name, properties)
        if 'species' in self.supported_properties:
            if (properties['species'] <= 0).any():
                raise ValueError('Species are atomic numbers, must be positive')

        numpy_properties = {k: properties[k].numpy()
                            for k in self.supported_properties.difference({'species'})}

        if 'species' in self.supported_properties:
            species = properties['species']
            numpy_species = np.asarray([PERIODIC_TABLE[j] for j in species], dtype=str)
            numpy_properties.update({'species': numpy_species})
        return self.append_numpy_conformers(group_name, numpy_properties,
                                            check_input=False,
                                            allow_arbitrary_keys=allow_arbitrary_keys)

    def delete_group(self, group_name: str) -> None:
        if group_name not in self.keys():
            raise KeyError(group_name)
        with h5py.File(self._store_file, 'r+') as f:
            del f[group_name]
        self._update_internal_cache()

    def __str__(self) -> str:
        str_ = "ANI HDF5 Dataset object:\n"
        str_ += f"Number of conformers: {self.num_conformers}\n"
        str_ += f"Number of conformer groups: {self.num_conformer_groups}\n"
        try:
            str_ += f"Present elements: {self.present_species()}\n"
        except ValueError:
            str_ += "Present elements: Unknown\n"
        str_ += "Metadata: \n"
        str_ += pformat(self.get_metadata(), compact=True, width=200)
        return str_

    def numpy_items(self, *args: Any, **kwargs: Any) -> Iterator[Tuple[str, NumpyProperties]]:
        for group_name in self.keys():
            yield group_name, self.get_numpy_conformers(group_name, *args, **kwargs)

    def numpy_values(self, *args: Any, **kwargs: Any) -> Iterator[NumpyProperties]:
        for group_name in self.keys():
            yield self.get_numpy_conformers(group_name, *args, **kwargs)

    def _check_keys_already_present(self,
                                    source_key: Optional[str] = None,
                                    dest_key: Optional[str] = None, strict: bool = False) -> bool:
        # Some functions need a source and/or a destination key to perform some
        # operations, this function checks that the source key exists and the
        # destination key does not, before performing the operation, if the
        # destination key already exists then the function may return early so
        # and a fl
        if source_key is not None and source_key not in self.supported_properties:
            raise ValueError(f"{source_key} is not in {self.supported_properties}")
        if dest_key is not None and dest_key in self.supported_properties:
            if not strict:
                return True
            raise ValueError(f"{dest_key} is already in {self.supported_properties}")
        return False

    @_may_need_cache_update
    def create_full_scalar_property(self,
                                    dest_key: str,
                                    fill_value: int = 0,
                                    strict: bool = False,
                                    dtype: DTypeLike = np.int64) -> DatasetWithFlag:
        should_exit_early = self._check_keys_already_present(dest_key=dest_key, strict=strict)
        if should_exit_early:
            return self, False
        with h5py.File(self._store_file, 'r+') as f:
            for group_name in self.keys():
                size = _get_properties_size(f[group_name], self._flag_property, self.supported_properties)
                data = np.full(size, fill_value=fill_value, dtype=dtype)
                f[group_name].create_dataset(dest_key, data=data)
        return self, bool(self.keys())

    @_may_need_cache_update
    def create_species_from_numbers(self,
                                           source_key: str = 'numbers',
                                           dest_key: str = 'species',
                                           strict: bool = True) -> DatasetWithFlag:
        should_exit_early = self._check_keys_already_present(source_key, dest_key, strict)
        if should_exit_early:
            return self, False
        with h5py.File(self._store_file, 'r+') as f:
            for group_name in self.keys():
                symbols = np.asarray([PERIODIC_TABLE[j] for j in f[group_name][source_key][()]], dtype=str)
                f[group_name].create_dataset(dest_key, data=symbols.astype(bytes))
        return self, bool(self.keys())

    @_may_need_cache_update
    def create_numbers_from_species(self,
                                           source_key: str = 'species',
                                           dest_key: str = 'numbers',
                                           strict: bool = True) -> DatasetWithFlag:
        should_exit_early = self._check_keys_already_present(source_key, dest_key, strict)
        if should_exit_early:
            return self, False
        with h5py.File(self._store_file, 'r+') as f:
            for group_name in self.keys():
                numbers = self._symbols_to_numbers(f[group_name][source_key][()].astype(str).tolist()).numpy()
                f[group_name].create_dataset(dest_key, data=numbers)
        return self, bool(self.keys())

    @_may_need_cache_update
    def extract_slice_as_new_group(self,
                                   source_key: str,
                                   dest_key: str,
                                   idx_to_slice: int,
                                   dim_to_slice: int,
                                   squeeze_dest_key: bool = True,
                                   strict: bool = True) -> DatasetWithFlag:
        # Annoyingly some properties are sometimes in this format:
        # "atomic_charges" with shape (C, A + 1), where charges[:, -1] is
        # actually the sum of the charges over all atoms. This function solves
        # the problem of dividing these properties as:
        # "atomic_charges (C, A + 1) -> "atomic_charges (C, A)", "charges (C, )"
        should_exit_early = self._check_keys_already_present(source_key, dest_key, strict)
        if should_exit_early:
            return self, False
        with h5py.File(self._store_file, 'r+') as f:
            for group_name, properties in self.numpy_items(include_properties=('source_key',),
                                                           repeat_nonbatch_keys=False):
                to_slice = properties[source_key]
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
                f[group_name].create_dataset(source_key, data=with_slice_deleted)
                f[group_name].create_dataset(dest_key, data=slice_)
        return self, True

    @_may_need_cache_update
    def rename_groups_to_formulas(self, verbose: bool = True) -> DatasetWithFlag:
        if 'species' not in self.supported_properties:
            raise ValueError('Cant rename groups, species doesnt seem to be supported')
        needs_cache_update = False
        for group_name, properties in tqdm(self.numpy_items(repeat_nonbatch_keys=False),
                                  total=self.num_conformer_groups,
                                  desc='Renaming groups to formulas',
                                  disable=not verbose):
            new_name = species_to_formula(properties['species'])
            if group_name != f'/{new_name}':
                with h5py.File(self._store_file, 'r+') as f:
                    del f[group_name]
                # mypy doesn't know that @wrap'ed functions have these attributesj
                # and fixing this is ugly
                needs_cache_update = self.append_numpy_conformers.__wrapped__(self, new_name, properties)[1]  # type: ignore
        assert isinstance(needs_cache_update, bool)
        return self, needs_cache_update

    @_may_need_cache_update
    def delete_properties(self, properties: Sequence[str]) -> DatasetWithFlag:
        properties_set = {p for p in properties if p in self.supported_properties}

        if properties_set:
            with h5py.File(self._store_file, 'r+') as f:
                for group_key in self.keys():
                    for property_ in properties_set:
                        del f[group_key][property_]
                    if not f[group_key].keys():
                        del f[group_key]
        return self, bool(properties_set)

    @_may_need_cache_update
    def rename_properties(self, old_new_dict: Dict[str, str]) -> DatasetWithFlag:
        old_new_dict = old_new_dict.copy()
        for old, new in old_new_dict.copy().items():
            has_old = old in self.supported_properties
            has_new = new in self.supported_properties
            if has_old and has_new:
                raise ValueError(f'Cant rename {old} into {new} since {new} already exists')
            elif not has_old and not has_new:
                raise ValueError(f'Cant rename {old} into {new} since neither exists')
            elif not has_old and has_new:
                # Already renamed
                del old_new_dict[old]

        if old_new_dict:
            with h5py.File(self._store_file, 'r+') as f:
                for k in self.keys():
                    for old_name, new_name in old_new_dict.items():
                        f[k].move(old_name, new_name)
        return self, bool(old_new_dict)

    @_may_need_cache_update
    def delete_conformers(self, group_name: str, idx: Optional[Tensor] = None) -> DatasetWithFlag:

        all_conformers = self.get_numpy_conformers(group_name, repeat_nonbatch_keys=False)
        size = _get_properties_size(all_conformers, self._flag_property, self.supported_properties)

        # Get the indices that will remain after deletion, duplicated entries
        # of idx are ignored
        if idx is not None:
            assert idx.dim() == 1, "index must be a dim 1 tensor"
            assert idx.max() < size, "Out of bounds error"
            # If an empty idx is passed don't delete anything
            if idx.size(0) == 0:
                return self, False
            good_idxs = [i for i in range(size) if i not in idx]
        else:
            good_idxs = []

        # If there are any indices remaining we delete the dataset and recreate
        # it using the indices, otherwise we just delete the whole group
        with h5py.File(self._store_file, 'r+') as f:
            del f[group_name]
            if good_idxs:
                good_conformers = {k: all_conformers[k][good_idxs]
                                   for k in self._supported_batch_keys}
                good_conformers.update({k: all_conformers[k]
                                        for k in self._supported_nonbatch_keys})
                f.create_group(group_name)
                _create_numpy_properties_handle_str(f[group_name], good_conformers)
        return self, True

    def present_species(self) -> Tuple[str, ...]:
        present_species: Set[str] = set()
        if 'species' not in self.supported_properties:
            raise ValueError('Cant find present species, species doesnt seem to be supported')
        for key in self.keys():
            species = self.get_numpy_conformers(key,
                                                 include_properties=('species',),
                                                 repeat_nonbatch_keys=False)['species']
            present_species.update(set(species))
        return tuple(sorted(tuple(present_species)))

    def iter_conformers(self,
                        include_properties: Optional[Sequence[str]] = None) -> Iterator[Properties]:
        for _, _, c in self.iter_key_idx_conformers(include_properties):
            yield c

    def iter_key_idx_conformers(self,
                                include_properties: Optional[Sequence[str]] = None) -> Iterator[Tuple[str, int, Properties]]:
        # Iterate sequentially over conformers also copies all the group
        # into memory first, so it is fast
        for k, size in self.group_sizes.items():
            conformer_group = self.get_conformers(k, include_properties=include_properties)
            for idx in range(size):
                single_conformer = {k: conformer_group[k][idx]
                                    for k in conformer_group.keys()}
                yield k, idx, single_conformer

    def get_numpy_conformers(self,
                             key: str,
                             idx: Optional[Tensor] = None,
                             include_properties: Optional[Sequence[str]] = None,
                             repeat_nonbatch_keys: bool = True) -> NumpyProperties:
        # if the group or any of the properties does not exist this function
        # raises a KeyError through h5py This function behaves correctly if idx
        # has duplicated entries, is out of order, indexes out of bounds or is
        # empty, since it directly passes index tensors to numpy.
        # If the dataset has no supported properties this just returns and
        # empty dict

        nonbatch_keys, batch_keys = self._split_key_kinds(include_properties)
        all_keys = batch_keys.union(nonbatch_keys)

        with h5py.File(self._store_file, 'r') as f:
            numpy_properties = {p: f[key][p][()] for p in all_keys}
            if idx is not None:
                assert idx.dim() == 1, "index must be a dim 1 tensor"
                numpy_properties.update({k: numpy_properties[k][idx.cpu().numpy()] for k in batch_keys})

        if 'species' in all_keys:
            numpy_properties['species'] = numpy_properties['species'].astype(str)

        if repeat_nonbatch_keys:
            assert batch_keys, "Element keys can't be repeated since there are no batch keys"
            num_conformations = _get_properties_size(numpy_properties, self._flag_property, batch_keys)
            for k in nonbatch_keys:
                numpy_properties[k] = np.tile(numpy_properties[k], (num_conformations, 1))
        return numpy_properties

    def get_conformers(self,
                       key: str,
                       idx: Optional[Tensor] = None,
                       include_properties: Optional[Sequence[str]] = None,
                       repeat_nonbatch_keys: bool = True) -> Properties:
        # This function is the tensor counterpart of get_numpy_conformers
        numpy_properties = self.get_numpy_conformers(key, idx, include_properties, repeat_nonbatch_keys)
        nonbatch_keys, batch_keys = self._split_key_kinds(include_properties)
        all_keys = nonbatch_keys.union(batch_keys)
        properties = {k: torch.tensor(numpy_properties[k]) for k in all_keys.difference({'species'})}

        # 'species' gets special treatment since it has to be transformed to
        # atomic numbers in order to output a tensor.
        if 'species' in all_keys:
            species = numpy_properties['species']
            if repeat_nonbatch_keys:
                tensor_species = self._symbols_to_numbers(species[0].tolist())
                tensor_species = tensor_species.repeat(species.shape[0], 1)
            else:
                tensor_species = self._symbols_to_numbers(species.tolist())
            properties.update({'species': tensor_species})
        return properties

    def _split_key_kinds(self, properties: Optional[Sequence[str]] = None) -> Tuple[Set[str], Set[str]]:
        if properties is None:
            nonbatch_keys = self._supported_nonbatch_keys
            batch_keys = self._supported_batch_keys
        elif set(properties).issubset(self.supported_properties):
            nonbatch_keys = {k for k in properties if k in self._supported_nonbatch_keys}
            batch_keys = {k for k in properties if k not in self._supported_nonbatch_keys}
        else:
            raise ValueError(f"Some of the properties demanded {set(properties)} are not "
                             f"in the dataset, which has properties {self.supported_properties}")
        return nonbatch_keys, batch_keys

    # Metadata:
    #
    # shapes, dtypes, units, functional and basis set are metadata
    # shapes are stored as repr'd tuples, which can have
    # integers or strings, integers mean those axes are
    # constant and have to have the exact same size for all
    # properties, strings mean the axes are variables which can
    # have different values in different Groups, but the same
    # values should be maintained within a Group, (even in
    # different properties)
    def get_metadata(self) -> Dict[str, Any]:
        units = self._get_attr_dict('units')
        dtypes = self._get_attr_dict('dtype')
        shapes = self._get_attr_dict('shapes')
        metadata = {k: dict(units=units[k], dtype=dtypes[k], shape=shapes[k])
                    for k in units.keys()}
        with h5py.File(self._store_file, 'r') as f:
            metadata.update({'functional': f.attrs.get('functional'),
                             'basis_set': f.attrs.get('basis_set')})
        return metadata

    def set_metadata(self, metadata: Dict[str, Any]) -> 'AniH5Dataset':
        metadata = deepcopy(metadata)
        for k in metadata.keys():
            for c in {'*', '/', '.'}:
                assert c not in k, f"character {c} not supported"

        functional = metadata.pop('functional')
        basis_set = metadata.pop('basis_set')

        with h5py.File(self._store_file, 'r+') as f:
            f.attrs.create('functional', data=functional)
            f.attrs.create('basis_set', data=basis_set)
        for prefix in {'units', 'shape', 'dtype'}:
            attr_dict = {p: d[prefix] for p, d in metadata.items()}

            if prefix == 'shape':
                attr_dict = {k: repr(tuple_) for k, tuple_ in attr_dict.items()}
            elif prefix == 'dtype':
                attr_dict = {k: np.dtype(v).name for k, v in attr_dict.items()}

            with h5py.File(self._store_file, 'r+') as f:
                for p, u in attr_dict.items():
                    f.attrs.create(f"{prefix}.{p}", data=u)
        return self

    def clear_metadata(self) -> 'AniH5Dataset':
        with h5py.File(self._store_file, 'r+') as f:
            for k in f.attrs.keys():
                del f.attrs[k]
        return self

    def validate_metadata(self, verbose: bool = True) -> 'AniH5Dataset':
        # Metadata keys should be the same as the supported properties
        metadata = self.get_metadata()
        metadata.pop('functional')
        metadata.pop('basis_set')
        if not self.supported_properties == set(metadata.keys()):
            raise RuntimeError(f"Metadata has properties {set(metadata.keys())} "
                               f"but expected {self.supported_properties}")

        for group_name, properties in tqdm(self.numpy_items(repeat_nonbatch_keys=False),
                                  total=self.num_conformer_groups,
                                  desc='Validating metadata',
                                  disable=not verbose):

            variable_shapes: Dict[str, int] = dict()
            for property_, meta in metadata.items():
                # check dtype
                expected_dtype = meta['dtype']
                dtype = np.dtype(properties[property_].dtype).name
                if not dtype == expected_dtype and not expected_dtype == 'str':
                    raise RuntimeError(f'{property_} of {group_name} has dtype {dtype} '
                                       f'but expected {expected_dtype}')
                # check ndims
                expected_shape = eval(meta['shape'])
                shape = properties[property_].shape
                if not len(shape) == len(expected_shape):
                    raise RuntimeError(f'{property_} of {group_name} has {len(shape)} dims '
                                       f' but expected {len(expected_shape)}')
                # check shape
                for j, (s, size) in enumerate(zip(expected_shape, shape)):
                    if isinstance(s, int):
                        expected_size = s
                    elif isinstance(s, str) and s in variable_shapes.keys():
                        expected_size = variable_shapes[s]
                    else:
                        variable_shapes[s] = size
                    if not expected_size == size:
                        raise RuntimeError(f'{property_} of {group_name} has dim size {size} '
                                           f' but expected {expected_size} on dim {j}')
        return self

    def _get_attr_dict(self, prefix: str) -> Dict[str, str]:
        with h5py.File(self._store_file, 'r+') as f:
            attr_dict = {k.split('.')[1]: v for k, v in f.attrs.items() if k.split('.')[0] == prefix}
        return attr_dict
