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
from contextlib import ExitStack, contextmanager
from collections import OrderedDict

import h5py
import torch
from torch import Tensor
import numpy as np

from ._annotations import Transform, Properties, NumpyProperties, MaybeNumpyProperties, PathLike, DTypeLike
from ..utils import species_to_formula, PERIODIC_TABLE, ATOMIC_NUMBERS, tqdm


DatasetWithFlag = Tuple['AniH5Dataset', bool]


def _may_need_cache_update(method: Callable[..., DatasetWithFlag]) -> Callable[..., 'AniH5Dataset']:
    # Decorator that wraps functions that modify the dataset in place.  Makes
    # sure that cache updating happens after dataset modification if needed

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


def _get_num_conformers(*args: Any, **kwargs: Any) -> int:
    # calculates number of conformers in some properties
    common_keys = {'coordinates', 'coord', 'energies', 'forces'}
    return _get_dim_size(common_keys, 0, *args, **kwargs)


def _get_num_atoms(*args: Any, **kwargs: Any) -> int:
    # calculates number of atoms in some properties
    common_keys = {'coordinates', 'coord', 'forces'}
    return _get_dim_size(common_keys, 1, *args, **kwargs)


def _get_dim_size(common_keys: Set[str], dim: int,
                  molecule_group: Union[h5py.Group, Properties, NumpyProperties],
                  flag_property: Optional[str] = None,
                  supported_properties: Optional[Set[str]] = None) -> int:
    # Calculates the dimension size in a set of properties
    if flag_property is not None:
        size = molecule_group[flag_property].shape[dim]
    else:
        assert supported_properties is not None
        present_keys = common_keys.intersection(supported_properties)
        if present_keys:
            size = molecule_group[tuple(present_keys)[0]].shape[dim]
        else:
            msg = (f'Could not infer dimension size of dim {dim} in properties '
                   f' since {common_keys} are missing')
            raise RuntimeError(msg)
    return size


class AniBatchedDataset(torch.utils.data.Dataset[Properties]):

    SUPPORTED_FILE_FORMATS = ('numpy', 'hdf5', 'pickle')
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

        # We use pickle or numpy or hdf5 since saving in
        # pytorch format is extremely slow
        def numpy_extractor(idx: int, paths: List[Path]) -> Properties:
            return {k: torch.as_tensor(v) for k, v in np.load(paths[idx]).items()}

        def pickle_extractor(idx: int, paths: List[Path]) -> Properties:
            with open(paths[idx], 'rb') as f:
                return {k: torch.as_tensor(v) for k, v in pickle.load(f).items()}

        def hdf5_extractor(idx: int, paths: List[Path]) -> Properties:
            with h5py.File(paths[idx], 'r') as f:
                return {k: torch.as_tensor(v[()]) for k, v in f['/'].items()}

        extractor_dict = {'numpy': partial(numpy_extractor, paths=self.batch_paths),
                          'pickle': partial(pickle_extractor, paths=self.batch_paths),
                          'hdf5': partial(hdf5_extractor, paths=self.batch_paths)}

        if file_format is None:
            format_suffix_map = {'.npz': 'numpy', '.pkl': 'pickle', '.h5': 'hdf5'}
            file_format = format_suffix_map[suffix]

        if file_format not in self.SUPPORTED_FILE_FORMATS:
            raise ValueError(f"The file format {file_format} is not in the"
                             f"supported formats {self.SUPPORTED_FILE_FORMATS}")

        self.extractor = extractor_dict[file_format]
        self._flag_property = flag_property

        try:
            with open(self.store_dir.parent.joinpath('creation_log.json'), 'r') as logfile:
                creation_log = json.load(logfile)
            self.is_inplace_transformed = creation_log['is_inplace_transformed']
            self.batch_size = creation_log['batch_size']
        except Exception:
            warnings.warn("No creation log found, is_inplace_transformed assumed False")
            self.is_inplace_transformed = False
            # All batches except the last are assumed to have the same length
            first_batch = self[-1]
            self.batch_size = _get_num_conformers(first_batch, self._flag_property, set(first_batch.keys()))

        if drop_last:
            # Drops last batch only if it is smallest than the rest
            last_batch = self[-1]
            last_batch_size = _get_num_conformers(last_batch, self._flag_property, set(last_batch.keys()))
            if last_batch_size < self.batch_size:
                self.batch_paths.pop()

        self._len = len(self.batch_paths)

    def cache(self,
              pin_memory: bool = True,
              verbose: bool = True,
              apply_transform: bool = True) -> 'AniBatchedDataset':
        r"""Save the complete dataset into RAM"""

        desc = f'Cacheing {self.split}, Warning: this may take a lot of RAM!'
        self._data = [self.extractor(idx) for idx in tqdm(range(len(self)),
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
            desc = 'Pinning memory, dont use pin_memory=True in the torch DataLoader'
            self._data = [{k: v.pin_memory()
                           for k, v in properties.items()}
                           for properties in tqdm(self._data,
                                                  total=len(self),
                                                  disable=not verbose,
                                                  desc=desc)]

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

    @contextmanager
    def keep_open(self, mode='r'):
        # Usage:
        # with ds.keep_open('r') as ro_ds:
        #     c = ro_ds.get_conformers('CH4')
        # etc
        with ExitStack() as stack:
            self._datasets = [stack.enter_context(d.keep_open(mode)) for d in self._datasets]
            yield self

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

        self.open_hdf5_file = None
        self._all_nonbatch_keys = set(nonbatch_keys)
        self._verbose = verbose
        self._store_file = Path(store_file).resolve()
        self._symbols_to_numbers = np.vectorize(lambda x: ATOMIC_NUMBERS[x])
        self._numbers_to_symbols = np.vectorize(lambda x: PERIODIC_TABLE[x])
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

    @contextmanager
    def keep_open(self, mode='r'):
        # Usage:
        # with ds.keep_open('r') as ro_ds:
        #     c = ro_ds.get_conformers('CH4')
        # etc
        # this speeds up access in the context of many operations in a block,
        # e.g.
        # with ds.keep_open('r') as ro_ds:
        #     for c in ro_ds.iter_conformers():
        #         print(c)
        # may be much faster than directly iterating over conformers
        self.open_hdf5_file = h5py.File(self._store_file, mode)
        try:
            yield self
        finally:
            self.open_hdf5_file.close()

    def _get_open_file(self, stack, mode: str = 'r'):
        # This trick makes methods fetch the open file directly
        # if they are being called from inside a "keep_open" context
        if self.open_hdf5_file is None:
            return stack.enter_context(h5py.File(self._store_file, mode))
        else:
            current_mode = self.open_hdf5_file.mode
            assert mode in ['r+', 'r'], f"Unsupported mode {mode}"
            if mode == 'r+' and current_mode == 'r':
                msg = ('Tried to open a file with mode "r+" but the dataset is '
                       'currently keeping its backend open with mode "r"')
                raise RuntimeError(msg)
            return self.open_hdf5_file

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
                with ExitStack() as stack:
                    f = self._get_open_file(stack, 'r')
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

            with ExitStack() as stack:
                f = self._get_open_file(stack, 'r')
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
        group_size = _get_num_conformers(molecule_group,
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

    def append_conformers(self,
                          group_name: str,
                          properties: Properties) -> 'AniH5Dataset':
        group_name, properties = self._check_append_input(group_name, properties)
        numpy_properties = {k: properties[k].numpy()
                            for k in self.supported_properties.difference({'species'})}

        # This correctly handles cases where species is a batch or nonbatch key
        if 'species' in self.supported_properties:
            if (properties['species'] <= 0).any():
                raise ValueError('Species are atomic numbers, must be positive')
            species = self._numbers_to_symbols(properties['species'].numpy())
            numpy_properties.update({'species': species})

        return self._append_numpy_conformers_no_check(group_name, numpy_properties)

    def append_numpy_conformers(self, group_name: str, properties: NumpyProperties) -> 'AniH5Dataset':
        group_name, properties = self._check_append_input(group_name, properties)
        return self._append_numpy_conformers_no_check(group_name, properties)

    def _check_append_input(self, group_name: str, properties: MaybeNumpyProperties) -> Tuple[str, MaybeNumpyProperties]:
        # check input kills first dimension of nonbatch keys
        if '/' in group_name:
            raise ValueError('Character "/" not supported in group_name')
        if not set(properties.keys()) == self.supported_properties:
            raise ValueError(f'Expected {self.supported_properties} but got {set(properties.keys())}')

        properties = deepcopy(properties)
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
    def _append_numpy_conformers_no_check(self, group_name: str, properties: NumpyProperties) -> DatasetWithFlag:
        # NOTE: Appending to datasets is allowed in HDF5 but only if
        # the dataset is created with "resizable" format, since this is not the
        # default  for simplicity we just rebuild the whole group with the new
        # properties appended.
        # NOTE: This function should never be called by user code since it
        # modifies properties in place.
        with ExitStack() as stack:
            f = self._get_open_file(stack, 'r+')
            try:
                f.create_group(group_name)
            except ValueError:
                old_properties = self.get_numpy_conformers(group_name, repeat_nonbatch_keys=False)
                if not all((old_properties == properties[k]).all() for k in self._supported_nonbatch_keys):
                    raise ValueError("Attempted to combine groups with different nonbatch key")
                properties.update({k: np.concatenate((old_properties[k], properties[k]), axis=0)
                                   for k in self._supported_batch_keys})
                del f[group_name]
                f.create_group(group_name)
            _create_numpy_properties_handle_str(f[group_name], properties)
        return self, True

    def delete_group(self, group_name: str) -> None:
        if group_name not in self.keys():
            raise KeyError(group_name)
        with ExitStack() as stack:
            f = self._get_open_file(stack, 'r+')
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

    @_may_need_cache_update
    def create_full_scalar_property(self,
                                    dest_key: str,
                                    fill_value: int = 0,
                                    strict: bool = False,
                                    dtype: DTypeLike = np.int64) -> DatasetWithFlag:
        if self._should_exit_early(dest_key=dest_key, strict=strict):
            return self, False
        with ExitStack() as stack:
            f = self._get_open_file(stack, 'r+')
            for group_name in self.keys():
                size = _get_num_conformers(f[group_name], self._flag_property, self.supported_properties)
                data = np.full(size, fill_value=fill_value, dtype=dtype)
                f[group_name].create_dataset(dest_key, data=data)
        return self, bool(self.keys())

    @_may_need_cache_update
    def create_species_from_numbers(self,
                                           source_key: str = 'numbers',
                                           dest_key: str = 'species',
                                           strict: bool = True) -> DatasetWithFlag:
        if self._should_exit_early(source_key, dest_key, strict):
            return self, False
        with ExitStack() as stack:
            f = self._get_open_file(stack, 'r+')
            for group_name in self.keys():
                symbols = np.asarray([PERIODIC_TABLE[j] for j in f[group_name][source_key][()]], dtype=str)
                f[group_name].create_dataset(dest_key, data=symbols.astype(bytes))
        return self, bool(self.keys())

    @_may_need_cache_update
    def create_numbers_from_species(self,
                                           source_key: str = 'species',
                                           dest_key: str = 'numbers',
                                           strict: bool = True) -> DatasetWithFlag:
        if self._should_exit_early(source_key, dest_key, strict):
            return self, False
        with ExitStack() as stack:
            f = self._get_open_file(stack, 'r+')
            for group_name in self.keys():
                numbers = self._symbols_to_numbers(f[group_name][source_key][()].astype(str))
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
        if self._should_exit_early(source_key, dest_key, strict):
            return self, False
        with ExitStack() as stack:
            f = self._get_open_file(stack, 'r+')
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

    def _should_exit_early(self,
                           source_key: Optional[str] = None,
                           dest_key: Optional[str] = None, strict: bool = False) -> bool:
        # Some functions need a source and/or a destination key to perform some
        # operations, this function checks that the source key exists and the
        # destination key does not, before performing the operation, if the
        # destination key already exists then the function should return early
        if source_key is not None and source_key not in self.supported_properties:
            raise ValueError(f"{source_key} is not in {self.supported_properties}")
        if dest_key is not None and dest_key in self.supported_properties:
            if not strict:
                return True
            raise ValueError(f"{dest_key} is already in {self.supported_properties}")
        return False

    @_may_need_cache_update
    def rename_groups_to_formulas(self, verbose: bool = True) -> DatasetWithFlag:
        if 'species' in self.supported_properties:
            parser = lambda s: species_to_formula(s['species'])  # noqa E731
        elif 'numbers' in self.supported_properties:
            parser = lambda s: species_to_formula(self._numbers_to_symbols(s['numbers']))  # noqa E731
        else:
            raise ValueError('"species" or "numbers" must be present to parse formulas')
        needs_cache_update = False
        for group_name, properties in tqdm(self.numpy_items(repeat_nonbatch_keys=False),
                                           total=self.num_conformer_groups,
                                           desc='Renaming groups to formulas',
                                           disable=not verbose):
            new_name = parser(properties)
            if group_name != f'/{new_name}':
                with ExitStack() as stack:
                    f = self._get_open_file(stack, 'r+')
                    del f[group_name]
                # mypy doesn't know that @wrap'ed functions have __wrapped__.
                # No need to check input here since it was inside the ds
                needs_cache_update = self._append_numpy_conformers_no_check.__wrapped__(self, new_name, properties)[1]  # type: ignore
        assert isinstance(needs_cache_update, bool)
        return self, needs_cache_update

    @_may_need_cache_update
    def delete_properties(self, properties: Sequence[str]) -> DatasetWithFlag:
        properties_set = {p for p in properties if p in self.supported_properties}

        if properties_set:
            with ExitStack() as stack:
                f = self._get_open_file(stack, 'r+')
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
                # Already renamed this property
                del old_new_dict[old]

        if old_new_dict:
            with ExitStack() as stack:
                f = self._get_open_file(stack, 'r+')
                for k in self.keys():
                    for old_name, new_name in old_new_dict.items():
                        f[k].move(old_name, new_name)
        return self, bool(old_new_dict)

    @_may_need_cache_update
    def delete_conformers(self, group_name: str, idx: Optional[Tensor] = None) -> DatasetWithFlag:

        all_conformers = self.get_numpy_conformers(group_name, repeat_nonbatch_keys=False)
        with ExitStack() as stack:
            f = self._get_open_file(stack, 'r+')
            del f[group_name]
            good_conformers = {k: np.delete(all_conformers[k], obj=idx, axis=0)
                               for k in self._supported_batch_keys}
            if all(v.shape[0] == 0 for v in good_conformers.values()):
                # if we deleted everything in the group then just return,
                # otherwise we recreate the group using the good conformers
                return self, True
            good_conformers.update({k: all_conformers[k]
                                    for k in self._supported_nonbatch_keys})
            f.create_group(group_name)
            _create_numpy_properties_handle_str(f[group_name], good_conformers)
        return self, True

    def present_species(self) -> Tuple[str, ...]:
        if 'species' in self.supported_properties:
            element_key = 'species'
            parser = lambda s: set(s['species'])  # noqa E731
        elif 'numbers' in self.supported_properties:
            element_key = 'numbers'
            parser = lambda s: set(self._numbers_to_symbols(s['numbers']))  # noqa E731
        else:
            raise ValueError('"species" or "numbers" must be present to parse symbols')
        present_species: Set[str] = set()
        for key in self.keys():
            species = self.get_numpy_conformers(key,
                                                include_properties=(element_key,),
                                                repeat_nonbatch_keys=False)
            present_species.update(parser(species))
        return tuple(sorted(present_species))

    def iter_conformers(self,
                        include_properties: Optional[Sequence[str]] = None) -> Iterator[Properties]:
        for _, _, c in self.iter_key_idx_conformers(include_properties):
            yield c

    def iter_key_idx_conformers(self,
                                include_properties: Optional[Sequence[str]] = None) -> Iterator[Tuple[str, int, Properties]]:
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
        # If the group or any of the properties does not exist this function
        # raises a KeyError through h5py This function behaves correctly if idx
        # has duplicated entries, is out of order, indexes out of bounds or is
        # empty, since it directly passes index tensors to numpy.
        # If the dataset has no supported properties this just returns and
        # empty dict
        nonbatch_keys, batch_keys = self._split_key_kinds(include_properties)
        all_keys = batch_keys.union(nonbatch_keys)
        with ExitStack() as stack:
            f = self._get_open_file(stack, 'r')
            numpy_properties = {p: f[key][p][()] for p in all_keys}
            if idx is not None:
                assert idx.dim() == 1, "index must be a dim 1 tensor"
                numpy_properties.update({k: numpy_properties[k][idx.cpu().numpy()] for k in batch_keys})

        if 'species' in all_keys:
            numpy_properties['species'] = numpy_properties['species'].astype(str)

        if repeat_nonbatch_keys:
            num_conformers = _get_num_conformers(numpy_properties, self._flag_property, batch_keys)
            numpy_properties.update({k: np.tile(numpy_properties[k], (num_conformers, 1))
                                     for k in nonbatch_keys})
        return numpy_properties

    def get_conformers(self,
                       key: str,
                       idx: Optional[Tensor] = None,
                       include_properties: Optional[Sequence[str]] = None,
                       repeat_nonbatch_keys: bool = True) -> Properties:
        # The tensor counterpart of get_numpy_conformers
        numpy_properties = self.get_numpy_conformers(key, idx, include_properties, repeat_nonbatch_keys)
        nonbatch_keys, batch_keys = self._split_key_kinds(include_properties)
        all_keys = nonbatch_keys.union(batch_keys)
        properties = {k: torch.tensor(numpy_properties[k]) for k in all_keys.difference({'species'})}

        # This correctly handles cases where species is a batch or nonbatch key
        if 'species' in all_keys:
            species = self._symbols_to_numbers(numpy_properties['species'])
            properties.update({'species': torch.from_numpy(species)})
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
        with ExitStack() as stack:
            f = self._get_open_file(stack, 'r')
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

        with ExitStack() as stack:
            f = self._get_open_file(stack, 'r+')
            f.attrs.create('functional', data=functional)
            f.attrs.create('basis_set', data=basis_set)
        for prefix in {'units', 'shape', 'dtype'}:
            attr_dict = {p: d[prefix] for p, d in metadata.items()}

            if prefix == 'shape':
                attr_dict = {k: repr(tuple_) for k, tuple_ in attr_dict.items()}
            elif prefix == 'dtype':
                attr_dict = {k: np.dtype(v).name for k, v in attr_dict.items()}

            with ExitStack() as stack:
                f = self._get_open_file(stack, 'r+')
                for p, u in attr_dict.items():
                    f.attrs.create(f"{prefix}.{p}", data=u)
        return self

    def clear_metadata(self) -> 'AniH5Dataset':
        with ExitStack() as stack:
            f = self._get_open_file(stack, 'r+')
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
        with ExitStack() as stack:
            f = self._get_open_file(stack, 'r+')
            attr_dict = {k.split('.')[1]: v for k, v in f.attrs.items() if k.split('.')[0] == prefix}
        return attr_dict
