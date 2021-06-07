import json
import re
import datetime
import math
import pickle
import warnings
import importlib
import itertools
from pathlib import Path
from functools import partial
from typing import Union, Optional, Dict, Sequence, Iterator, Tuple, List, Set, Callable, overload, Mapping, Any
from collections import OrderedDict

import h5py
import torch
from torch import Tensor
import numpy as np
from numpy import ndarray
from torchvision.datasets.utils import download_and_extract_archive, list_files, check_integrity
from torchani.utils import ChemicalSymbolsToAtomicNumbers, pad_atomic_properties, PADDING, cumsum_from_zero

# torch hub has a dummy implementation of tqdm which can be used if tqdm is not installed
try:
    from tqdm.auto import tqdm
    warnings.warn("tqdm could not be found, for better progress bars install tqdm")
except ImportError:
    from torch.hub import tqdm

PKBAR_INSTALLED = importlib.util.find_spec('pkbar') is not None
if PKBAR_INSTALLED:
    import pkbar

# type aliases
Transform = Callable[[Dict[str, Tensor]], Dict[str, Tensor]]
Properties = Dict[str, Tensor]
NumpyProperties = Dict[str, ndarray]
IdxType = Optional[Union[int, ndarray, Tensor]]


class AniBatchedDataset(torch.utils.data.Dataset[Properties]):

    SUPPORTED_FILE_FORMATS = ('numpy', 'hdf5', 'single_hdf5', 'pickle')
    batch_size: int

    def __init__(self, store_dir: Union[str, Path],
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

    def cache(self, pin_memory: bool = True,
                    verbose: bool = True,
                    apply_transform: bool = True) -> 'AniBatchedDataset':
        if verbose:
            print(f"Cacheing split {self.split} of dataset, this may take some time...")
            print("Important: Cacheing the dataset may use a lot of memory, be careful!")

        def memory_extractor(idx: int, ds: AniBatchedDataset) -> Properties:
            return ds._data[idx]

        self._data = [self.extractor(idx) for idx in range(len(self))]

        if apply_transform:
            if verbose:
                print("Applying transforms if they are present...")
                print("Important: Transformations, if there are any present,"
                      " will be applied once during cacheing and then discarded.")
                print("If you want a different behavior pass apply_transform=False")
            with torch.no_grad():
                self._data = [self.transform(properties) for properties in self._data]
            # discard transform after aplication
            self.transform = lambda x: x

        # When the dataset is cached memory pinning is done here. When the
        # dataset is not cached memory pinning is done by the torch DataLoader.
        if pin_memory:
            if verbose:
                print("Pinning memory...")
                print("Important: Cacheing pins memory automatically.")
                print("Do **not** use pin_memory=True in torch.utils.data.DataLoader")
            self._data = [{k: v.pin_memory() for k, v in properties.items()} for properties in self._data]

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
    def __init__(self, dataset_paths: Sequence[Union[str, Path]], **h5_dataset_kwargs: Any):

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
        return self._datasets[file_idx].get_conformers(*args, **kwargs)

    def get_numpy_conformers(self, file_idx: int, *args: Any, **kwargs: Any) -> NumpyProperties:
        return self._datasets[file_idx].get_numpy_conformers(*args, **kwargs)

    def iter_file_key_idx_conformers(self, include_properties: Optional[Sequence[str]] = None,
                                yield_file_idx: bool = True,
                                **get_group_kwargs: bool) -> Iterator[Tuple[Union[int, Path], str, int, Properties]]:

        # chain yields key, idx, conformers
        k_i_c_chain = itertools.chain.from_iterable(d.iter_key_idx_conformers(include_properties, **get_group_kwargs)
                                              for d in self._datasets)
        repeats: Iterator[Union[int, Path]]
        if yield_file_idx:
            repeats = itertools.chain.from_iterable(itertools.repeat(j, d.num_conformers)
                                                    for j, d in enumerate(self._datasets))
        else:
            repeats = itertools.chain.from_iterable(itertools.repeat(self._dataset_paths[j], d.num_conformers)
                                                    for j, d in enumerate(self._datasets))
        yield from ((f, k, i, c) for f, (k, i, c) in zip(repeats, k_i_c_chain))

    def iter_conformers(self, include_properties: Optional[Sequence[str]] = None,
                        **get_group_kwargs: bool) -> Iterator[Properties]:
        for _, _, _, c in self.iter_file_key_idx_conformers(include_properties, **get_group_kwargs):
            yield c

    def iter_file_key(self) -> Iterator[Tuple[int, str]]:
        yield from ((j, k) for j, d in enumerate(self._datasets) for k in d.group_sizes.keys())


_BASE_URL = 'http://moria.chem.ufl.edu/animodel/datasets/'


class _BaseBuiltinBatchedDataset(AniBatchedDataset):

    def __init__(self, root: Union[str, Path],
                       download: bool = False,
                       archive: Optional[str] = None,
                       md5: Optional[str] = None,
                       **batched_ds_kwargs):
        root = Path(root).resolve()
        self._archive: str = '' if archive is None else archive
        self._md5: str = '' if md5 is None else md5
        download_and_extract_archive(url=f'{_BASE_URL}{self._archive}', download_root=root, md5=self._md5)
        super().__init__(root, **batched_ds_kwargs)


class ANI1xBatched(_BaseBuiltinBatchedDataset):

    _ARCHIVES_AND_MD5S = {'train-valid': ('batched-ANI-1x-wB97X-6-31Gd-train-valid-2560.tar.gz', 'sdfsfd'),
                          '8-folds': ('batched-ANI-1x-wB97X-6-31Gd-8-folds-2560.tar.gz', 'sldkfsf'),
                          '5-folds': ('batched-ANI-1x-wB97X-6-31Gd-5-folds-2560.tar.gz', 'sdfdf')}

    def __init__(self, root: Union[str, Path], download: bool = False, kind='train-valid', **batched_ds_kwargs):
        if kind not in self._ARCHIVES.keys():
            raise ValueError(f"kind {kind} should be one of {list(self._ARCHIVES.keys())}")
        archive, md5 = self._ARCHIVES[kind]
        super().__init__(root, download, archive=archive, md5=md5, **batched_ds_kwargs)


class ANI2xBatched(_BaseBuiltinBatchedDataset):

    _ARCHIVES_AND_MD5S = {'train-valid': ('batched-ANI-2x-wB97X-6-31Gd-train-valid-2560.tar.gz', 'sdfsfd'),
                          '8-folds': ('batched-ANI-2x-wB97X-6-31Gd-8-folds-2560.tar.gz', 'sldkfsf'),
                          '5-folds': ('batched-ANI-2x-wB97X-6-31Gd-5-folds-2560.tar.gz', 'sdfdf')}

    def __init__(self, root: Union[str, Path], download: bool = False, kind='train-valid', **batched_ds_kwargs):
        if kind not in self._ARCHIVES.keys():
            raise ValueError(f"kind {kind} should be one of {list(self._ARCHIVES.keys())}")
        archive, md5 = self._ARCHIVES[kind]
        super().__init__(root, download, archive=archive, md5=md5, **batched_ds_kwargs)


class _BaseBuiltinRawDataset(AniH5DatasetList):
    # NOTE: Code heavily borrows from celeb dataset of torchvision

    def __init__(self, root: Union[str, Path],
                       download: bool = False,
                       archive: Optional[str] = None,
                       files_and_md5s: Optional[Dict[str, str]] = None,
                       **h5_dataset_list_kwargs):

        self._archive: str = '' if archive is None else archive
        self._files_and_md5s: Dict[str, str] = dict() if files_and_md5s is None else files_and_md5s

        root = Path(root).resolve()
        if download:
            if not self._maybe_download_hdf5_archive_and_check_integrity(root):
                raise RuntimeError('Dataset could not be download or is corrupted, '
                                   'please try downloading again')
        else:
            if not self._check_hdf5_files_integrity(root):
                raise RuntimeError('Dataset not found or is corrupted, '
                                   'you can use "download = True" to download it')

        dataset_paths = list_files(root, suffix='.h5', prefix=True)
        super().__init__(dataset_paths, flag_property='coordinates', element_keys=('species',), **h5_dataset_list_kwargs)

    def _check_hdf5_files_integrity(self, root: Union[str, Path]) -> bool:
        # Checks that all HDF5 files in the provided path are equal to the
        # expected ones and have the correct checksum, other files such as
        # tar.gz archives are neglected
        present_files = [Path(f).resolve() for f in list_files(root, suffix='.h5', prefix=True)]
        expected_file_names = set(self._files_and_md5s.keys())
        present_file_names = set([f.name for f in present_files])
        if expected_file_names != present_file_names:
            print(f"Wrong files found for dataset {self.__class__.__name__}, "
                  f"expected {expected_file_names} but found {present_file_names}")
            return False
        for f in tqdm(present_files, desc=f'Checking integrity of files for dataset {self.__class__.__name__}'):
            if not check_integrity(f, self._files_and_md5s[f.name]):
                print(f"All expected files for dataset {self.__class__.__name__} "
                      f"were found but file {f.name} failed integrity check")
                return False
        return True

    def _maybe_download_hdf5_archive_and_check_integrity(self, root: Union[str, Path]) -> bool:
        # Downloads only if the files have not been found or are corrupted
        root = Path(root).resolve()
        if root.is_dir() and self._check_hdf5_files_integrity(root):
            return True
        download_and_extract_archive(url=f'{_BASE_URL}{self._archive}', download_root=root, md5=None)
        return self._check_hdf5_files_integrity(root)


class RawANI1x(_BaseBuiltinRawDataset):
    _ARCHIVE = 'ANI-1x-wB97X-6-31Gd-data.tar.gz'
    _FILES_AND_MD5S = {'ANI-1x-wB97X-6-31Gd.h5': 'c9d63bdbf90d093db9741c94d9b20972'}

    def __init__(self, root: Union[str, Path], download: bool = False, **base_kwargs):
        super().__init__(root, download, archive=self._ARCHIVE, files_and_md5s=self._FILES_AND_MD5S, **base_kwargs)


class RawANI2x(_BaseBuiltinRawDataset):

    _ARCHIVE = 'ANI-2x-wB97X-6-31Gd-data.tar.gz'
    _FILES_AND_MD5S = {'ANI-1x-wB97X-6-31Gd.h5': 'c9d63bdbf90d093db9741c94d9b20972',
                     'ANI-2x-heavy-wB97X-6-31Gd.h5': '49ec3dc5d046f5718802f5d1f102391c',
                     'ANI-2x-dimers-wB97X-6-31Gd.h5': '3455d82a50c63c389126b68607fb9ca8'}

    def __init__(self, root: Union[str, Path], download: bool = False, **base_kwargs):
        super().__init__(root, download, archive=self._ARCHIVE, files_and_md5s=self._FILES_AND_MD5S, **base_kwargs)


class AniH5Dataset(Mapping[str, Properties]):

    def __init__(self,
                 store_file: Union[str, Path],
                 flag_property: Optional[str] = None,
                 element_keys: Sequence[str] = ('species', 'numbers', 'atomic_numbers'),
                 assume_standarized: bool = False):
        store_file = Path(store_file).resolve()
        if not store_file.is_file():
            raise FileNotFoundError(f"The h5 file in {store_file.as_posix()} could not be found")

        self._store_file = store_file

        # flag key is used to infer size of molecule groups
        # when iterating over the dataset
        self._flag_property = flag_property
        group_sizes, supported_properties = self._cache_group_sizes_and_properties(assume_standarized)
        self.group_sizes = OrderedDict(group_sizes)
        self.supported_properties = supported_properties

        # element keys are treated differently because they don't have a batch dimension
        self._supported_element_keys = tuple((k for k in self.supported_properties if k in element_keys))
        # smiles is not supported until we agree on a format for it, since it can have any strange format
        # in principle
        self._supported_non_element_keys = tuple((k for k in self.supported_properties
                                                  if k not in element_keys and k != 'smiles'))

        self.num_conformers = sum(self.group_sizes.values())
        self.num_conformer_groups = len(self.group_sizes.keys())

        self.symbols_to_atomic_numbers = ChemicalSymbolsToAtomicNumbers()

    def __getitem__(self, key: str) -> Properties:
        # this is a simple extraction that just fetches everything
        # and always return a Dict[str, Tensor] since numpy_output is set to False
        properties = self._get_group(key, self._supported_non_element_keys,
                                          self._supported_element_keys,
                                          numpy_output=False,
                                          idx=None,
                                          repeat_element_keys=True)
        return properties

    def __len__(self) -> int:
        return self.num_conformer_groups

    def __iter__(self) -> Iterator[str]:
        # Iterating over groups and yield the associated molecule groups as
        # dictionaries of numpy arrays (except for species, which is a list of
        # strings)
        return iter(self.group_sizes.keys())

    def get_conformers(self,
                       key: str,
                       idx: IdxType = None,
                       include_properties: Optional[Sequence[str]] = None,
                       **kwargs: bool) -> Properties:
        # kwargs are flags for _get_group (numpy_output, repeat_element_keys and strict)
        element_keys, non_element_keys = self._properties_into_keys(include_properties)
        # fetching a conformer actually copies all the group into memory first,
        # because indexing directly into hdf5 is much slower.
        return self._get_group(key, non_element_keys, element_keys, idx=idx, **kwargs)

    def get_numpy_conformers(self,
                       key: str,
                       idx: IdxType = None,
                       include_properties: Optional[Sequence[str]] = None,
                       **kwargs: bool) -> NumpyProperties:
        element_keys, non_element_keys = self._properties_into_keys(include_properties)
        return self._get_numpy_group(key, non_element_keys, element_keys, idx=idx, **kwargs)

    def delete_conformers_inplace(self, key: str, idx: IdxType = None) -> 'AniH5Dataset':
        # first we fetch all conformers from the group
        all_conformers = self.get_numpy_conformers(key,
                                             include_properties=tuple(self.supported_properties),
                                             repeat_element_keys=False)
        size = _get_properties_size(all_conformers, self._flag_property, self.supported_properties)
        all_idxs = torch.arange(size)

        if idx is not None:
            if not isinstance(idx, Tensor):
                idx = torch.tensor(idx)
            assert isinstance(idx, Tensor)
            if idx.dim() == 0:
                idx = idx.unsqueeze(0)
            good_idxs = torch.stack([all_idxs != i for i in idx.cpu()], dim=0).all(dim=0).nonzero().squeeze()
        else:
            good_idxs = torch.empty(0)

        # if there are any good conformations remaining we delete the dataset and
        # recreate it using the good conformations, otherwise we just delete the whole group
        if good_idxs.numel() > 0:
            # sanity check
            if good_idxs.dim() == 0:
                good_idxs = torch.tensor([good_idxs.item()])
            assert len(good_idxs) <= len(all_idxs)
            good_idxs = good_idxs.cpu().numpy()
            good_conformers = {k: all_conformers[k][good_idxs] for k in self._supported_non_element_keys}
            good_conformers.update({k: all_conformers[k] for k in self._supported_element_keys})
            with h5py.File(self._store_file, 'r+') as f:
                del f[key]
                f.create_group(key)
                group = f[key]
                for k, v in good_conformers.items():
                    try:
                        group.create_dataset(name=k, data=v)
                    except TypeError:
                        group.create_dataset(name=k, data=v.astype(bytes))
        else:
            with h5py.File(self._store_file, 'r+') as f:
                del f[key]
        return self

    def iter_conformers(self,
                        include_properties: Optional[Sequence[str]] = None,
                        **get_group_kwargs: bool) -> Iterator[Properties]:
        for _, _, c in self.iter_key_idx_conformers(include_properties, **get_group_kwargs):
            yield c

    def iter_key_idx_conformers(self,
                                include_properties: Optional[Sequence[str]] = None,
                                **get_group_kwargs: bool) -> Iterator[Tuple[str, int, Properties]]:
        if 'repeat_element_keys' in get_group_kwargs.keys():
            raise ValueError("Repeat element keys not supported")
        element_keys, non_element_keys = self._properties_into_keys(include_properties)
        # Iterate sequentially over conformers also copies all the group
        # into memory first, so it is also fast
        for k, size in self.group_sizes.items():
            conformer_group = self._get_group(k, non_element_keys, element_keys, idx=None, **get_group_kwargs)
            for idx in range(size):
                single_conformer = {k: conformer_group[k][idx]
                                    for k in itertools.chain(element_keys, non_element_keys)}
                yield k, idx, single_conformer

    def _properties_into_keys(self,
                              properties: Optional[Sequence[str]] = None) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
        if properties is None:
            element_keys = self._supported_element_keys
            non_element_keys = self._supported_non_element_keys
        elif set(properties).issubset(self.supported_properties):
            element_keys = tuple((k for k in properties if k in self._supported_element_keys))
            non_element_keys = tuple((k for k in properties if k not in self._supported_element_keys))
        else:
            raise ValueError(f"Some of the properties demanded {properties} are not "
                             f"in the dataset, which has properties {self.supported_properties}")
        return element_keys, non_element_keys

    def _cache_group_sizes_and_properties(self, assume_standarized: bool = False) -> Tuple[List[Tuple[str, int]], Set[str]]:
        # cache paths of all molecule groups into a list
        # and all supported properties into a set

        def visitor_fn(name: str,
                       object_: Union[h5py.Dataset, h5py.Group],
                       group_sizes: List[Tuple[str, int]],
                       supported_properties: Set[str],
                       flag_property: Optional[str] = None,
                       pbar=None) -> None:
            if pbar is not None:
                pbar.update()
            if isinstance(object_, h5py.Dataset):
                molecule_group = object_.parent
                # Check if we already visited this group via one of its
                # children or not
                if molecule_group.name not in [tup[0] for tup in group_sizes]:
                    # Collect properties and check that all the datasets have
                    # the same properties
                    if not supported_properties:
                        supported_properties.update(set(molecule_group.keys()))
                        if flag_property is not None and flag_property not in supported_properties:
                            raise RuntimeError(f"Flag property {flag_property} "
                                               f"not found in {supported_properties}")
                    else:
                        if not set(molecule_group.keys()) == supported_properties:
                            raise RuntimeError(f"group {molecule_group.name} has incompatible keys, "
                                               f"which should be {supported_properties}, inferred from other groups")
                    # Check for format correctness
                    for v in molecule_group.values():
                        if not isinstance(v, h5py.Dataset):
                            raise RuntimeError("Invalid dataset format, there "
                                               "shouldn't be Groups inside Groups "
                                               "that have Datasets")
                    group_sizes.append((molecule_group.name,
                                         _get_properties_size(molecule_group,
                                                              flag_property,
                                                              supported_properties)))

        group_sizes: List[Tuple[str, int]] = []
        supported_properties: Set[str] = set()

        if assume_standarized:
            # This is much faster (x30) than a visitor function but it assumes
            # the format is somewhat standarized which means that all Groups
            # have depth 1, with the same number and name of Datasets each
            # (properties such as species, coordinates, etc), and all Datasets
            # have depth 2
            with tqdm(desc='Scanning HDF5 file assuming standard format') as pbar:
                with h5py.File(self._store_file, 'r') as f:
                    for j, (k, g) in enumerate(f.items()):
                        pbar.update()
                        if j == 0:
                            supported_properties = set(g.keys())
                            if self._flag_property is not None and self._flag_property not in supported_properties:
                                raise RuntimeError(f"Flag property {self._flag_property} "
                                                   f"not found in {supported_properties}")
                        group_sizes.append((k, _get_properties_size(g, self._flag_property, supported_properties)))
        else:
            with h5py.File(self._store_file, 'r') as f:
                with tqdm(desc='Scanning HDF5 file and verifying format correctness') as pbar:
                    f.visititems(partial(visitor_fn,
                                         pbar=pbar,
                                         group_sizes=group_sizes,
                                         supported_properties=supported_properties,
                                         flag_property=self._flag_property))
        # By default iteration of HDF5 should be alphanumeric in which case
        # sorting should not be necessary, this internal assert ensures the
        # groups were not created with 'track_order=True', and that the visitor
        # function worked properly, which could make iteration non-alphanumeric
        # and would be surprising for a user.
        assert group_sizes == sorted(group_sizes), "Groups were not iterated upon alphanumerically"
        return group_sizes, supported_properties

    def _get_numpy_group(self,
                         key: str,
                         non_element_keys: Tuple[str, ...],
                         element_keys: Tuple[str, ...],
                         idx: IdxType = None,
                         strict: bool = False,
                         numpy_output: bool = False,
                         repeat_element_keys: bool = True) -> NumpyProperties:
        # This function is essentially an internal implementation of
        # __getitem__ and get_conformers, it is not meant to be called directly
        # by user code

        # We allow Tensor as an index but it is internally converted to a numpy
        # array since a numpy array is needed for indexing HDF5 datasets
        if isinstance(idx, Tensor):
            idx = idx.cpu().numpy()
            assert isinstance(idx, ndarray)

        # The index being an int is a special case since it gets rid of a
        # dimension, this is handled automatically, but we don't need to
        # repeat element keys in this case.
        get_single_conformer = isinstance(idx, int)

        # NOTE: If some keys are not found then
        # this returns a partial result with the keys that are found, (maybe
        # even empty) unless strict is passed.
        with h5py.File(self._store_file, 'r') as f:
            group = f[key]
            if strict and not all([p in group.keys() for p in element_keys + non_element_keys]):
                raise RuntimeError('Some of the requested properties could not '
                                  f'be found in group {key}')

            numpy_properties: NumpyProperties = {k: np.copy(group[k]) for k in element_keys}
            if idx is None:
                numpy_properties.update({k: np.copy(group[k]) for k in non_element_keys})
            else:
                numpy_properties.update({k: np.copy(group[k])[idx] for k in non_element_keys})

        if 'species' in element_keys:
            numpy_properties['species'] = numpy_properties['species'].astype(str)

        if repeat_element_keys and not get_single_conformer:
            # here we use any of the non element keys as a flag property
            num_conformations = _get_properties_size(numpy_properties, None, set(non_element_keys))
            for k in element_keys:
                numpy_properties[k] = np.tile(numpy_properties[k], (num_conformations, 1))

        # If we want the raw values of the h5 dataset we return early
        return numpy_properties

    def _get_group(self,
                   key: str,
                   non_element_keys: Tuple[str, ...],
                   element_keys: Tuple[str, ...],
                   **kwargs: Any) -> Properties:
        # This function is the tensor counterpart of _get_numpy_group
        numpy_properties = self._get_numpy_group(key, non_element_keys, element_keys, **kwargs)

        # Here we do some more processing to return tensors including
        # converting species to atomic numbers. NOTE: All non-element-key properties
        # are assumed to be floats
        properties: Properties = {k: torch.tensor(numpy_properties[k], dtype=torch.float)
                                  for k in non_element_keys}
        properties.update({k: torch.tensor(numpy_properties[k], dtype=torch.long)
                           for k in element_keys if k != 'species'})

        # 'species' gets special treatment since it has to be transformed to
        # atomic numbers in order to output a tensor, and we need a flattened
        # list to convert to atomic numbers
        if 'species' in element_keys:
            tensor_species: Tensor
            species = numpy_properties['species']
            if species.ndim == 2:
                num_molecules = species.shape[0]
                num_atoms = species.shape[1]
                tensor_species = self.symbols_to_atomic_numbers(species.ravel().tolist())
                tensor_species = tensor_species.view(num_molecules, num_atoms)
            else:
                tensor_species = self.symbols_to_atomic_numbers(species.tolist())
            properties.update({'species': tensor_species})
        return properties


def _save_batch(path: Path, idx: int, batch: Properties, file_format: str) -> None:
    # We use pickle, numpy or hdf5 since saving in
    # pytorch format is extremely slow
    batch = {k: v.numpy() for k, v in batch.items()}
    if file_format == 'pickle':
        with open(path.joinpath(f'batch{idx}.pkl'), 'wb') as batch_file:
            pickle.dump(batch, batch_file)
    elif file_format == 'numpy':
        np.savez(path.joinpath(f'batch{idx}'), **batch)
    elif file_format == 'hdf5':
        with h5py.File(path.joinpath(f'batch{idx}.h5'), 'w-') as f:
            for k, v in batch.items():
                f.create_dataset(k, data=v)
    elif file_format == 'single_hdf5':
        with h5py.File(path.joinpath(f'{path.name}_single.h5'), 'a') as f:
            f.create_group(f'batch{idx}')
            g = f[f'batch{idx}']
            for k, v in batch.items():
                g.create_dataset(k, data=v)


def create_batched_dataset(h5_path: Union[str, Path],
                           dest_path: Optional[Union[str, Path]] = None,
                           shuffle: bool = True,
                           shuffle_seed: Optional[int] = None,
                           file_format: str = 'hdf5',
                           include_properties: Optional[Sequence[str]] = ('species', 'coordinates', 'energies'),
                           batch_size: int = 2560,
                           max_batches_per_packet: int = 350,
                           padding: Optional[Dict[str, float]] = None,
                           splits: Optional[Dict[str, float]] = None,
                           folds: Optional[int] = None,
                           inplace_transform: Optional[Transform] = None,
                           verbose: bool = True) -> None:

    if folds is not None and splits is not None:
        raise ValueError('Only one of ["folds", "splits"] should be specified')

    # NOTE: All the tensor manipulation in this function is handled in CPU
    if file_format == 'single_hdf5':
        warnings.warn('Depending on the implementation, a single HDF5 file may'
                      'not support parallel reads, so using num_workers > 1 may'
                      'have a detrimental effect on performance, its probably better'
                      'to save in many hdf5 files with file_format=hdf5')
    if dest_path is None:
        dest_path = Path(f'./batched_dataset_{file_format}').resolve()
    dest_path = Path(dest_path).resolve()

    h5_path = Path(h5_path).resolve()
    if h5_path.is_dir():
        # sort paths according to file names for reproducibility
        paths_list = [p for p in h5_path.iterdir() if p.suffix == '.h5']
        filenames_list = [p.name for p in paths_list]
        sorted_paths = [p for _, p in sorted(zip(filenames_list, paths_list))]
        h5_datasets = AniH5DatasetList(sorted_paths)
    elif h5_path.is_file():
        h5_datasets = AniH5DatasetList([h5_path])

    # (1) Get all indices and shuffle them if needed
    #
    # These are pairs of indices that index first the group and then the
    # specific conformer, it is possible to just use one index for
    # everything but this is simpler at the cost of slightly more memory.
    # First we get all group sizes for all datasets concatenated in a tensor, in the same
    # order as h5_datasets
    group_sizes_values = torch.cat([torch.tensor(list(h5ds.group_sizes.values()), dtype=torch.long) for h5ds in h5_datasets])
    conformer_indices = torch.cat([torch.stack((torch.full(size=(s.item(),), fill_value=j, dtype=torch.long),
                                     (torch.arange(0, s.item(), dtype=torch.long))), dim=-1)
                                     for j, s in enumerate(group_sizes_values)])

    rng = _get_random_generator(shuffle, shuffle_seed)

    conformer_indices = _maybe_shuffle_indices(conformer_indices, rng)

    # (2) Split shuffled indices according to requested dataset splits or folds
    # by defaults we use splits, if folds or splits is specified we
    # do the specified operation
    if folds is not None:
        conformer_splits, split_paths = _divide_into_folds(conformer_indices, dest_path, folds, rng)
    else:
        if splits is None:
            splits = {'training': 0.8, 'validation': 0.2}

        if not math.isclose(sum(list(splits.values())), 1.0):
            raise ValueError("The sum of the split fractions has to add up to one")

        conformer_splits, split_paths = _divide_into_splits(conformer_indices, dest_path, splits)

    # (3) Compute the batch indices for each split and save the conformers to disk
    _save_splits_into_batches(split_paths,
                              conformer_splits,
                              inplace_transform,
                              file_format,
                              include_properties,
                              h5_datasets,
                              padding,
                              batch_size,
                              max_batches_per_packet,
                              verbose)

    # log creation data
    creation_log = {'datetime_created': str(datetime.datetime.now()),
                    'source_path': h5_path.as_posix(),
                    'splits': splits,
                    'folds': folds,
                    'padding': PADDING if padding is None else padding,
                    'is_inplace_transformed': inplace_transform is not None,
                    'shuffle': shuffle,
                    'shuffle_seed': shuffle_seed,
                    'include_properties': include_properties if include_properties is not None else 'all',
                    'batch_size': batch_size,
                    'total_num_conformers': len(conformer_indices),
                    'total_conformer_groups': len(group_sizes_values)}

    with open(dest_path.joinpath('creation_log.json'), 'w') as logfile:
        json.dump(creation_log, logfile, indent=1)


def _get_random_generator(shuffle: bool = False, shuffle_seed: Optional[int] = None) -> Optional[torch.Generator]:

    if shuffle_seed is not None:
        assert shuffle
        seed = shuffle_seed
    else:
        # non deterministic seed
        seed = torch.random.seed()

    rng: Optional[torch.Generator]

    if shuffle:
        rng = torch.random.manual_seed(seed)
    else:
        rng = None

    return rng


def _get_properties_size(molecule_group: Union[h5py.Group, Properties, NumpyProperties],
                        flag_property: Optional[str] = None,
                        supported_properties: Optional[Set[str]] = None) -> int:
    if flag_property is not None:
        size = molecule_group[flag_property].shape[0]
    else:
        assert supported_properties is not None
        if 'coordinates' in supported_properties:
            size = molecule_group['coordinates'].shape[0]
        elif 'energies' in supported_properties:
            size = molecule_group['energies'].shape[0]
        elif 'forces' in supported_properties:
            size = molecule_group['forces'].shape[0]
        else:
            raise RuntimeError('Could not infer number of molecules in properties'
                               ' since "coordinates", "forces" and "energies" dont'
                               ' exist, please provide a key that holds an array/tensor with the'
                               ' molecule size as its first axis/dim')
    return size


def _maybe_shuffle_indices(conformer_indices: Tensor,
                           rng: Optional[torch.Generator] = None) -> Tensor:
    total_num_conformers = len(conformer_indices)
    if rng is not None:
        shuffle_indices = torch.randperm(total_num_conformers, generator=rng)
        conformer_indices = conformer_indices[shuffle_indices]
    else:
        warnings.warn("Dataset will not be shuffled, this should only be used for debugging")
    return conformer_indices


def _divide_into_folds(conformer_indices: Tensor,
                        dest_path: Path,
                        folds: int,
                        rng: Optional[torch.Generator] = None) -> Tuple[Tuple[Tensor, ...], 'OrderedDict[str, Path]']:

    # the idea here is to work with "blocks" of size num_conformers / folds
    # cast to list for mypy
    conformer_blocks = list(torch.chunk(conformer_indices, folds))
    conformer_splits: List[Tensor] = []
    split_paths_list: List[Tuple[str, Path]] = []

    print(f"Generating {folds} folds for cross validation or ensemble training")
    for f in range(folds):
        # the first shuffle is necessary so that validation splits are shuffled
        validation_split = conformer_blocks[f]

        training_split = torch.cat(conformer_blocks[:f] + conformer_blocks[f + 1:])
        # afterwards all training folds are reshuffled to get different
        # batching for different models in the ensemble / cross validation
        # process (it is technically redundant to reshuffle the first one but
        # it is done for simplicity)
        training_split = _maybe_shuffle_indices(training_split, rng)
        conformer_splits.extend([training_split, validation_split])
        split_paths_list.extend([(f'training{f}', dest_path.joinpath(f'training{f}')),
                                 (f'validation{f}', dest_path.joinpath(f'validation{f}'))])
    split_paths = OrderedDict(split_paths_list)

    _create_split_paths(split_paths)

    return tuple(conformer_splits), split_paths


def _divide_into_splits(conformer_indices: Tensor,
                        dest_path: Path,
                        splits: Dict[str, float]) -> Tuple[Tuple[Tensor, ...], 'OrderedDict[str, Path]']:
    total_num_conformers = len(conformer_indices)
    split_sizes = OrderedDict([(k, int(total_num_conformers * v)) for k, v in splits.items()])
    split_paths = OrderedDict([(k, dest_path.joinpath(k)) for k in split_sizes.keys()])

    _create_split_paths(split_paths)

    leftover = total_num_conformers - sum(split_sizes.values())
    if leftover != 0:
        # We slightly modify a random section if the fractions don't split
        # the dataset perfectly. This also automatically takes care of the
        # cases leftover > 0 and leftover < 0
        any_key = list(split_sizes.keys())[0]
        split_sizes[any_key] += leftover
        assert sum(split_sizes.values()) == total_num_conformers
    conformer_splits = torch.split(conformer_indices, list(split_sizes.values()))
    assert len(conformer_splits) == len(split_sizes.values())
    print(f'Splits have number of conformers: {dict(split_sizes)}.'
          f' The requested percentages were: {splits}')
    return conformer_splits, split_paths


def _create_split_paths(split_paths: 'OrderedDict[str, Path]') -> None:
    for p in split_paths.values():
        if p.is_dir():
            subdirs = [d for d in p.iterdir()]
            if subdirs:
                raise ValueError('The dest_path provided already has files'
                                 ' or directories, please provide'
                                 ' a different path')
        else:
            if p.is_file():
                raise ValueError('The dest_path is a file, it should be a directory')
            p.mkdir(parents=True)


def _save_splits_into_batches(split_paths: 'OrderedDict[str, Path]',
                              conformer_splits: Tuple[Tensor, ...],
                              inplace_transform: Optional[Transform],
                              file_format: str,
                              include_properties: Optional[Sequence[str]],
                              h5_datasets: AniH5DatasetList,
                              padding: Optional[Dict[str, float]],
                              batch_size: int,
                              max_batches_per_packet: int,
                              verbose: bool) -> None:
    # NOTE: Explanation for following logic, please read
    #
    # This sets up a given number of batches (packet) to keep in memory and
    # then scans the dataset and find the conformers needed for the packet. It
    # then saves the batches and fetches the next packet.
    #
    # A "packet" is a list that has tensors, each of which
    # has batch indices, for instance [tensor([[0, 0, 1, 1, 2], [1, 2, 3, 5]]),
    #                                  tensor([[3, 5, 5, 5], [1, 2, 3, 3]])]
    # would be a "packet" of 2 batch_indices, each of which has in the first
    # row the index for the group, and in the second row the index for the
    # conformer
    #
    # It is important to do this with a packet and not only 1 batch.  The
    # number of reads to the h5 file is batches x conformer_groups x 3 for 1x
    # (factor of 3 from energies, species, coordinates), which means ~ 2000 x
    # 3000 x 3 = 9M reads, this is a bad bottleneck and very slow, even if we
    # fetch all necessary molecules from each conformer group simultaneously.
    #
    # Doing it for all batches at the same time is (reasonably) fast, ~ 9000
    # reads, but in this case it means we will have to put all, or almost all
    # the dataset into memory at some point, which is not feasible for larger
    # datasets.
    if inplace_transform is None:
        inplace_transform = lambda x: x  # noqa: E731

    # get all group keys concatenated in a list, with the associated file indexes
    file_idxs_and_group_keys = list(h5_datasets.iter_file_key())

    use_pbar = PKBAR_INSTALLED and verbose
    for split_path, indices_of_split in zip(split_paths.values(), conformer_splits):
        all_batch_indices = torch.split(indices_of_split, batch_size)

        all_batch_indices_packets = [all_batch_indices[j:j + max_batches_per_packet]
                                    for j in range(0, len(all_batch_indices), max_batches_per_packet)]
        num_batch_indices_packets = len(all_batch_indices_packets)

        overall_batch_idx = 0
        for j, batch_indices_packet in enumerate(all_batch_indices_packets):
            num_batches_in_packet = len(batch_indices_packet)
            # Now first we cat and sort according to the first index in order to
            # fetch all conformers of the same group simultaneously
            batch_indices_cat = torch.cat(batch_indices_packet, 0)
            indices_to_sort_batch_indices_cat = torch.argsort(batch_indices_cat[:, 0])
            sorted_batch_indices_cat = batch_indices_cat[indices_to_sort_batch_indices_cat]
            uniqued_idxs_cat, counts_cat = torch.unique_consecutive(sorted_batch_indices_cat[:, 0],
                                                                    return_counts=True)
            cumcounts_cat = cumsum_from_zero(counts_cat)

            # batch_sizes and indices_to_unsort are needed for the
            # reverse operation once the conformers have been
            # extracted
            batch_sizes = [len(batch_indices) for batch_indices in batch_indices_packet]
            indices_to_unsort_batch_cat = torch.argsort(indices_to_sort_batch_indices_cat)
            assert len(batch_sizes) <= max_batches_per_packet

            if use_pbar:
                pbar = pkbar.Pbar(f'=> Saving batch packet {j + 1} of {num_batch_indices_packets}'
                                  f' of split {split_path.name},'
                                  f' in format {file_format}', len(counts_cat))

            all_conformers: List[Properties] = []
            for step, (group_idx, count, start_index) in enumerate(zip(uniqued_idxs_cat, counts_cat, cumcounts_cat)):
                # select the specific group from the whole list of files
                file_idx, group_key = file_idxs_and_group_keys[group_idx.item()]

                # get a slice with the indices to extract the necessary
                # conformers from the group for all batches in pack.
                end_index = start_index + count
                selected_indices = sorted_batch_indices_cat[start_index:end_index, 1]

                # Important: to prevent possible bugs / errors, that may happen
                # due to incorrect conversion to indices, species is **always*
                # converted to atomic numbers when saving the batched dataset.
                conformers = h5_datasets.get_conformers(file_idx, group_key,
                                                                  selected_indices,
                                                                  include_properties)
                all_conformers.append(conformers)
                if use_pbar:
                    pbar.update(step)
            batches_cat = pad_atomic_properties(all_conformers, padding)
            # Now we need to reassign the conformers to the specified
            # batches. Since to get here we cat'ed and sorted, to
            # reassign we need to unsort and split.
            # The format of this is {'species': (batch1, batch2, ...), 'coordinates': (batch1, batch2, ...)}
            batch_packet_dict = {k: torch.split(t[indices_to_unsort_batch_cat], batch_sizes)
                                 for k, t in batches_cat.items()}

            for packet_batch_idx in range(num_batches_in_packet):
                batch = {k: v[packet_batch_idx] for k, v in batch_packet_dict.items()}
                batch = inplace_transform(batch)
                _save_batch(split_path, overall_batch_idx, batch, file_format)
                overall_batch_idx += 1
