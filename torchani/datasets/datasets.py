from pathlib import Path
from functools import partial
import pickle
import warnings
import importlib
from typing import Union, Optional, List, Dict, Any, Callable, Generator
from collections import OrderedDict
from collections.abc import Mapping

import torch
import h5py
import numpy as np

from torchani.utils import pad_atomic_properties, ChemicalSymbolsToAtomicNumbers, cumsum_from_zero

PKBAR_INSTALLED = importlib.util.find_spec('pkbar') is not None  # type: ignore
if PKBAR_INSTALLED:
    import pkbar

PADDING = {
    'species': -1,
    'numbers': -1,
    'atomic_numbers': -1,
    'coordinates': 0.0,
    'forces': 0.0,
    'energies': 0.0
}

# These keys are treated differently because they don't have a batch dimension
ELEMENT_KEYS = ('species', 'numbers', 'atomic_numbers')


class AniBatchedDataset(torch.utils.data.Dataset):

    SUPPORTED_FILE_FORMATS = ('numpy', 'hdf5', 'single_hdf5', 'pickle')

    def __init__(self, store_dir: Union[str, Path],
                       file_format: Optional[str] = None,
                       split: str = 'training',
                       transform: Callable = lambda x: x):

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
        self.transform = transform

        def numpy_extractor(idx, paths):
            return {
                k: torch.as_tensor(v)
                for k, v in np.load(paths[idx]).items()
            }

        def pickle_extractor(idx, paths):
            with open(paths[idx], 'rb') as f:
                return {
                    k: torch.as_tensor(v)
                    for k, v in pickle.load(f).items()
                }

        def hdf5_extractor(idx, paths):
            with h5py.File(paths[idx], 'r') as f:
                return {k: torch.as_tensor(v[()]) for k, v in f['/'].items()}

        def single_hdf5_extractor(idx, group_keys, path):
            k = group_keys[idx]
            with h5py.File(path, 'r') as f:
                return {k: torch.as_tensor(v[()]) for k, v in f[k].items()}

        # We use pickle or numpy or hdf5 since saving in
        # pytorch format is extremely slow
        format_suffix_map = {'.npz': 'numpy', '.pkl': 'pickle', '.h5': 'hdf5'}
        self._len = len(self.batch_paths)

        if file_format is None:
            file_format = format_suffix_map[suffix]
            if file_format == 'hdf5' and ('single' in self.batch_paths[0].name):
                file_format = 'single_hdf5'

        assert file_format is not None
        assert file_format in self.SUPPORTED_FILE_FORMATS
        if file_format == 'numpy':
            self.extractor = partial(numpy_extractor, paths=self.batch_paths)
        elif file_format == 'pickle':
            self.extractor = partial(pickle_extractor, paths=self.batch_paths)
        elif file_format == 'hdf5':
            self.extractor = partial(hdf5_extractor, paths=self.batch_paths)
        elif file_format == 'single_hdf5':
            warnings.warn('Depending on the implementation, a single HDF5 file'
                          ' may not support parallel reads, so using num_workers > 1'
                          ' may have a detrimental effect on performance')
            with h5py.File(self.batch_paths[0], 'r') as f:
                keys = list(f.keys())
                self._len = len(keys)
                self.extractor = partial(single_hdf5_extractor, group_keys=keys, path=self.batch_paths[0])
        else:
            msg = f'Format for file with extension {suffix} could not be infered, please specify explicitly'
            raise RuntimeError(msg)

    def cache(self, pin_memory=True, verbose=True, apply_transform=True):
        if verbose:
            print("Cacheing dataset, this may take some time...")
            print("Cacheing the dataset may use a lot of memory, be careful!")

        def memory_extractor(idx, ds):
            return ds._data[idx]

        self._data = [self.extractor(idx) for idx in range(len(self))]

        if apply_transform:
            if verbose:
                print("Transformations will be applied once during cacheing and then discarded.")
            with torch.no_grad():
                self._data = [self.transform(properties) for properties in self._data]
            # discard transform after aplication
            self.transform = lambda x: x

        # When the dataset is cached memory pinning is done here. When the
        # dataset is not cached memory pinning is done by the torch DataLoader.
        if pin_memory:
            if verbose:
                print("Cacheing pins memory automatically, do **not** use pin_memory=True in torch.utils.data.DataLoader")
            self._data = [{k: v.pin_memory() for k, v in properties.items()} for properties in self._data]

        self.extractor = partial(memory_extractor, ds=self)
        return self

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # integral indices must be provided for compatibility with pytorch
        # DataLoader API
        properties = self.extractor(idx)
        with torch.no_grad():
            properties = self.transform(properties)
        return properties

    def __iter__(self) -> Generator[Dict[str, Any], None, None]:
        j = 0
        try:
            while True:
                yield self[j]
                j += 1
        except IndexError:
            return

    def __len__(self):
        return self._len


class AniH5Dataset(Mapping):

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
        self._groups: Dict[str, Any] = OrderedDict()
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
            conformer_group = self._get_group(k, include_properties, strict)
            for j in range(size):
                yield self._extract_from_molecule_group(conformer_group, j, **kwargs)
    # end API

    def _cache_group_paths_and_sizes(self):
        # cache paths of all molecule groups into a list
        self.group_sizes = []

        def visitor_fn(name, object_, ds: AniH5Dataset):
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
                    ds.group_sizes.append((molecule_group.name, size))

        with h5py.File(self._store_file, 'r') as f:
            f.visititems(partial(visitor_fn, ds=self))
        self._groups = OrderedDict(self.group_sizes)

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


def _save_batch(path, idx, batch, file_format):
    # We use pickle or numpy since saving in
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


def create_batched_dataset(h5_path: Union[str, Path, List[Union[str, Path]]],
                           dest_path: Optional[Union[str, Path]] = None,
                           shuffle: bool = True,
                           shuffle_seed: Optional[int] = None,
                           file_format: str = 'hdf5',
                           include_properties=('species', 'coordinates', 'energies'),
                           batch_size: int = 2560,
                           max_batches_per_packet: int = 350,
                           collate_fn: Optional[Callable] = None,
                           padding: Optional[Dict[str, Any]] = None,
                           splits: Optional[Dict[str, float]] = None,
                           inplace_transform: Callable[[Dict[str, Any]], Dict[str, Any]] = lambda x: x,
                           verbose: bool = True):
    if file_format == 'single_hdf5':
        warnings.warn('Depending on the implementation, a single HDF5 file may'
                      'not support parallel reads, so using num_workers > 1 may'
                      'have a detrimental effect on performance, its probably better'
                      'to save in many hdf5 files with file_format=hdf5')
    # NOTE: all the tensor manipulation in this function is handled in CPU
    # NOTE: an inplace transform can be applied to the dataset if the transform
    # is very costly to perform on the fly when training

    # Properties that are ELEMENT_KEYS have to be treated differently because
    # they don't have a batch dimension, so we separate them here
    include_element_keys = tuple((k for k in include_properties if k in ELEMENT_KEYS))
    include_properties = tuple((k for k in include_properties if k not in ELEMENT_KEYS))

    if not shuffle:
        warnings.warn("Dataset will not be shuffled, this should only be used for debugging")

    if padding is None:
        padding = PADDING

    if collate_fn is None:
        collate_fn = pad_atomic_properties

    if dest_path is None:
        dest_path = Path(f'./batched_dataset_{file_format}').resolve()
    dest_path = Path(dest_path).resolve()

    if isinstance(h5_path, (str, Path)):
        h5_path = Path(h5_path).resolve()
        assert isinstance(h5_path, Path)
        if h5_path.is_dir():
            h5_paths = [p for p in h5_path.iterdir() if p.suffix == '.h5']
        elif h5_path.is_file():
            h5_paths = [h5_path]
    else:
        try:
            h5_paths = [Path(p).resolve() for p in h5_path]
        except TypeError:
            raise TypeError("Expected a path to an h5 file or a dir containing h5 files, or a list of h5 file paths")

    h5_datasets = [AniH5Dataset(p) for p in h5_paths]

    total_num_conformers = sum([h5ds.num_conformers for h5ds in h5_datasets])
    # get all group sizes for all datasets concatenated in a row, in the same
    # order as h5_datasets
    group_sizes = torch.cat([torch.tensor(list(h5ds._groups.values()), dtype=torch.long) for h5ds in h5_datasets])
    # get all group keys concatenated in a row, with the associated file indexes
    file_idxs_and_group_keys = [{'idx': j, 'key': k}
                  for j, h5ds in enumerate(h5_datasets)
                  for k in h5ds._groups.keys()]

    # I do this here so thet split_sizes and split_paths are synchronized.
    # splits is a dictionary with "split_name" "split_percentage"
    if splits is None:
        splits = {'training': 0.8, 'validation': 0.2}
    else:
        assert isinstance(splits, dict)
    if not torch.isclose(torch.tensor(sum(list(splits.values()))), torch.tensor(1.0)):
        raise ValueError("The sum of the split fractions has to add up to one")

    split_sizes = OrderedDict([(k, int(total_num_conformers * v)) for k, v in splits.items()])
    split_paths = OrderedDict([(k, dest_path.joinpath(k)) for k in split_sizes.keys()])

    for p in split_paths.values():
        if p.is_dir():
            subdirs = [d for d in p.iterdir()]
            assert not subdirs, "The dest_path provided already has files or directories, please provide a different path"
        else:
            assert not p.is_file(), "The dest_path provided is a file, not a directory"
            p.mkdir(parents=True)

    # Important: to prevent possible bugs / errors, that may happen due to
    # incorrect conversion to indices, species is **always** converted to
    # atomic numbers when saving the batched dataset.
    symbols_to_atomic_numbers = ChemicalSymbolsToAtomicNumbers()

    use_pbar = PKBAR_INSTALLED and verbose

    # (1) Get all indices and shuffle them if needed
    # These are pairs of indices that index first the group and then the
    # specific conformer, it is possible to just use one index for
    # everything but this is simpler at the cost of slightly more memory
    conformer_indices = torch.cat([torch.stack((torch.full(size=(s.item(),), fill_value=j),
                                     (torch.arange(0, s.item()))), dim=-1)
                                     for j, s in enumerate(group_sizes)])
    if shuffle:
        if shuffle_seed is None:
            shuffle_indices = torch.randperm(total_num_conformers)
        else:
            generator = torch.manual_seed(shuffle_seed)
            shuffle_indices = torch.randperm(total_num_conformers, generator=generator)
        conformer_indices = conformer_indices[shuffle_indices]

    # (2) Split shuffled indices according to requested dataset splits
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

    # (3) Compute the batch indices for each split and save the conformers to disk
    for split_key, indices_of_split in zip(split_sizes.keys(), conformer_splits):
        all_batch_indices = torch.split(indices_of_split, batch_size)
        # NOTE: Explanation for complicated logic, please read
        #
        # This sets up a given number of batches (packet) to keep in memory
        # and then scans the dataset and find the conformers needed for
        # the packet. It then saves the batches and fetches the next packet.
        #
        # A "packet" is just a list that has tensors, each of which
        # has batch indices, for instance [tensor([[0, 0, 1, 1, 2], [1, 2, 3, 5]]),
        #                                  tensor([[3, 5, 5, 5], [1, 2, 3, 3]])]
        # would be a "packet" of 2 batch_indices, each of which has in the first row the
        # index for the group, and in the second row the index for the conformer
        #
        # It is important to do this with a packet and not only 1 batch
        # the number of reads to the h5 file is batches x conformer_groups
        # x 3 for 1x (factor of 3 from energies, species, coordinates),
        # which means ~ 2000 x 3000 x 3 = 9M reads, this is a bad
        # bottleneck and very slow, even if we fetch all necessary
        # molecules from each conformer group simultaneously.
        #
        # Doing it for all batches at the same time is (reasonably) fast,
        # ~ 9000 reads, but in this case it means we will have to put
        # all, or almost all the dataset into memory at some point, which
        # is not feasible for larger datasets so it is better if the max
        # number of batches in each packet is some intermediate number.
        # (Maybe some heuristic is needed to calculate this automatically,
        # but I found 350 to be a good compromise).
        #
        # if max_batches_per_packet = num_batches many of the following
        # logic is not necessary, but it is not worth it to simplify for
        # this specific case since the bottleneck is IO by far.

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
            uniqued_idxs_cat, counts_cat = torch.unique_consecutive(sorted_batch_indices_cat[:, 0], return_counts=True)
            cumcounts_cat = cumsum_from_zero(counts_cat)
            assert len(uniqued_idxs_cat) == len(counts_cat)
            assert len(counts_cat) == len(cumcounts_cat)

            # batch_sizes and indices_to_unsort are needed for the
            # reverse operation once the conformers have been
            # extracted
            batch_sizes = [len(batch_indices) for batch_indices in batch_indices_packet]
            indices_to_unsort_batch_cat = torch.argsort(indices_to_sort_batch_indices_cat)
            assert len(batch_sizes) <= max_batches_per_packet

            all_conformers = []
            if use_pbar:
                pbar = pkbar.Pbar(f'=> Saving batch packet {j + 1} of {num_batch_indices_packets}'
                                  f' of split {split_paths[split_key].name},'
                                  f' in format {file_format}', len(counts_cat))

            # no need to wrap this file opening code in a try/except block for now, since an
            # exception during this should abort immediatly anyways
            h5_files = [h5py.File(h5ds._store_file, 'r') for h5ds in h5_datasets]
            for step, (group_idx, count, start_index) in enumerate(zip(uniqued_idxs_cat, counts_cat, cumcounts_cat)):
                # select the group from the whole list of files
                idx_and_key = file_idxs_and_group_keys[group_idx.item()]
                group = h5_files[idx_and_key['idx']][idx_and_key['key']]

                end_index = start_index + count
                # get a slice with the indices to extract the necessary
                # conformers from the group for all batches in pack.
                selected_indices = sorted_batch_indices_cat[start_index:end_index, 1]
                # copying this avoids directly indexing the HDF5
                # dataset, which is extremely expensive
                conformers = {k: np.copy(group[k]) for k in include_properties}
                conformers = {k: v[selected_indices.cpu().numpy()] for k, v in conformers.items()}
                element_keys = ('species',)
                conformers.update({k: np.copy(group[k]).astype(str) for k in include_element_keys})

                if 'species' in element_keys:
                    converted_species = symbols_to_atomic_numbers(conformers['species'])
                    conformers['species'] = converted_species

                for k in element_keys:
                    conformers[k] = conformers[k].view(1, -1).repeat(count, 1)

                conformers = {k: torch.as_tensor(v) for k, v in conformers.items()}
                all_conformers.append(conformers)
                if use_pbar:
                    pbar.update(step)
            for f in h5_files:
                f.close()

            batches_cat = collate_fn(all_conformers, padding)
            # Now we need to reassign the conformers to the specified
            # batches. Since to get here we cat'ed and sorted, to
            # reassign we need to unsort and split.
            # The format of this is {'species': (batch1, batch2, ...), 'coordinates': (batch1, batch2, ...)}
            batch_packet_dict = {k: torch.split(t[indices_to_unsort_batch_cat], batch_sizes)
                                 for k, t in batches_cat.items()}

            for packet_batch_idx in range(num_batches_in_packet):
                batch = {k: v[packet_batch_idx] for k, v in batch_packet_dict.items()}
                batch = inplace_transform(batch)
                _save_batch(split_paths[split_key], overall_batch_idx, batch, file_format)
                overall_batch_idx += 1
