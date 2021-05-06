from pathlib import Path
import pickle
import warnings
import torch
import h5py
import importlib
from typing import Union, Optional, List, Dict, Any, Callable
import numpy as np
from collections import OrderedDict
from collections.abc import Mapping
from functools import partial
from torchani.utils import pad_atomic_properties
from torchani.utils import ChemicalSymbolsToInts
from torchani.utils import cumsum_from_zero
# from torchani.nn import SpeciesConverter


PKBAR_INSTALLED = importlib.util.find_spec('pkbar') is not None  # type: ignore
if PKBAR_INSTALLED:
    import pkbar

PADDING = {
    'species': -1,
    'coordinates': 0.0,
    'forces': 0.0,
    'energies': 0.0
}


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
    if use_pbar:
        pbar = pkbar.Pbar(f'=> saving dataset to {path}, in format {file_format}, total molecules: {len(dataset)}', len(dataset))
    for j, batch in enumerate(dataset):
        _save_batch(path, j, batch, file_format)
        if use_pbar:
            pbar.update(j)


def _save_batch(path, idx, batch, file_format):
    # We use pickle or numpy since saving in
    # pytorch format is extremely slow
    if file_format == 'pickle':
        with open(path.joinpath(f'batch{idx}.pkl'), 'wb') as batch_file:
            pickle.dump({'species': batch['species'].numpy(),
                    'coordinates': batch['coordinates'].numpy(),
                    'energies': batch['energies'].numpy()}, batch_file)
    elif file_format == 'numpy':
        np.savez(path.joinpath(f'batch{idx}'),
                species=batch['species'].numpy(),
                coordinates=batch['coordinates'].numpy(),
                energies=batch['energies'].numpy())


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

        def numpy_extractor(idx, paths):
            return {k: torch.as_tensor(v) for k, v in np.load(paths[idx]).items()}

        def pickle_extractor(idx, paths):
            with open(paths[idx], 'rb') as f:
                return {k: torch.as_tensor(v) for k, v in pickle.load(f).items()}

        # We use pickle or numpy since saving in
        # pytorch format is extremely slow
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

    # NOTE: I think this function signature should be changed, we should probably follow
    # something like torchvision API and pass a bunch of transformation
    # objects, instead of this many optional parameters, but at the moment
    # I do this for simplicity.
    #
    # maybe transformation objects should be:
    # IncludeProperties(ppties), Batch(batch_size, collate_fn, padding, num_batches_per_packet),
    # Split(splits), Shuffle(), ConvertSpecies(elements)
    #
    # and signature would be:
    # to_batched_dataset(path, transforms, verbose, file_format)
    #
    # This would also keep things similar to Xiang's dataloader
    def to_batched_dataset(self,
                           path: Union[str, Path],
                           shuffle: bool = True,
                           file_format: str = 'numpy',
                           include_properties=('species', 'coordinates', 'energies'),
                           batch_size: int = 2560,
                           collate_fn: Optional[Callable] = None,
                           padding: Optional[Dict[str, Any]] = None,
                           splits: Optional[Dict[str, float]] = None,
                           verbose: bool = True,
                           max_batches_per_packet: int = 300,
                           elements=('H', 'C', 'N', 'O')):
        # NOTE: all the tensor manipulation in this function is handled in CPU
        # temporary hack #
        element_keys = ('species', 'numbers', 'atomic_numbers')
        element_keys = tuple((k for k in include_properties if k in element_keys))
        include_properties = tuple((k for k in include_properties if k not in element_keys))
        # ################

        chemical_symbols_to_ints = ChemicalSymbolsToInts(elements)
        use_pbar = PKBAR_INSTALLED and verbose

        if not shuffle:
            warnings.warn("Dataset will not be shuffled, this should only be used for debugging")
        if padding is None:
            padding = PADDING
        if collate_fn is None:
            collate_fn = pad_atomic_properties
        if isinstance(path, str):
            path = Path(path).resolve()

        # I do this here so thet split_sizes and split_paths are synchronized
        # splits is a dictionary with "split_name" "split_percentage"
        if splits is None:
            splits = {'training': 0.8, 'validation': 0.2}
        else:
            assert isinstance(splits, dict)
            if not torch.isclose(torch.tensor(sum(list(splits.values()))), 1.0):
                raise ValueError("The sum of the split fractions has to add up to one")

        split_sizes = OrderedDict([(k, int(self.num_conformers * v)) for k, v in splits.items()])
        split_paths = OrderedDict([(k, path.joinpath(k)) for k in split_sizes.keys()])

        for p in split_paths.values():
            if p.is_dir():
                subdirs = [d for d in p.iterdir()]
                assert not subdirs, "The path provided already has files or directories, please provide a different path"
            else:
                assert not p.is_file(), "The path provided is a file, not a directory"
                p.mkdir(parents=True)

        # (1) Get all indices and shuffle them if needed
        group_sizes = torch.tensor(list(self._groups.values()), dtype=torch.long)
        # These are pairs of indices that index first the group and then the
        # specific conformer, it is possible to just use one index for
        # everything but this is simpler at the cost of slightly more memory
        conformer_indices = [torch.stack((torch.full(size=(s,), fill_value=j),
                                         (torch.arange(0, s))), dim=-1)
                                         for j, s in enumerate(group_sizes)]
        conformer_indices = torch.cat(conformer_indices)
        if shuffle:
            shuffle_indices = torch.randperm(self.num_conformers)
            conformer_indices = conformer_indices[shuffle_indices]

        # (2) Split shuffled indices according to requested dataset splits
        leftover = self.num_conformers - sum(split_sizes.values())
        if leftover != 0:
            # We slightly modify the max section if the fractions don't split
            # the dataset perfectly. This also automatically takes care of the
            # cases leftover > 0 and leftover < 0
            split_sizes[max(split_sizes, key=split_sizes.get)] += leftover
            assert sum(split_sizes.values()) == self.num_conformers
        conformer_splits = torch.split(conformer_indices, list(split_sizes.values()))
        assert len(conformer_splits) == len(split_sizes.values())
        print(f"""Splits have number of conformers: {dict(split_sizes)}.
                  The requested percentages were: {splits}""")

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

            with h5py.File(self._store_file, 'r') as hdf5_file:
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
                        pbar = pkbar.Pbar(f"""=> Saving batch packet {j + 1} of
                                          {num_batch_indices_packets} into
                                          {split_paths[split_key].as_posix()}, in format
                                          {file_format}""",
                                          len(counts_cat))
                    for step, (group_idx, count, start_index) in enumerate(zip(uniqued_idxs_cat, counts_cat, cumcounts_cat)):
                        group = hdf5_file[list(self._groups.keys())[group_idx.item()]]
                        end_index = start_index + count
                        # get a slice with the indices to extract the necessary
                        # conformers from the group for all batches in pack.
                        selected_indices = sorted_batch_indices_cat[start_index:end_index, 1]
                        # copying this avoids directly indexing the HDF5
                        # dataset, which is extremely expensive
                        conformers = {k: np.copy(group[k]) for k in include_properties}
                        conformers = {k: v[selected_indices.cpu().numpy()] for k, v in conformers.items()}
                        element_keys = ('species',)
                        conformers.update({k: np.copy(group[k]).astype(str).tolist() for k in element_keys})

                        if 'species' in element_keys:
                            converted_species = chemical_symbols_to_ints(conformers['species'])
                            converted_species = converted_species.view(1, -1).repeat(count, 1)
                            conformers['species'] = converted_species

                        conformers = {k: torch.as_tensor(v) for k, v in conformers.items()}
                        all_conformers.append(conformers)
                        if use_pbar:
                            pbar.update(step)
                    batches_cat = collate_fn(all_conformers, padding)
                    # Now we need to reassign the conformers to the specified
                    # batches. Since to get here we cat'ed and sorted, to
                    # reassign we need to unsort and split.
                    # The format of this is {'species': (batch1, batch2, ...), 'coordinates': (batch1, batch2, ...)}
                    batch_packet_dict = {k: torch.split(t[indices_to_unsort_batch_cat], batch_sizes)
                               for k, t in batches_cat.items()}
                    for batch_idx in range(num_batches_in_packet):
                        batch = {k: v[batch_idx] for k, v in batch_packet_dict.items()}
                        _save_batch(split_paths[split_key], batch_idx, batch, file_format)

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


if __name__ == '__main__':

    ds = H5Dataset('/home/ignacio/Datasets/ani1x_release_wb97x_dz.h5')
    ds.to_batched_dataset('./1x_batched_ds', file_format='pytorch')
