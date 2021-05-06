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

    @profile
    def to_batched_dataset(self,
                           path: Union[str, Path],
                           shuffle: bool = True,
                           file_format: str = 'numpy',
                           include_properties=('species', 'coordinates', 'energies'),
                           device: str = 'cpu',
                           batch_size: int = 2560,
                           collate_fn: Optional[Callable] = None,
                           padding: Optional[Dict[str, Any]] = None,
                           splits: Optional[Dict[str, float]] = None,
                           verbose: bool = True,
                           elements=('H', 'C', 'N', 'O')):
        # temporary hack
        element_keys = ('species', 'numbers', 'atomic_numbers')
        element_keys = tuple((k for k in include_properties if k in element_keys))
        include_properties = tuple((k for k in include_properties if k not in element_keys))
        # NOTE: I think this should be changed, we should probably follow
        # something like torchvision API and pass a bunch of transformation
        # objects, instead of this many optional parameters, but at the moment
        # I do this for simplicity.
        #
        # maybe transformation objects should be:
        #
        # IncludeProperties(ppties), Batch(batch_size, collate_fn, padding), Split(splits), Shuffle(), ConvertSpecies(elements)
        # and signature would be:
        #
        # to_batched_dataset(path, transforms, verbose, device, file_format)
        #
        # This would also keep things similar to Xiang's dataloader but avoid the memory
        # explosion
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

        if device == 'cpu':
            warnings.warn("""Index fetching and shuffling operation will be
                handled by CPU, this is usually fine, but if you have
                performance issues for dataset loading pass device="gpu" to
                offload this calculation to GPU""")

        # (1) Get all indices and shuffle them if needed
        group_sizes = torch.tensor(list(self._groups.values()), dtype=torch.long, device=device)
        # These are pairs of indices that index first the group and then the
        # specific conformer, it is possible to just use one index for everything
        # but this is a bit simpler at the cost of some more GPU memory
        conformer_indices = [torch.stack((torch.full(size=(s,), fill_value=j, device=device), (torch.arange(0, s, device=device))), dim=-1) for j, s in enumerate(group_sizes)]
        conformer_indices = torch.cat(conformer_indices)
        if shuffle:
            shuffle_indices = torch.randperm(self.num_conformers, device=device)
            conformer_indices = conformer_indices[shuffle_indices]

        # (2) Split shuffled indices according to requested dataset splits
        leftover = self.num_conformers - sum(split_sizes.values())
        if leftover != 0:
            # modifying the maximum section will have the least effect
            # hopefully, if the fractions don't split the dataset perfectly
            # this automatically takes care of the cases leftover > 0 and leftover < 0
            split_sizes[max(split_sizes, key=split_sizes.get)] += leftover
            assert sum(split_sizes.values()) == self.num_conformers
        conformer_splits = torch.split(conformer_indices, list(split_sizes.values()))
        print(f'Splits have the following number of conformers each: {split_sizes}. The requested percentages were {splits}')

        # (3) Compute the batch indices for each split and save them to disk
        for k, indices_of_split in zip(split_sizes.keys(), conformer_splits):
            all_batch_indices = torch.split(indices_of_split, batch_size)
            if use_pbar:
                pbar = pkbar.Pbar(f'=> saving to {split_paths[k].as_posix()}, in format {file_format}, total batches: {len(all_batch_indices)}', len(all_batch_indices))

            with h5py.File(self._store_file, 'r') as hdf5_file:
                for j, batch_indices in enumerate(all_batch_indices):
                    # now that we have batches of indices for a specific section, we
                    # need to index into the database and generate all necessary
                    # batches
                    # first we sort according to the first index in order to fetch all conformers of the same group simultaneously
                    sort_idxs = torch.argsort(batch_indices[:, 0])
                    sorted_batch_indices = batch_indices[sort_idxs]
                    uniqued_group_index, counts = torch.unique_consecutive(sorted_batch_indices[:, 0], return_counts=True)
                    cumcounts = cumsum_from_zero(counts)
                    all_batch_conformers = []
                    for idx, count, start_index in zip(uniqued_group_index, counts, cumcounts):
                        group_key = list(self._groups.keys())[idx.item()]
                        group = hdf5_file[group_key]
                        end_index = start_index + count
                        # get a slice with the indices to extract the necessary
                        # conformers from the group
                        selected_indices, _ = torch.sort(sorted_batch_indices[start_index:end_index, 1])
                        selected_indices = selected_indices.numpy()
                        #
                        # Note that in the original code all conformers are
                        # individually fetched, and here we fetch in batches from
                        # the same groups, however, it may actually be faster to
                        # individually fetch all conformers if the number of
                        # conformers we get from each group is low, because I don't
                        # index hdf5 directly some benchmarking is needed to test
                        # this. Another alternative is to index hdf5 directly and
                        # pre-sort the shuffled indices
                        element_keys = ('species',)
                        conformers = {k: np.copy(group[k][selected_indices]) for k in include_properties}
                        conformers.update({k: np.copy(group[k]) for k in element_keys})
                        if 'species' in element_keys:
                            conformers['species'] = chemical_symbols_to_ints(conformers['species']).view(1, -1).repeat(count, 1)
                        conformers = {k: torch.as_tensor(v) for k, v in conformers.items()}
                        all_batch_conformers.append(conformers)
                    # Now that we have a list of conformers for this specific batch,
                    # the next step is to collate them using the collate_fn
                    batch = collate_fn(all_batch_conformers, padding)
                    _save_batch(split_paths[k], j, batch, file_format)
                    if use_pbar:
                        pbar.update(j)

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
    ds.to_batched_dataset('./1x_batched_ds')
