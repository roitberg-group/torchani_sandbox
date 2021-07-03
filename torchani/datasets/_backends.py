import warnings
from pathlib import Path
from functools import partial
from abc import ABC, abstractmethod
from typing import ContextManager, Iterator, Mapping, Optional, Dict, Any, Set, Tuple, Union
from collections import OrderedDict

import numpy as np

from ._annotations import NumpyConformers
from ..utils import tqdm

try:
    import h5py
    _H5PY_AVAILABLE = True
except ImportError:
    warnings.warn('Currently the only supported backend for ANIDataset is h5py,'
                  ' very limited options are available otherwise. Installing'
                  ' h5py (pip install h5py or conda install h5py) is'
                  ' recommended if you want to use the torchani.datasets'
                  ' module')
    _H5PY_AVAILABLE = False


def StoreAdaptorFactory(store_location: Path, mode: str, backend: str, property_alias: Optional[Dict[str, str]] = None) -> '_DatasetStoreAdaptor':
    if backend == 'h5py':
        if not _H5PY_AVAILABLE:
            raise ValueError('h5py backend was specified but h5py could not be found, please install h5py')
        return _H5DatasetStoreAdaptor(h5py.File(store_location, mode), property_alias)
    else:
        raise RuntimeError(f"Bad backend {backend}")


class CacheHolder:
    def __init__(self, group_sizes: 'OrderedDict[str, int]', properties: Set[str], has_standard_format: bool):
        self.group_sizes = group_sizes
        self.properties = properties
        self.has_standard_format = has_standard_format


# ConformerGroupAdaptor and DatasetStoreAdaptor are abstract classes from which
# all backends should inherit in order to correctly interact with ANIDataset.
# adding support for a new backend can be done just by coding these two classes and
# adding the support for the backend inside StoreAdaptorFactory
class _ConformerGroupAdaptor(Mapping[str, np.ndarray], ABC):

    def __init__(self, *args, **kwargs) -> None:
        pass

    def create_numpy_values(self, conformers: NumpyConformers) -> None:
        for p, v in conformers.items():
            self._create_property_with_data(p, v)

    def append_numpy_values(self, conformers: NumpyConformers) -> None:
        for p, v in conformers.items():
            self._append_property_with_data(p, v)

    @property
    @abstractmethod
    def is_resizable(self) -> bool: pass  # noqa E704

    @abstractmethod
    def _append_property_with_data(self, p: str, data: np.ndarray) -> None: pass  # noqa E704

    @abstractmethod
    def _create_property_with_data(self, p: str, data: np.ndarray) -> None: pass  # noqa E704

    @abstractmethod
    def move(self, src: str, dest: str) -> None: pass  # noqa E704

    @abstractmethod
    def __delitem__(self, k: str) -> None: pass  # noqa E704


class _DatasetStoreAdaptor(ContextManager['_DatasetStoreAdaptor'], Mapping[str, '_ConformerGroupAdaptor'], ABC):

    def __init__(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def _set_grouping(self, grouping: str) -> None: pass  # noqa E704

    @property
    @abstractmethod
    def mode(self) -> str: pass # noqa E704

    @property
    @abstractmethod
    def grouping(self) -> str: pass # noqa E704

    @abstractmethod
    def quick_standard_format_check(self) -> bool: pass # noqa E704

    @abstractmethod
    def __delitem__(self, k: str) -> None: pass # noqa E704

    @abstractmethod
    def create_conformer_group(self, name: str) -> '_ConformerGroupAdaptor': pass # noqa E704

    @abstractmethod
    def close(self) -> None: pass # noqa E704

    def update_cache(self,
                     has_standard_format: bool,
                     check_properties: bool = False,
                     verbose: bool = True) -> Tuple['OrderedDict[str, int]', Set[str], bool]:
        cache = CacheHolder(group_sizes=OrderedDict(),
                            properties=set(),
                            has_standard_format=has_standard_format)
        if has_standard_format:
            self._update_cache_standard(cache, check_properties)
        else:
            self._update_cache_nonstandard(cache, check_properties, verbose)
        # By default iteration of HDF5 should be alphanumeric in which case
        # sorting should not be necessary, this internal check ensures the
        # groups were not created with 'track_order=True', and that the visitor
        # function worked properly.
        if list(cache.group_sizes) != sorted(cache.group_sizes):
            raise RuntimeError("Groups were not iterated upon alphanumerically")
        return cache.group_sizes, cache.properties, cache.has_standard_format

    def _update_cache_nonstandard(self, cache: CacheHolder, check_properties: bool, verbose: bool) -> None:
        pass

    def _update_cache_standard(self, cache: CacheHolder, check_properties: bool) -> None:
        pass


# Backend Specific code starts here
class _H5DatasetStoreAdaptor(_DatasetStoreAdaptor):
    def __init__(self, store_obj: h5py.File, property_aliases: Optional[Dict[str, str]] = None):
        self._store_obj = store_obj
        self._storename_to_alias = dict() if property_aliases is None else property_aliases
        self._alias_to_storename = {v: k for k, v in self._storename_to_alias.items()}

    # This is much faster (x30) than a visitor function but it assumes the
    # format is somewhat standard which means that all Groups have depth 1, and
    # all Datasets have depth 2. A pbar is not needed here since this is
    # extremely fast
    def _update_cache_standard(self, cache: CacheHolder, check_properties: bool) -> None:
        for k, g in self._store_obj.items():
            if g.name in ['/_created', '/_meta']:
                continue
            self._update_properties_cache(cache, g, check_properties)
            self._update_groups_cache(cache, g)

    def _update_cache_nonstandard(self, cache: CacheHolder, check_properties: bool, verbose: bool) -> None:
        def visitor_fn(name: str,
                       object_: Union[h5py.Dataset, h5py.Group],
                       store: '_H5DatasetStoreAdaptor',
                       cache: CacheHolder,
                       check_properties: bool,
                       pbar: Any) -> None:
            pbar.update()
            # We make sure the node is a Dataset, and We avoid Datasets
            # called _meta or _created since if present these store units
            # or other metadata. We also check if we already visited this
            # group via one of its children.
            if not isinstance(object_, h5py.Dataset) or\
                   object_.name in ['/_created', '/_meta'] or\
                   object_.parent.name in cache.group_sizes.keys():
                return
            g = object_.parent
            # Check for format correctness
            for v in g.values():
                if isinstance(v, h5py.Group):
                    raise RuntimeError(f"Invalid dataset format, there shouldn't be "
                                       "Groups inside Groups that have Datasets, "
                                       f"but {g.name}, parent of the dataset "
                                       f"{object_.name}, has group {v.name} as a "
                                       "child")
            store._update_properties_cache(cache, g, check_properties)
            store._update_groups_cache(cache, g)

        with tqdm(desc='Verifying format correctness', disable=not verbose) as pbar:
            self._store_obj.visititems(partial(visitor_fn,
                                               store=self,
                                               cache=cache,
                                               pbar=pbar,
                                               check_properties=check_properties))
        # If the visitor function succeeded and this condition is met the
        # dataset must be in standard format
        cache.has_standard_format = not any('/' in k[1:] for k in cache.group_sizes.keys())

    # Updates the "_properties", variables. "_nonbatch_properties" are keys
    # that don't have a batch dimension, their shape must be (atoms,), they
    # only make sense if ordering by formula or smiles
    def _update_properties_cache(self, cache: CacheHolder, conformers: h5py.Group, check_properties: bool = False) -> None:
        if not cache.properties:
            cache.properties = {self._storename_to_alias.get(p, p) for p in set(conformers.keys())}
        elif check_properties:
            found_properties = {self._storename_to_alias.get(p, p) for p in set(conformers.keys())}
            if not found_properties == cache.properties:
                raise RuntimeError(f"Group {conformers.name} has bad keys, "
                                   f"found {found_properties}, but expected "
                                   f"{cache.properties}")

    # updates "group_sizes" which holds the batch dimension (number of
    # molecules) of all groups in the dataset.
    def _update_groups_cache(self, cache: CacheHolder, group: h5py.Group) -> None:
        present_keys = {'coordinates', 'coord', 'energies'}.intersection(set(group.keys()))
        try:
            any_key = tuple(present_keys)[0]
        except IndexError:
            raise RuntimeError('To infer conformer size need one of "coordinates", "coord", "energies"')
        cache.group_sizes.update({group.name[1:]: group[any_key].shape[0]})

    def _set_grouping(self, grouping: str) -> None:
        self._store_obj.attrs['grouping'] = grouping

    # Check if the raw hdf5 file has the '/_created' dataset if this is the
    # case then it can be assumed to have standard format. All other h5
    # files are assumed to **not** have standard format.
    def quick_standard_format_check(self) -> bool:
        try:
            self._store_obj['/_created']
            return True
        except KeyError:
            return False

    @property
    def mode(self) -> str:
        mode = self._store_obj.mode
        assert isinstance(mode, str)
        return mode

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

    def create_conformer_group(self, name: str) -> '_H5ConformerGroupAdaptor':
        self._store_obj.create_group(name)
        return self[name]

    def close(self) -> None:
        self._store_obj.close()

    def __enter__(self) -> '_H5DatasetStoreAdaptor':
        self._store_obj.__enter__()
        return self

    def __exit__(self, *args) -> None:
        self._store_obj.__exit__(*args)

    def __getitem__(self, name: str) -> '_H5ConformerGroupAdaptor':
        return _H5ConformerGroupAdaptor(self._store_obj[name], self._storename_to_alias)

    def __len__(self) -> int:
        return len(self._store_obj)

    def __iter__(self) -> Iterator[str]:
        return iter(self._store_obj)


class _H5ConformerGroupAdaptor(_ConformerGroupAdaptor):
    def __init__(self, group_obj: h5py.Group, property_aliases: Optional[Dict[str, str]] = None):
        self._group_obj = group_obj
        _storename_to_alias = dict() if property_aliases is None else property_aliases
        self._alias_to_storename = {v: k for k, v in _storename_to_alias.items()}

    @property
    def is_resizable(self) -> bool:
        return all(ds.maxshape[0] is None for ds in self._group_obj.values())

    def _append_property_with_data(self, p: str, data: np.ndarray) -> None:
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
