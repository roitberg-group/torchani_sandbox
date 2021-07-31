import tempfile
from pathlib import Path
from typing import ContextManager, Set, Tuple, Optional
from collections import OrderedDict

import numpy as np

from .._annotations import StrPath
from .interface import _Store, _ConformerGroup, _ConformerWrapper, CacheHolder, _OnDiskLocation
from .h5py_impl import _H5Store


try:
    import zarr  # noqa
    _ZARR_AVAILABLE = True
except ImportError:
    _ZARR_AVAILABLE = False


class _ZarrTemporaryLocation(ContextManager[StrPath]):
    def __init__(self) -> None:
        self._tmp_location = tempfile.TemporaryDirectory(suffix='.zarr')

    def __enter__(self) -> str:
        return self._tmp_location.name

    def __exit__(self, *args) -> None:
        if Path(self._tmp_location.name).exists():  # check necessary for python 3.6
            self._tmp_location.cleanup()


# Backend Specific code starts here
class _ZarrStore(_H5Store):

    location = _OnDiskLocation('.zarr', kind='dir')

    def __init__(self, store_location: StrPath):
        self.location = store_location
        self._store_obj = None
        self._mode: Optional[str] = None

    def validate_location(self) -> None:
        if not self.location.is_dir():
            raise FileNotFoundError(f"The store in {self._store_location} could not be found")

    def make_empty(self, grouping: str) -> None:
        store = zarr.storage.DirectoryStore(self._store_location)
        with zarr.hierarchy.group(store=store, overwrite=True) as g:
            g.attrs['grouping'] = grouping

    def open(self, mode: str = 'r') -> '_Store':
        store = zarr.storage.DirectoryStore(self._store_location)
        self._store_obj = zarr.hierarchy.open_group(store, mode)
        self._mode = mode
        return self

    def close(self) -> '_Store':
        # Zarr Groups actually wrap a store, but DirectoryStore has no "close"
        # method Other stores may have a "close" method though
        try:
            self._store.store.close()
        except AttributeError:
            pass
        self._store_obj = None
        return self

    @property
    def _store(self) -> "zarr.Group":
        if self._store_obj is None:
            raise RuntimeError("Can't access store")
        return self._store_obj

    def update_cache(self,
                     check_properties: bool = False,
                     verbose: bool = True) -> Tuple['OrderedDict[str, int]', Set[str]]:
        cache = CacheHolder()
        for k, g in self._store.items():
            self._update_properties_cache(cache, g, check_properties)
            self._update_groups_cache(cache, g)
        if list(cache.group_sizes) != sorted(cache.group_sizes):
            raise RuntimeError("Groups were not iterated upon alphanumerically")
        return cache.group_sizes, cache.properties

    @property
    def mode(self) -> str:
        if self._mode is None:
            raise RuntimeError("Can't access a closed store")
        return self._mode

    @property
    def grouping(self) -> str:
        g = self._store.attrs['grouping']
        assert isinstance(g, str)
        return g

    def __getitem__(self, name: str) -> '_ConformerGroup':
        return _ZarrConformerGroup(self._store[name])


class _ZarrConformerGroup(_ConformerWrapper):
    def __init__(self, data: "zarr.Group"):
        self._data = data

    def _append_to_property(self, p: str, v: np.ndarray) -> None:
        try:
            self._data[p].append(v, axis=0)
        except TypeError:
            self._data[p].append(v.astype(bytes), axis=0)

    def __setitem__(self, p: str, v: np.ndarray) -> None:
        try:
            self._data.create_dataset(name=p, data=v)
        except TypeError:
            self._data.create_dataset(name=p, data=v.astype(bytes))

    def move(self, src_p: str, dest_p: str) -> None:
        self._data.move(src_p, dest_p)
