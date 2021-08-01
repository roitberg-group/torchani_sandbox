import tempfile
from pathlib import Path
from typing import ContextManager, Set, Tuple
from collections import OrderedDict

import numpy as np

from .._annotations import StrPath
from .interface import _Store, _ConformerGroup, _ConformerWrapper, CacheHolder, _HierarchicalStoreWrapper

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


class _ZarrStore(_HierarchicalStoreWrapper["zarr.Group"]):
    def __init__(self, store_location: StrPath):
        super().__init__(store_location, '.zarr', 'dir')

    @classmethod
    def make_empty(cls, store_location: StrPath, grouping: str) -> '_Store':
        store = zarr.storage.DirectoryStore(store_location)
        with zarr.hierarchy.group(store=store, overwrite=True) as g:
            g.attrs['grouping'] = grouping
        return cls(store_location)

    def open(self, mode: str = 'r') -> '_Store':
        store = zarr.storage.DirectoryStore(self.location.root)
        self._store_obj = zarr.hierarchy.open_group(store, mode)
        setattr(self._store_obj, 'mode', mode)
        return self

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

    def __getitem__(self, name: str) -> '_ConformerGroup':
        return _ZarrConformerGroup(self._store[name])


class _ZarrConformerGroup(_ConformerWrapper["zarr.Group"]):
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
