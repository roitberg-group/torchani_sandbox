import json
import tempfile
from pathlib import Path
from typing import Iterator, Set, Union, Tuple
from collections import OrderedDict

import numpy as np

from .._annotations import StrPath
from .interface import _Store, _StoreWrapper, _ConformerGroup, CacheHolder, _FileOrDirLocation
from .zarr_impl import _ZarrTemporaryLocation


try:
    import cudf
    _CUDF_AVAILABLE = True
    default_engine = cudf
except ImportError:
    _CUDF_AVAILABLE = False

try:
    import pandas
    _PANDAS_AVAILABLE = True
    if not _CUDF_AVAILABLE:
        default_engine = pandas
except ImportError:
    _PANDAS_AVAILABLE = False

_PQ_AVAILABLE = _PANDAS_AVAILABLE or _CUDF_AVAILABLE


def _to_dict_pandas(df, **kwargs):
    return df.to_dict(df, **kwargs)


def _to_dict_cudf(df, **kwargs):
    return df.to_pandas().to_dict(df, **kwargs)


class _PqLocation(_FileOrDirLocation):
    def __init__(self, root: StrPath):
        self._meta_location: Path = None
        self._pq_location: Path = None
        super().__init__(root, '.pqdir', 'dir')

    @property
    def meta(self) -> StrPath:
        return self._meta_location

    @property
    def pq(self) -> StrPath:
        return self._pq_location

    @property
    def root(self) -> StrPath:
        return super(__class__, __class__).root.fget(self)

    @root.setter
    def root(self, value: StrPath) -> None:
        super(__class__, __class__).root.fset(self, value)
        root = self._root_location
        self._meta_location = root / root.with_suffix('.json').name
        self._pq_location = root / root.with_suffix('.pq').name
        if not (self._pq_location.is_file()
               and self._meta_location.is_file()):
            raise FileNotFoundError(f"The store in {self._root_location} could not be found or is invalid")

    @root.deleter
    def root(self) -> None:
        super(__class__, __class__).root.fdel(self)
        self._meta_location = None
        self._pq_location = None


class _PqTemporaryLocation(_ZarrTemporaryLocation):
    def __init__(self) -> None:
        self._tmp_location = tempfile.TemporaryDirectory(suffix='.pqdir')


class _PqStore(_StoreWrapper[Union["pandas.DataFrame", "cudf.DataFrame"]]):
    def __init__(self, store_location: StrPath, use_cudf: bool = False):
        self.location = _PqLocation(store_location)
        self._store_obj = None
        if use_cudf:
            self._engine = cudf
            self._to_dict = _to_dict_cudf
        else:
            self._engine = pandas
            self._to_dict = _to_dict_pandas

    def update_cache(self,
                     check_properties: bool = False,
                     verbose: bool = True) -> Tuple['OrderedDict[str, int]', Set[str]]:
        cache = CacheHolder()
        group_sizes_df = self._df['group'].value_counts().sort_index()
        cache.group_sizes = OrderedDict(sorted([(k, v) for k, v in self._to_dict(group_sizes_df).items()]))
        cache.properties = set(self._df.columns.tolist()).difference({'group'})
        return cache.group_sizes, cache.properties

    @classmethod
    def make_empty(cls, store_location: StrPath, grouping: str, **kwargs) -> '_Store':
        root = Path(store_location).resolve()
        store_location.mkdir(exist_ok=False)
        meta_location = root / root.with_suffix('.json').name
        pq_location = root / root.with_suffix('.pq').name
        default_engine.DataFrame().to_parquet(pq_location)
        with open(meta_location, 'x') as f:
            json.dump({'grouping': grouping}, f)
        return cls(store_location, **kwargs)

    # File-like
    def open(self, mode: str = 'r', only_meta: bool = False) -> '_Store':
        if not only_meta:
            self._store = self._engine.read_parquet(self._pq_location)
        else:
            class DummyStore:
                pass
            self._store = DummyStore()
        with open(self.location.mota, mode) as f:
            meta = json.loads(f)
        self._store.attrs = meta
        # monkey patch
        self._store.mode = mode
        self._store._is_dirty = False
        self._store._meta_is_dirty = False
        return self

    def close(self) -> '_Store':
        if self._store._is_dirty:
            self._store.to_parquet(self.location.pq)
        if self._store._meta_is_dirty:
            with open(self.location.meta, 'w') as f:
                json.dump(self._store.attrs, f)
        self._store_obj = None
        return self

    # ContextManager
    def __exit__(self, *args, **kwargs) -> None:
        self.close()

    # Mapping
    def __getitem__(self, name: str) -> '_ConformerGroup':
        df_group = self._store[self._store['group'] == name]
        return _PqConformerGroup(df_group, self._meta_location, self._dummy_properties)

    def __delitem__(self, name: str) -> None:
        # Instead of deleting we just reassign the store to everything that is
        # not the requested name here, since this dirties the dataset,
        # only this part will be written to disk on closing
        self._store = self._store[self._store['group'] != name]
        self._store._is_dirty = True

    def __len__(self) -> int:
        return len(self._store['group'].unique())

    def __iter__(self) -> Iterator[str]:
        keys = self._df['group'].unique().tolist()
        keys.sort()
        return iter(keys)


class _PqConformerGroup(_ConformerGroup):
    def __init__(self, group_obj, dummy_properties):
        self._group_obj = group_obj

    # parquet groups are immutable, mutable operations are done directly in the
    # store
    def _is_resizable(self) -> bool:
        return False

    def _append_to_property(self, p: str, data: np.ndarray) -> None:
        raise ValueError("Not implemented for pq groups")

    def move(self, src: str, dest: str) -> None:
        raise ValueError("Not implemented for pq groups")

    def __delitem__(self, k: str) -> None:
        raise ValueError("Not implemented for pq groups")

    # TODO: implement these
    def _getitem_impl(self, p: str) -> np.ndarray:
        pass

    def _len_impl(self) -> int:
        pass

    def _iter_impl(self):
        pass
