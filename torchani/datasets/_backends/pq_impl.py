import shutil
import json
import tempfile
from pathlib import Path
from typing import Iterator, Set, Union, Tuple, Optional
from collections import OrderedDict

import numpy as np

from .._annotations import StrPath
from .interface import _Store, _ConformerGroup, CacheHolder
from .zarr_impl import _ZarrTemporaryLocation


try:
    import cudf
    _CUDF_AVAILABLE = True
except ImportError:
    _CUDF_AVAILABLE = False

try:
    import pandas
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False


def _to_dict_pandas(df, **kwargs):
    return df.to_dict(df, **kwargs)


def _to_dict_cudf(df, **kwargs):
    return df.to_pandas().to_dict(df, **kwargs)


class _PqTemporaryLocation(_ZarrTemporaryLocation):
    def __init__(self) -> None:
        self._tmp_location = tempfile.TemporaryDirectory(suffix='.zarr')


class _PqStore(_Store):
    def __init__(self, store_location: StrPath, use_cudf: bool = False):
        self._mode: Optional[str] = None
        self._store_obj = None
        self._store_location = Path(store_location).resolve()
        # This actually manages two files inside store_location, a .pq file and
        # a .json metadata file
        self._pq_location, self._meta_location = self._parse_store(self._store_location)
        self._use_cudf = use_cudf
        if use_cudf:
            self._engine = cudf
            self._to_dict = _to_dict_cudf
        else:
            self._engine = pandas
            self._to_dict = _to_dict_pandas

    @staticmethod
    def _parse_store(store_location: Path) -> Tuple[Path, Path]:
        name = store_location.stem
        parquet_store = store_location / f'{name}_store.pq'
        json_meta = store_location / f'{name}_meta.json'
        return parquet_store, json_meta

    def validate_location(self) -> None:
        if (self._store_location.is_dir()
           and self._pq_location.is_file()
           and self._meta_location.is_file()):
            return
        raise FileNotFoundError(f"The store in {self._store_location} could not be found or is invalid")

    def transfer_location_to(self, other_store: '_Store') -> None:
        self.delete_location()
        other_store.location = Path(self.location).with_suffix('')

    @property
    def location(self) -> StrPath:
        return self._store_location

    @location.setter
    def location(self, value: StrPath) -> None:
        value = Path(value).resolve()
        if value.suffix == '':
            value = value.with_suffix('.pq')
        if value.suffix != '.pq':
            raise ValueError(f"Incorrect location {value}")
        _pq_location, _meta_location = self._parse_store(value)
        # pathlib.rename() may fail if src and dst are in different mounts
        value.mkdir()
        shutil.move(self._pq_location, _pq_location)
        shutil.move(self._meta_location, _meta_location)
        self.delete_location()
        self._pq_location = _pq_location
        self._meta_location = _meta_location
        self._store_location = value

    def delete_location(self) -> None:
        shutil.rmtree(self._store_location)

    @property
    def _store(self) -> Union["pandas.DataFrame", "cudf.DataFrame"]:
        if self._store_obj is None:
            raise RuntimeError("Can't access store")
        return self._store_obj

    def update_cache(self,
                     check_properties: bool = False,
                     verbose: bool = True) -> Tuple['OrderedDict[str, int]', Set[str]]:
        cache = CacheHolder()
        group_sizes_df = self._df['group'].value_counts().sort_index()
        cache.group_sizes = OrderedDict(sorted([(k, v) for k, v in self._to_dict(group_sizes_df).items()]))
        cache.properties = set(self._df.columns.tolist()).difference({'group'})
        return cache.group_sizes, cache.properties

    def make_empty(self, grouping: str) -> None:
        self._store_location.mkdir(exist_ok=False)
        self._engine.DataFrame().to_parquet(self._pq_location)
        with open(self._meta_location, 'x') as f:
            json.dump({'grouping': grouping}, f)

    @property
    def grouping(self) -> str:
        with open(self._meta_location, 'r') as f:
            grouping = json.loads(f)['grouping']
        return grouping

    def create_conformer_group(self, name: str) -> '_ConformerGroup':
        self._store.create_group(name)
        return self[name]

    # File-like
    def open(self, mode: str = 'r') -> '_Store':
        self._mode = mode
        self._store_obj = self._engine.read_parquet(self._pq_location)
        return self

    def close(self) -> '_Store':
        if self._is_dirty:
            self._store.to_parquet(self._pq_store)
        self._mode = None
        self._store_obj = None
        return self

    @property
    def is_open(self) -> bool:
        try:
            self._store
        except RuntimeError:
            return False
        return True

    @property
    def mode(self) -> str:
        if self._mode is None:
            raise RuntimeError("Can't access closed store")
        return self._mode

    # ContextManager
    def __enter__(self) -> '_Store':
        return self

    def __exit__(self, *args, **kwargs) -> None:
        self.close()

    # Mapping
    def __getitem__(self, name: str) -> '_ConformerGroup':
        df_group = self._store[self._store['group'] == name]
        return _PqConformerGroup(df_group, self._meta_location)

    def __delitem__(self, k: str) -> None:
        del self._store[k]

    def __len__(self) -> int:
        return len(self._store['group'].unique())

    def __iter__(self) -> Iterator[str]:
        keys = self._df['group'].unique().tolist()
        keys.sort()
        return iter(keys)


class _PqConformerGroup(_ConformerGroup):
    def __init__(self, group_obj):
        self._group_obj = group_obj

    def _append_property_with_data(self, p: str, data: np.ndarray) -> None:
        h5_dataset = self._group_obj[p]
        h5_dataset.resize(h5_dataset.shape[0] + data.shape[0], axis=0)
        try:
            h5_dataset[-data.shape[0]:] = data
        except TypeError:
            h5_dataset[-data.shape[0]:] = data.astype(bytes)

    def _create_property_with_data(self, p: str, data: np.ndarray) -> None:
        # This correctly handles strings and make the first axis resizable
        maxshape = (None,) + data.shape[1:]
        try:
            self._group_obj.create_dataset(name=p, data=data, maxshape=maxshape)
        except TypeError:
            self._group_obj.create_dataset(name=p, data=data.astype(bytes), maxshape=maxshape)

    def move(self, src: str, dest: str) -> None:
        self._group_obj.move(src, dest)

    # Mapping
    def __delitem__(self, k: str) -> None:
        del self._group_obj[k]

    def __getitem__(self, p: str) -> np.ndarray:
        array = self._group_obj[p][()]
        assert isinstance(array, np.ndarray)
        return array

    def __len__(self) -> int:
        return len(self._group_obj)

    def __iter__(self) -> Iterator[str]:
        yield from self._group_obj.keys()
