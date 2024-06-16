import typing as tp
import json
from pathlib import Path
from collections import OrderedDict

import numpy as np
import typing_extensions as tpx

from torchani.annotations import StrPath, Grouping, Backend
from torchani.datasets.backends.interface import (
    _Store,
    _ConformerGroup,
    CacheHolder,
)

try:
    import pandas

    _PANDAS_AVAILABLE = True
    default_engine = pandas
except ImportError:
    _PANDAS_AVAILABLE = False

try:
    import cudf

    _CUDF_AVAILABLE = True
    default_engine = cudf
except ImportError:
    _CUDF_AVAILABLE = False


DataFrame = tp.Union["pandas.DataFrame", "cudf.DataFrame"]


class DataFrameAdaptor:
    def __init__(self, df: tp.Optional[DataFrame] = None):
        self._df = df
        self.attrs: tp.Dict[str, tp.Any] = dict()
        self.mode: tp.Optional[str] = None
        self._is_dirty: bool = False
        self._attrs_is_dirty: bool = False

    def __getattr__(self, k):
        if self._df is None:
            raise RuntimeError("Data frame was not opened")
        return getattr(self._df, k)

    def __getitem__(self, k):
        if self._df is None:
            raise RuntimeError("Data frame was not opened")
        return self._df[k]

    def __enter__(self) -> tpx.Self:
        return self

    def __setitem__(self, k, v):
        if self._df is None:
            raise RuntimeError("Data frame was not opened")
        self._df[k] = v


class _ParquetStore(_Store[DataFrame]):
    suffix: str = ".pqdir"
    backend: Backend = "parquet"
    _AVAILABLE: bool = _PANDAS_AVAILABLE

    def __init__(
        self,
        root: StrPath,
        dummy_properties: tp.Optional[tp.Dict[str, tp.Any]] = None,
        grouping: Grouping = "any",
    ):
        super().__init__(root, dummy_properties, grouping)
        self._queued_appends: tp.List[DataFrame] = []
        self._engine = pandas

    @staticmethod
    def _to_numpy(series):
        return series.to_numpy()

    @staticmethod
    def _to_dict(df, **kwargs):
        return df.to_dict(**kwargs)

    @property
    def _pq_path(self) -> Path:
        root = self.location.root()
        return root / root.with_suffix(".pq").name

    @property
    def _attrs_path(self) -> Path:
        root = self.location.root()
        return root / root.with_suffix(".json").name

    @classmethod
    def make_empty(
        cls,
        root: StrPath,
        dummy_properties: tp.Optional[tp.Dict[str, tp.Any]] = None,
        grouping: Grouping = "any",
    ) -> tpx.Self:
        if grouping == "any":
            grouping = "by_num_atoms"
        if grouping not in ("by_num_atoms", "by_formula"):
            raise RuntimeError(f"Invalid grouping for new dataset: {grouping}")
        root = Path(root).resolve()
        root.mkdir(exist_ok=True)
        assert not list(root.iterdir()), "location is not empty"
        attrs_location = root / root.with_suffix(".json").name
        pq_location = root / root.with_suffix(".pq").name
        default_engine.DataFrame().to_parquet(pq_location)
        with open(attrs_location, "x") as f:
            json.dump({"grouping": grouping}, f)
        return cls(root, dummy_properties, grouping)

    def update_cache(
        self, check_properties: bool = False, verbose: bool = True
    ) -> tp.Tuple[tp.OrderedDict[str, int], tp.Set[str]]:
        cache = CacheHolder()
        try:
            group_sizes_df = self._store["group"].value_counts().sort_index()
        except KeyError:
            return cache.group_sizes, cache.properties
        cache.group_sizes = OrderedDict(
            sorted([(k, v) for k, v in self._to_dict(group_sizes_df).items()])
        )
        cache.properties = set(self._store.columns.tolist()).difference({"group"})
        self._dummy_properties = {
            k: v for k, v in self._dummy_properties.items() if k not in cache.properties
        }
        return cache.group_sizes, cache.properties.union(self._dummy_properties)

    # Avoid pickling modules
    def __getstate__(self):
        d = self.__dict__.copy()
        d["_engine"] = self._engine.__name__
        return d

    # Restore modules from names when unpickling
    def __setstate__(self, d):
        if d["_engine"] == "pandas":
            import pandas  # noqa

            d["_engine"] == pandas
        elif d["_engine"] == "cudf":
            import cudf  # noqa

            d["_engine"] == cudf
        else:
            raise RuntimeError("Incorrect _engine value")
        self.__dict__ = d

    # File-like
    def open(self, mode: str = "r", only_attrs: bool = False) -> tpx.Self:
        if not only_attrs:
            self._store_obj = DataFrameAdaptor(
                self._engine.read_parquet(self._pq_path)
            )
        else:
            self._store_obj = DataFrameAdaptor()
        with open(self._attrs_path, mode) as f:
            attrs = json.load(f)
        if "extra_dims" not in attrs.keys():
            attrs["extra_dims"] = dict()
        if "dtypes" not in attrs.keys():
            attrs["dtypes"] = dict()
        self._store_obj.attrs = attrs
        # monkey patch
        self._store_obj.mode = mode
        return self

    def close(self) -> tpx.Self:
        if self._queued_appends:
            self.execute_queued_appends()
        if self._store._is_dirty:
            self._store.to_parquet(self._pq_path)
        if self._store._attrs_is_dirty:
            with open(self._attrs_path, "w") as f:
                json.dump(self._store.attrs, f)
        self._store_obj = None
        return self

    # ContextManager
    def __exit__(self, *args, **kwargs) -> None:
        self.close()

    # Mapping
    def __getitem__(self, name: str) -> "_ConformerGroup":
        df_group = self._store[self._store["group"] == name]
        group = _ParquetConformerGroup(df_group, self._dummy_properties, self._store)
        # mypy does not understand monkey patching
        group._to_numpy = self._to_numpy  # type: ignore
        return group

    def __setitem__(self, name: str, conformers: "_ConformerGroup") -> None:
        num_conformers = conformers[next(iter(conformers.keys()))].shape[0]
        tmp_df = self._engine.DataFrame()
        tmp_df["group"] = self._engine.Series([name] * num_conformers)
        for k, v in conformers.items():
            if v.ndim == 1:
                tmp_df[k] = self._engine.Series(v)
            elif v.ndim == 2:
                tmp_df[k] = self._engine.Series(v.tolist())
            else:
                extra_dims = self._store.attrs["extra_dims"].get(k, None)
                if extra_dims is not None:
                    assert v.shape[2:] == tuple(
                        extra_dims
                    ), "Bad dimensions in appended property"
                else:
                    self._store.attrs["extra_dims"][k] = v.shape[2:]
                tmp_df[k] = self._engine.Series(v.reshape(num_conformers, -1).tolist())
            dtype = self._store.attrs["dtypes"].get(k, None)
            if dtype is not None:
                assert np.dtype(v.dtype).name == dtype, "Bad dtype in appended property"
            else:
                self._store.attrs["dtypes"][k] = np.dtype(v.dtype).name
        self._queued_appends.append(tmp_df)

    def execute_queued_appends(self):
        attrs = self._store_obj.attrs
        mode = self._store_obj.mode
        self._store_obj = DataFrameAdaptor(
            self._engine.concat([self._store._df] + self._queued_appends)
        )
        self._store_obj.attrs = attrs
        self._store_obj.mode = mode
        self._store._is_dirty = True
        self._store._attrs_is_dirty = True
        self._queued_appends = []

    def __delitem__(self, name: str) -> None:
        # Instead of deleting we just reassign the store to everything that is
        # not the requested name here, since this dirties the dataset,
        # only this part will be written to disk on closing
        attrs = self._store_obj.attrs
        mode = self._store_obj.mode
        attrs_is_dirty = self._store_obj._attrs_is_dirty
        self._store_obj = DataFrameAdaptor(self._store[self._store["group"] != name])
        self._store_obj.attrs = attrs
        self._store_obj.mode = mode
        self._store._attrs_is_dirty = attrs_is_dirty
        self._store._is_dirty = True

    # TODO Fix these type ignores
    def __len__(self) -> int:
        return len(self._store["group"].unique())  # type: ignore

    def __iter__(self) -> tp.Iterator[str]:
        keys = self._store["group"].unique().tolist()  # type: ignore
        keys.sort()
        return iter(keys)

    def create_full_direct(
        self, dest_key, is_atomic, extra_dims, fill_value, dtype, num_conformers
    ):
        if is_atomic:
            raise ValueError(
                "creation of atomic properties not supported in parquet datasets"
            )
        if extra_dims:
            extra_dims = (np.asarray(extra_dims).prod()[0],)
        new_property = np.full(
            shape=(num_conformers,) + extra_dims, fill_value=fill_value, dtype=dtype
        )
        self._store.attrs["dtypes"][dest_key] = np.dtype(dtype).name
        if len(extra_dims) > 1:
            self._store.attrs["extra_dims"][dest_key] = extra_dims[1:]
        self._store[dest_key] = self._engine.Series(new_property)
        self._store._attrs_is_dirty = True
        self._store._is_dirty = True

    def rename_direct(self, old_new_dict: tp.Dict[str, str]) -> None:
        self._store.rename(columns=old_new_dict, inplace=True)
        self._store._is_dirty = True

    def delete_direct(self, properties: tp.Iterable[str]) -> None:
        self._store.drop(labels=list(properties), inplace=True, axis="columns")
        if self._store.columns.tolist() == ["group"]:
            self._store.drop(labels=["group"], inplace=True, axis="columns")
        self._store._is_dirty = True


class _CudfParquetStore(_ParquetStore):
    suffix: str = ".pqdir"
    backend: Backend = "cudf"
    _AVAILABLE: bool = _CUDF_AVAILABLE

    def __init__(
        self,
        root: StrPath,
        dummy_properties: tp.Optional[tp.Dict[str, tp.Any]] = None,
        grouping: Grouping = "any",
    ):
        super().__init__(root, dummy_properties, grouping)
        self._queued_appends: tp.List[DataFrame] = []
        self._engine = cudf

    @staticmethod
    def _to_numpy(series):
        return series.to_pandas().to_numpy()

    @staticmethod
    def _to_dict(df, **kwargs):
        return df.to_pandas().to_dict(**kwargs)


class _ParquetConformerGroup(_ConformerGroup):
    def __init__(self, group_obj, dummy_properties, store_pointer):
        super().__init__(dummy_properties=dummy_properties)
        self._group_obj = group_obj
        self._store_pointer = store_pointer

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

    def __setitem__(self, p: str, v: np.ndarray) -> None:
        raise ValueError("Not implemented for pq groups")

    def _getitem_impl(self, p: str) -> np.ndarray:
        # mypy doesn't understand monkey patching
        property_ = np.stack(self._to_numpy(self._group_obj[p]))  # type: ignore
        extra_dims = self._store_pointer.attrs["extra_dims"].get(p, None)
        dtype = self._store_pointer.attrs["dtypes"].get(p, None)
        if extra_dims is not None:
            if property_.ndim == 1:
                property_ = property_.reshape(-1, *extra_dims)
            else:
                property_ = property_.reshape(property_.shape[0], -1, *extra_dims)
        return property_.astype(dtype)

    def _len_impl(self) -> int:
        return len(self._group_obj.columns) - 1

    def _iter_impl(self):
        for c in self._group_obj.columns:
            if c != "group":
                yield c
