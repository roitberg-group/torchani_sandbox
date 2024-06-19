import typing as tp
from pathlib import Path

from torchani.annotations import StrPath, Grouping, Backend
from torchani.datasets.backends.interface import _Store
from torchani.datasets.backends.hdf5_impl import _HDF5Store
from torchani.datasets.backends.zarr_impl import _ZarrStore
from torchani.datasets.backends.parquet_impl import _PandasStore, _CudfStore


STORE_TYPE: tp.Dict[Backend, tp.Type[_Store]] = {
    "hdf5": _HDF5Store,
    "zarr": _ZarrStore,
    "pandas": _PandasStore,
    "cudf": _CudfStore,
}

_SUFFIXES: tp.Dict[str, Backend] = {".h5": "hdf5", ".zarr": "zarr", ".pqdir": "pandas"}


def Store(
    # root can be the string "tmp" to create a temporary store
    root: StrPath,
    backend: tp.Optional[Backend] = None,
    grouping: tp.Optional[Grouping] = None,
    dummy_properties: tp.Optional[tp.Dict[str, tp.Any]] = None,
) -> _Store:
    if backend is None:
        backend = _SUFFIXES.get(Path(root).resolve().suffix)
        if backend is None:
            raise RuntimeError("Backend could not be inferred from root suffix")
    if root == "tmp":
        return STORE_TYPE[backend].make_tmp(dummy_properties, grouping)
    if not Path(root).exists():
        return STORE_TYPE[backend].make_new(root, dummy_properties, grouping)
    return STORE_TYPE[backend](root, dummy_properties, grouping)
