import typing as tp
from pathlib import Path

from torchani.annotations import StrPath, Grouping, Backend
from torchani.datasets.backends.interface import _Store
from torchani.datasets.backends.hdf5_impl import _HDF5Store
from torchani.datasets.backends.zarr_impl import _ZarrStore
from torchani.datasets.backends.parquet_impl import _ParquetStore, _CudfParquetStore

STORE_TYPE: tp.Dict[Backend, tp.Type[_Store]] = {
    "hdf5": _HDF5Store,
    "zarr": _ZarrStore,
    "parquet": _ParquetStore,
    "cudf": _CudfParquetStore,
}

_SUFFIXES: tp.Dict[str, Backend] = {".h5": "hdf5", ".zarr": "zarr", ".pqdir": "parquet"}


def Store(
    root: StrPath,
    backend: tp.Optional[Backend] = None,
    grouping: Grouping = "any",
    dummy_properties: tp.Optional[tp.Dict[str, tp.Any]] = None,
    _force_overwrite: bool = False,
) -> _Store:
    if backend is None:
        suffix = Path(root).resolve().suffix
        backend = _SUFFIXES.get(suffix)
        if backend is None:
            raise RuntimeError("Backend could not be inferred from root suffix")

    if not Path(root).exists() or _force_overwrite:
        return STORE_TYPE[backend].make_empty(root, dummy_properties, grouping)
    return STORE_TYPE[backend](root, dummy_properties, grouping)
