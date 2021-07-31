from pathlib import Path
from typing import ContextManager

from .._annotations import StrPath
from .interface import _Store
from .h5py_impl import _H5PY_AVAILABLE, _H5Store, _H5TemporaryLocation
from .zarr_impl import _ZARR_AVAILABLE, _ZarrStore, _ZarrTemporaryLocation

_BACKEND_AVAILABLE = {'h5py': _H5PY_AVAILABLE, 'zarr': _ZARR_AVAILABLE}


def _infer_backend(store_location: StrPath) -> str:
    suffix = Path(store_location).resolve().suffix
    if suffix == '.h5':
        return 'h5py'
    elif suffix == '.zarr':
        return 'zarr'
    raise RuntimeError("Backend could not be infered from store location")


def StoreFactory(store_location: StrPath, backend: str = None) -> '_Store':
    if backend is None:
        backend = _infer_backend(store_location)
    if not _BACKEND_AVAILABLE.get(backend, False):
        raise ValueError(f'{backend} could not be found, please install it if supported.'
                         f' Supported backends are {set(_BACKEND_AVAILABLE.keys())}')
    if backend == 'h5py':
        store = _H5Store(store_location)
    elif backend == 'zarr':
        store = _ZarrStore(store_location)
    store.backend = backend
    return store


def TemporaryLocation(backend: str) -> 'ContextManager[StrPath]':
    if not _BACKEND_AVAILABLE.get(backend, False):
        raise ValueError(f'{backend} could not be found, please install it if supported.'
                         f' Supported backends are {set(_BACKEND_AVAILABLE.keys())}')
    if backend == 'h5py':
        tmp = _H5TemporaryLocation()
    elif backend == 'zarr':
        tmp = _ZarrTemporaryLocation()
    return tmp
