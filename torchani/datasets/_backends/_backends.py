from pathlib import Path
from typing import ContextManager

from .._annotations import StrPath
from .interface import _StoreAdaptor
from .h5py_impl import _H5PY_AVAILABLE, _H5StoreAdaptor, _H5TemporaryLocation


def infer_backend(store_location: StrPath) -> str:
    if Path(store_location).resolve().suffix == '.h5':
        return 'h5py'
    else:
        raise RuntimeError("Backend could not be infered from store location")


def StoreAdaptorFactory(store_location: StrPath, backend: str) -> '_StoreAdaptor':
    if backend == 'h5py':
        if not _H5PY_AVAILABLE:
            raise ValueError('h5py backend was specified but h5py could not be found, please install h5py')
        return _H5StoreAdaptor(store_location)
    else:
        raise RuntimeError(f"Bad backend {backend}")


def TemporaryLocation(backend: str) -> 'ContextManager[StrPath]':
    if backend == 'h5py':
        return _H5TemporaryLocation()
    else:
        raise ValueError(f"Bad backend {backend}")
