from pathlib import Path
from typing import Dict, Any, Type

from .._annotations import StrPath
from .interface import _Store, _TemporaryLocation
from .h5py_impl import _H5PY_AVAILABLE, _H5Store, _H5TemporaryLocation
from .zarr_impl import _ZARR_AVAILABLE, _ZarrStore, _ZarrTemporaryLocation
from .pq_impl import _PQ_AVAILABLE, _PqStore, _PqTemporaryLocation

# This should probably be obtained directly from getattr
_BACKEND_AVAILABLE = {
    "h5py": _H5PY_AVAILABLE,
    "zarr": _ZARR_AVAILABLE,
    "pq": _PQ_AVAILABLE,
}


class Backend:
    def __init__(
        self,
        name: str,
        suffix: str,
        temporary_location_cls: Type[_TemporaryLocation],
        store_cls: Type[_Store],
    ):
        self._name = name
        self._suffix = suffix
        self._temporary_location_cls = temporary_location_cls
        self._store_cls = store_cls

    def store(
        self,
        location: StrPath,
        grouping: str = None,
        create: bool = False,
        dummy_properties: Dict[str, Any] = None,
        use_cudf: bool = False,
    ) -> "_Store":
        dummy_properties = dict() if dummy_properties is None else dummy_properties
        kwargs: Dict[str, Any] = {"dummy_properties": dummy_properties}

        if create:
            grouping = grouping if grouping is not None else "by_formula"
            store = self._store_cls.make_empty(location, grouping, **kwargs)
        else:
            if grouping is not None:
                raise ValueError("Can't specify a grouping for an existing dataset")
            if self._name == "pq":
                kwargs.update({"use_cudf": use_cudf})
            store = self._store_cls(location, **kwargs)
            setattr(store, "backend", self._name)  # Monkey patch is this needed?
        return store

    def temporary_location(self) -> "_TemporaryLocation":
        return self._temporary_location_cls()


BACKENDS = {
    "h5py": Backend("h5py", ".h5", _H5TemporaryLocation, _H5Store),
    "pq": Backend("pq", ".pq", _PqTemporaryLocation, _PqStore),
    "zarr": Backend("zarr", ".zarr", _ZarrTemporaryLocation, _ZarrStore),
}


def infer_backend(location: StrPath) -> Backend:
    suffix = Path(location).resolve().suffix
    for backend in BACKENDS:
        if backend.suffix == suffix:
            return backend
    raise RuntimeError("Backend could not be infered from store location")
