from abc import ABC, abstractmethod
import shutil
from pathlib import Path
from typing import ContextManager, MutableMapping, Set, Tuple
from typing_extensions import Protocol
from collections import OrderedDict

import numpy as np

from .._annotations import NumpyConformers, StrPath


class CacheHolder:
    group_sizes: 'OrderedDict[str, int]'
    properties: Set[str]

    def __init__(self) -> None:
        self.group_sizes = OrderedDict()
        self.properties = set()


# _ConformerGroup and _Store are abstract classes from which all backends
# should inherit in order to correctly interact with ANIDataset.  adding
# support for a new backend can be done just by coding these two classes and
# adding the support for the backend inside StoreFactory


# This is kind of like a dict, but with the extra functionality that you can
# directly "append" to it, and rename its keys
class _ConformerGroup(MutableMapping[str, np.ndarray], ABC):
    def _is_resizable(self) -> bool:
        return True

    @abstractmethod
    def _append_to_property(self, p: str, v: np.ndarray) -> None:
        pass

    def append_conformers(self, conformers: NumpyConformers) -> None:
        if self._is_resizable():
            for p, v in conformers.items():
                self._append_to_property(p, v)
            return
        raise ValueError("Can't append conformers, conformer group is not resizable")

    @abstractmethod
    def move(self, src_p: str, dest_p: str) -> None:
        pass


class _ConformerWrapper(_ConformerGroup):
    def __init__(self, data: MutableMapping[str, np.ndarray]) -> None:
        self._data = data

    def __setitem__(self, p: str, v: np.ndarray) -> None:
        self._data[p] = v

    def __delitem__(self, p: str) -> None:
        del self._data[p]

    def __getitem__(self, p: str) -> np.ndarray:
        return self._data[p][()]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        yield from self._data.keys()

    def move(self, src_p: str, dest_p: str) -> None:
        self._data[dest_p] = self._data.pop(src_p)

    def _append_to_property(self, p: str, v: np.ndarray) -> None:
        self._data[p] = np.append(self._data[p], v, axis=0)


class _BaseLocation(Protocol):
    def __set__(self, obj, value: StrPath) -> None:
        pass

    def __get__(self, obj, objtype) -> StrPath:
        pass

    def __delete__(self, obj) -> None:
        pass


class _OnDiskLocation:
    def __init__(self, suffix: str, kind: str = 'file'):
        if kind not in ['file', 'dir']:
            raise ValueError("Kind must be one of 'file' or 'dir'")
        self._kind = kind
        self._suffix = suffix

    def __set__(self, obj, value: StrPath) -> None:
        value = Path(value).resolve()
        if value.suffix == '':
            value = value.with_suffix(self._suffix)
        if value.suffix != self._suffix:
            raise ValueError(f"incorrect location {value}")
        if hasattr(obj, '_store_location'):
            # pathlib.rename() may fail if src and dst are in different filesystems
            shutil.move(obj._store_location, value)
        obj._store_location = value

    def __get__(self, obj, objtype=None) -> StrPath:
        return obj._store_location

    def __delete__(self, obj) -> None:
        if self._kind == 'file':
            obj._store_location.unlink()
        else:
            shutil.rmtree(obj._store_location)
        delattr(obj, '_store_location')


class _Store(ContextManager['_Store'], MutableMapping[str, '_ConformerGroup'], ABC):

    location: _BaseLocation

    def transfer_location_to(self, other_store: '_Store') -> None:
        loc = self.location
        del self.location
        other_store.location = Path(loc).with_suffix('')

    @abstractmethod
    def validate_location(self) -> None:
        pass

    @abstractmethod
    def make_empty(self, grouping: str) -> None:
        pass

    @property
    @abstractmethod
    def mode(self) -> str:
        pass

    @property
    @abstractmethod
    def is_open(self) -> bool:
        pass

    @abstractmethod
    def close(self) -> '_Store':
        pass

    @abstractmethod
    def open(self, mode: str = 'r') -> '_Store':
        pass

    @property
    @abstractmethod
    def grouping(self) -> str:
        pass

    @abstractmethod
    def update_cache(self, check_properties: bool = False, verbose: bool = True) -> Tuple['OrderedDict[str, int]', Set[str]]:
        pass
