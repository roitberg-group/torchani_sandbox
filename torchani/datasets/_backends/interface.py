from abc import ABC, abstractmethod
from os import fspath
import shutil
from pathlib import Path
from typing import ContextManager, MutableMapping, Set, Tuple, Generic, TypeVar, Optional
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


_MutMapSubtype = TypeVar('_MutMapSubtype', bound=MutableMapping[str, np.ndarray])


class _ConformerWrapper(_ConformerGroup, Generic[_MutMapSubtype]):
    def __init__(self, data: _MutMapSubtype) -> None:
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


class _LocationManager(ABC):
    @property
    def root(self) -> StrPath:
        pass

    @root.setter
    def root(self, value: StrPath) -> None:
        pass

    @root.deleter
    def root(self) -> None:
        pass

    @abstractmethod
    def plain_root(self) -> StrPath:
        pass


class _FileOrDirLocation(_LocationManager):
    def __init__(self, root: StrPath, suffix: str = '', kind: str = 'file'):
        if kind not in ['file', 'dir']:
            raise ValueError("Kind must be one of 'file' or 'dir'")
        self._kind = kind
        self._suffix = suffix
        self._root_location: Optional[Path] = None
        self.root = root

    @property
    def root(self) -> StrPath:
        root = self._root_location
        if root is None:
            raise ValueError("Location is invalid")
        return root

    @root.setter
    def root(self, value: StrPath) -> None:
        value = Path(value).resolve()
        if value.suffix == '':
            value = value.with_suffix(self._suffix)
        if value.suffix != self._suffix:
            raise ValueError(f"incorrect location {value}")
        if self._root_location is not None:
            # pathlib.rename() may fail if src and dst are in different filesystems
            shutil.move(fspath(self._root_location), fspath(value))
        self._root_location = Path(value).resolve()
        self._validate()

    @root.deleter
    def root(self) -> None:
        if self._root_location is not None:
            if self._kind == 'file':
                self._root_location.unlink()
            else:
                shutil.rmtree(self._root_location)
        self._root_location = None

    def plain_root(self) -> StrPath:
        return Path(self.root).with_suffix('')

    def _validate(self) -> None:
        root = Path(self.root)
        _kind = self._kind
        if (_kind == 'dir' and not root.is_dir()
           or _kind == 'file' and not root.is_file()):
            raise FileNotFoundError(f"The store in {root} could not be found")


class _Store(ContextManager['_Store'], MutableMapping[str, '_ConformerGroup'], ABC):
    location: _LocationManager

    def transfer_location_to(self, other_store: '_Store') -> None:
        root = self.location.plain_root()
        del self.location.root
        other_store.location.root = root

    @classmethod
    @abstractmethod
    def make_empty(cls, store_location: StrPath, grouping: str) -> '_Store':
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
