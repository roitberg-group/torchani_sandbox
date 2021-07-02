import warnings
import numpy as np
from typing import ContextManager, Iterator, Mapping, Optional, Dict, Any
from ._annotations import NumpyConformers

try:
    import h5py  # noqa F401
    _H5PY_AVAILABLE = True
    H5Group = h5py.Group
    H5Dataset = h5py.Dataset
    H5File = h5py.File
except ImportError:
    warnings.warn('Currently the only supported backend for ANIDataset is h5py, very limited options are available otherwise.'
                  ' installing h5py (pip install h5py or conda install h5py) is recommended if you want to use '
                  ' the torchani.datasets module')
    _H5PY_AVAILABLE = False
    H5Group = Any
    H5Dataset = Any
    H5File = Any


def _DatasetStoreAdaptor(store_location: str, mode: str, backend: str, property_alias: Optional[Dict[str, str]] = None):
    if backend == 'h5py':
        return _H5DatasetStoreAdaptor(h5py.File(store_location, mode), property_alias)
    else:
        raise RuntimeError(f"Bad backend {backend}")


# Wrapper around an open hdf5 file object that returns ConformerGroup facades
# which renames properties on access and creation
class _H5DatasetStoreAdaptor(ContextManager['_H5DatasetStoreAdaptor'], Mapping[str, '_H5ConformerGroupAdaptor']):
    def __init__(self, store_obj: H5File, alias_to_storename: Optional[Dict[str, str]] = None):
        # an open h5py file object
        self._store_obj = store_obj
        self._alias_to_storename = alias_to_storename if alias_to_storename is not None else dict()
        self.mode = self._store_obj.mode

    def _set_grouping(self, grouping: str) -> None:
        self._store_obj.attrs['grouping'] = grouping

    @property
    def grouping(self) -> str:
        try:
            data = self._store_obj.attrs['grouping']
        except (KeyError, OSError):
            data = ''
        assert isinstance(data, str)
        return data

    def __delitem__(self, k: str) -> None:
        del self._store_obj[k]

    def create_conformer_group(self, name) -> '_H5ConformerGroupAdaptor':
        self._store_obj.create_group(name)
        return self[name]

    def close(self) -> None:
        self._store_obj.close()

    def __enter__(self) -> '_H5DatasetStoreAdaptor':
        self._store_obj.__enter__()
        return self

    def __exit__(self, *args) -> None:
        self._store_obj.__exit__(*args)

    def __getitem__(self, name) -> '_H5ConformerGroupAdaptor':
        return _H5ConformerGroupAdaptor(self._store_obj[name], self._alias_to_storename)

    def __len__(self) -> int:
        return len(self._store_obj)

    def __iter__(self) -> Iterator[str]:
        return iter(self._store_obj)


class _H5ConformerGroupAdaptor(Mapping[str, np.ndarray]):
    def __init__(self, group_obj: H5Group, alias_to_storename: Optional[Dict[str, str]] = None):
        self._group_obj = group_obj
        self._alias_to_storename = alias_to_storename if alias_to_storename is not None else dict()

    def create_numpy_values(self, conformers: NumpyConformers) -> None:
        for p, v in conformers.items():
            self._create_property_with_data(p, v)

    def append_numpy_values(self, conformers: NumpyConformers) -> None:
        for p, v in conformers.items():
            self._append_property_with_data(p, v)

    def is_resizable(self) -> bool:
        return all(ds.maxshape[0] is None for ds in self._group_obj.values())

    def _append_property_with_data(self, p: str, data: np.ndarray) -> None:
        p = self._alias_to_storename.get(p, p)
        h5_dataset = self._group_obj[p]
        h5_dataset.resize(h5_dataset.shape[0] + data.shape[0], axis=0)
        try:
            h5_dataset[-data.shape[0]:] = data
        except TypeError:
            h5_dataset[-data.shape[0]:] = data.astype(bytes)

    def _create_property_with_data(self, p: str, data: np.ndarray) -> None:
        # this correctly handles strings (species and _id) and
        # key aliases
        p = self._alias_to_storename.get(p, p)
        # make the first axis resizable
        maxshape = (None,) + data.shape[1:]
        try:
            self._group_obj.create_dataset(name=p, data=data, maxshape=maxshape)
        except TypeError:
            self._group_obj.create_dataset(name=p, data=data.astype(bytes), maxshape=maxshape)

    def move(self, src: str, dest: str) -> None:
        src = self._alias_to_storename.get(src, src)
        dest = self._alias_to_storename.get(dest, dest)
        self._group_obj.move(src, dest)

    def __delitem__(self, k: str) -> None:
        del self._group_obj[k]

    def __getitem__(self, p: str) -> np.ndarray:
        p = self._alias_to_storename.get(p, p)
        array = self._group_obj[p][()]
        assert isinstance(array, np.ndarray)
        return array

    def __len__(self) -> int:
        return len(self._group_obj)

    def __iter__(self) -> Iterator[str]:
        for k in self._group_obj.keys():
            yield self._alias_to_storename.get(k, k)
