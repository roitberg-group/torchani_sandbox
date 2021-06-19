import warnings
import numpy as np
from typing import ContextManager, Iterator, Mapping, Optional, Dict, Any
from ._annotations import NumpyProperties

try:
    import h5py
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

try:
    import cudf  # noqa F401
    _CUDF_AVAILABLE = True
    warnings.warn('cuDF is available but support for this backend is experimental and subject to change without prior warning,'
                  ' currently it is recommended to use cudf/rapids directly, since the support is just a very thin layer that'
                  ' reads .pq files and converts the output to numpy arrays')
    CUDataFrame = cudf.DataFrame
except ImportError:
    warnings.warn('cuDF is not available, to use experimental cudf / rapids support install cudf'
                  ' (conda install -c rapidsai -c nvidia -c numba -c conda-forge cudf=21.06 python=3.7')
    _CUDF_AVAILABLE = False
    CUDataFrame = Any


class _CUDFDatasetStoreAdaptor(ContextManager['_CUDFDatasetStoreAdaptor'], Mapping[str, '_CUDFConformerGroupAdaptor']):
    # Wrapper around an open DataFrame object that
    # returns ConformerGroup facades which renames properties on access and
    # creation, currently the parquet file is kept always in memory as a dataframe, so context managers do nothing
    # NOTE: It is assumed that the keys are saved in a flattened format and that a "shape" key is also saved
    def __init__(self, store_obj: CUDataFrame, property_to_alias: Optional[Dict[str, str]] = None):
        # an open h5py file object
        self._store_obj = store_obj
        self._property_to_alias = property_to_alias if property_to_alias is not None else dict()
        self.mode = 'r+'  # set to read + write so that no call complains
        self._keys = self._store_obj['group_key'].unique().sort_values()

    def __delitem__(self, k: str) -> None:
        raise NotImplementedError("Currently not possible to delete groups using the CUDF backend")

    def create_conformer_group(self, name: str) -> '_CUDFConformerGroupAdaptor':
        raise NotImplementedError("Currently not possible to create groups using the CUDF backend")

    def close(self) -> None:
        pass

    def __enter__(self) -> '_CUDFDatasetStoreAdaptor':
        return self

    def __exit__(self, *args) -> None:
        pass

    def __getitem__(self, name) -> '_CUDFConformerGroupAdaptor':
        df_group = self._store_obj.loc[self._store_obj['group_key'] == name].drop('group_key', axis=1)
        return _CUDFConformerGroupAdaptor(df_group, self._property_to_alias)

    def __len__(self) -> int:
        return len(self._keys.count())

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys.to_pandas())


class _CUDFConformerGroupAdaptor(Mapping[str, np.ndarray]):
    def __init__(self, group_obj: CUDataFrame, property_to_alias: Optional[Dict[str, str]] = None):
        self._group_obj = group_obj.to_pandas().to_dict('list')

        for k in self._group_obj.keys():
            if not k.endswith('shape'):
                try:
                    self._group_obj[k] = np.stack(self._group_obj[k], axis=0)
                    shape = self._group_obj[f'{k}_shape'][0].tolist()
                    self._group_obj[k] = self._group_obj[k].reshape(-1, *shape)
                except KeyError:
                    pass

        self._group_obj = {k: v for k, v in self._group_obj.items() if not k.endswith('shape')}
        self._property_to_alias = property_to_alias if property_to_alias is not None else dict()

    def create_numpy_property(self, p: str, data: np.ndarray) -> None:
        raise NotImplementedError("Creation not implemented for cudf backend")

    def create_numpy_properties(self, properties: NumpyProperties) -> None:
        raise NotImplementedError("Creation not implemented for cudf backend")

    def move(self, src: str, dest: str) -> None:
        raise NotImplementedError("Renaming not implemented for cudf backend")

    def __delitem__(self, k: str) -> None:
        raise NotImplementedError("Deleting not implemented for cudf backend")

    def __getitem__(self, p: str) -> np.ndarray:
        p = self._property_to_alias.get(p, p)
        array = self._group_obj[p]
        assert isinstance(array, np.ndarray)
        return array

    def __len__(self) -> int:
        return len(self._group_obj)

    def __iter__(self) -> Iterator[str]:
        for k in self._group_obj.keys():
            yield self._property_to_alias.get(k, k)


class _H5DatasetStoreAdaptor(ContextManager['_H5DatasetStoreAdaptor'], Mapping[str, '_H5ConformerGroupAdaptor']):
    # Wrapper around an open hdf5 file object that
    # returns ConformerGroup facades which renames properties on access and
    # creation
    def __init__(self, store_obj: h5py.File, property_to_alias: Optional[Dict[str, str]] = None):
        # an open h5py file object
        self._store_obj = store_obj
        self._property_to_alias = property_to_alias if property_to_alias is not None else dict()
        self.mode = self._store_obj.mode

    def __delitem__(self, k: str) -> None:
        del self._store_obj[k]

    def create_conformer_group(self, name) -> '_H5ConformerGroupAdaptor':
        # this wraps create_group
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
        return _H5ConformerGroupAdaptor(self._store_obj[name], self._property_to_alias)

    def __len__(self) -> int:
        return len(self._store_obj)

    def __iter__(self) -> Iterator[str]:
        return iter(self._store_obj)


class _H5ConformerGroupAdaptor(Mapping[str, np.ndarray]):
    def __init__(self, group_obj: h5py.Group, property_to_alias: Optional[Dict[str, str]] = None):
        self._group_obj = group_obj
        self._property_to_alias = property_to_alias if property_to_alias is not None else dict()

    def create_numpy_property(self, p: str, data: np.ndarray) -> None:
        # this wraps create_dataset, correctly handling strings (species and _id) and
        # key aliases
        p = self._property_to_alias.get(p, p)
        try:
            self._group_obj.create_dataset(name=p, data=data)
        except TypeError:
            self._group_obj.create_dataset(name=p, data=data.astype(bytes))

    def create_numpy_properties(self, properties: NumpyProperties) -> None:
        for p, v in properties.items():
            self.create_numpy_property(p, v)

    def move(self, src: str, dest: str) -> None:
        src = self._property_to_alias.get(src, src)
        dest = self._property_to_alias.get(dest, dest)
        self._group_obj.move(src, dest)

    def __delitem__(self, k: str) -> None:
        del self._group_obj[k]

    def __getitem__(self, p: str) -> np.ndarray:
        p = self._property_to_alias.get(p, p)
        array = self._group_obj[p][()]
        assert isinstance(array, np.ndarray)
        return array

    def __len__(self) -> int:
        return len(self._group_obj)

    def __iter__(self) -> Iterator[str]:
        for k in self._group_obj.keys():
            yield self._property_to_alias.get(k, k)
