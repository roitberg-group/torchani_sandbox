import h5py
from ._annotations import NdarrayProperties


def _create_numpy_properties_handle_str(group: h5py.Group, numpy_properties: NdarrayProperties) -> None:
    # creates a dataset with dtype bytes if the array is a string array
    for k, v in numpy_properties.items():
        try:
            group.create_dataset(name=k, data=v)
        except TypeError:
            group.create_dataset(name=k, data=v.astype(bytes))
