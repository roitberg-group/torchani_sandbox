from pathlib import Path
import h5py
import numpy as np


class ANIDataset:

    def __init__(self, store_file):

        if isinstance(store_file, str):
            store_file = Path(store_file).resolve()
        assert isinstance(store_file, Path)
        if not store_file.is_file():
            raise RuntimeError(f"The h5 file in {store_file.as_posix()} could not be found")

        self.store_file = store_file
        self.group_paths = set()

        with h5py.File(self.store_file, 'r') as f:
            f.visititems(self._cache_group_paths)

    def __len__(self):
        return len(self.group_paths)

    def __iter__(self):
        return self

    def __next__(self):
        # Iterate over group paths and yield the associated molecule groups as
        # dictionaries of numpy arrays (except for species, which is a list of
        # strings)
        with h5py.File(self.store_file, 'r') as f:
            for group_path in self.group_paths:
                molecules = dict()
                for k, v in f[group_path].items():
                    molecules[k] = np.asarray(v[()])
                    if molecules[k].dtype == np.bytes_:
                        molecules[k] = [s.decode('ascii') for s in molecules[k]]
                return molecules

    def _cache_group_paths(self, name, object_):
        # validate format of the dataset
        if isinstance(object_, h5py.Dataset):
            molecule_group = object_.parent
            for k, v in molecule_group.items():
                if not isinstance(v, h5py.Dataset):
                    msg = "Invalid dataset format, there shouldn't be Groups inside Groups with Datasets"
                    raise RuntimeError(msg)

            # if the format is correct cache the molecule group path in a set
            self.group_paths.update({molecule_group.name})


# dataset = ANIDataset('/home/ignacio/Datasets/ani1x_release_wb97x_dz.h5')
