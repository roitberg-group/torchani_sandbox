from pathlib import Path
from ..datasets import ANIDataset


def h5info(path):
    path_ = Path(path)
    assert path_.exists(), f"{path} does not exist"
    if path_.is_file():
        files = [path_]
    else:
        files = list(Path(path_).glob("*.h5"))
        assert len(files) > 0, f"no h5 files found at {path_}"
    names = [f.stem for f in files]
    ds = ANIDataset(locations=files, names=names)
    print(ds)
    groups = list(ds.keys())
    conformer = ds.get_conformers(groups[0], 0)
    print('\nFirst Conformer Properties (Non-batched): ')
    key_max_len = max([len(k)for k in conformer.keys()]) + 3
    for k, value in conformer.items():
        print(f'  {k.ljust(key_max_len)} shape: {str(list(value.shape)).ljust(10)} dtype: {value.dtype}')
