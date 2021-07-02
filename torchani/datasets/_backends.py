import warnings
try:
    import h5py
    _H5PY_AVAILABLE = True
except ImportError:
    warnings.warn('Currently the only supported backend for ANIDataset is h5py, very limited options are available otherwise.'
                  ' installing h5py (pip install h5py or conda install h5py) is recommended if you want to use '
                  ' the torchani.datasets module')
    _H5PY_AVAILABLE = False

__all__ = ['h5py']
