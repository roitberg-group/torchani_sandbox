# TorchANI Conda Package

## Building with and without CUDA Toolkit in the host in meta.yaml

The `meta.yaml` file assumes that the extensions are built using the system's
CUDA Toolkit, located in `/usr/local/cuda`. If this is not possible then the
next 4 lines should be added, and remove CUDA_HOME=/usr/local/cuda in the build
script.

- pytorch::pytorch-cuda={{ cuda }}
- nvidia::cuda-libraries-dev={{ cuda }}
- nvidia::cuda-nvcc={{ cuda }}
- nvidia::cuda-cccl={{ cuda }}

Note that adding the CUDA Toolkit libraries to the "host" environment
significantly increases the time it takes to build the pkg.

Note also that any changes to `meta.yaml`, `conda_build_config.yaml`,
`filter_rendered_meta.py` or `build_pkg_requirements.txt` will result in a
Docker cache invalidation and consequently the time for CI will be much longer.
