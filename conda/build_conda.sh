#!/bin/bash
set -ex

# helper functions
script_dir=$(dirname $(realpath $0))
. "$script_dir/pkg_helpers.bash"

# source root dir
export SOURCE_ROOT_DIR="$script_dir/../"

# setup variables
setup_build_version 2.2
setup_cuda_home
setup_conda_pytorch_constraint
setup_conda_cudatoolkit_constraint

export PYTHON_VERSION="3.8"
# build
conda build $CONDA_CHANNEL_FLAGS --no-anaconda-upload --python "$PYTHON_VERSION" "$script_dir/torchani"
# conda install anaconda-client