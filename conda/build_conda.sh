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

# build
conda install conda-build conda-verify anaconda-client -y
anaconda -t ${{ secrets.CONDA_TOKEN }}
conda build $CONDA_CHANNEL_FLAGS "$script_dir/torchani"
