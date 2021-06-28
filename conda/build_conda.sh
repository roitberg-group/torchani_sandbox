#!/bin/bash
set -ex

# USAGE:
# 1. test
# ./build_conda.sh test
# 2. release
# ./build_conda.sh release CONDA_TOKEN

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

if (( $# == 0 )); then
    >&2 echo "Illegal number of parameters"
    exit 1
fi

# conda-build dependency
conda install conda-build conda-verify anaconda-client -y
export PATH="$PREFIX/envs/foo/bin:$PATH"  # anaconda bin location

# do not upload
if [[ $1 == "test" ]]; then
    conda build $CONDA_CHANNEL_FLAGS --no-anaconda-upload "$script_dir/torchani"
    echo test
fi

# upload to anaconda.org
if [[ $1 == "release" ]]; then
    if (( $# < 2 )); then
        >&2 echo "No conda token provided "
        exit 1
    fi
    CONDA_TOKEN=$2
    anaconda -t $CONDA_TOKEN
    conda build $CONDA_CHANNEL_FLAGS "$script_dir/torchani"
fi
