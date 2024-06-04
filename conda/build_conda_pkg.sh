#!/bin/bash
set -ex

# Build packages of torchani
#
# To run first install the 'build_conda_pkg_requirements.txt'
#
# Usage:
#
# 1. Build locally
# ./build_conda.sh
#
# 2. Build and upload to internal group server
# CONDA_TOKEN=TOKEN ./build_conda.sh internal
#
# 3. Build and upload to Anaconda server
# CONDA_TOKEN=TOKEN ./build_conda.sh release

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
RECIPE_PATH="$SCRIPT_DIR/torchani"

# Set build version
VERSION_PREFIX=2.3
BUILD_VERSION="${VERSION_PREFIX}.dev$(TZ='America/New_York' date "+%Y%m%d")"

# The following variables are used by meta.yaml too
export BUILD_VERSION
export REPO_ROOT="$SCRIPT_DIR/../"
export PYTHON_VERSION=3.10
export PYTORCH_VERSION=2.3
export CUDA_VERSION=11.8

CONDA_ROOT="$(conda info --base)"

# Build conda pkg
conda build --no-anaconda-upload --no-copy-test-source-files "$RECIPE_PATH"

BUILD_FILE="${CONDA_ROOT}/conda-bld/linux-64/${PACKAGE_NAME}-${BUILD_VERSION}-py${PYTHON_VERSION//./}_torch${PYTORCH_VERSION}_cuda${CUDA_VERSION}.tar.bz2"
echo "Build file is: $BUILD_FILE"

# Upload to anaconda.org
if [[ $1 == 'release' ]]; then
    anaconda \
        --token "$CONDA_TOKEN" \
        upload \
            --user roitberg-group \
            --force "$BUILD_FILE"
# Upload to internal group server
elif [[ $1 == 'internal' ]]; then
    mkdir -p /release/conda-packages/linux-64
    cp "$BUILD_FILE" /release/conda-packages/linux-64
    conda build purge-all
    conda index /release/conda-packages
    chown -R 1003:1003 /release/conda-packages
    rsync \
        --archive \
        --verbose \
        --delete \
        -e "ssh -p $SERVER_PORT -o StrictHostKeyChecking=no" \
        /release/conda-packages/ \
        "$SERVER_USERNAME@roitberg.chem.ufl.edu:/home/statics/conda-packages/"
fi
