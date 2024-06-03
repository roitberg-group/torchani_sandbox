#!/bin/bash
set -ex

# Build packages of torchani
#
# To run first install the 'build_conda_pkg_requirements.txt'
#
# Usage:
#
# 0. Check the build recipe
# ./build_conda.sh check
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

# Set build version
VERSION_PREFIX=2.3
BUILD_VERSION="${VERSION_PREFIX}.dev$(TZ='America/New_York' date "+%Y%m%d")"

# The following variables are used by meta.yaml too
export BUILD_VERSION
export PACKAGE_ROOT="$SCRIPT_DIR/../"
export PACKAGE_NAME=sandbox
export PYTHON_VERSION=3.10
export PYTORCH_VERSION=2.3
export CUDA_VERSION=11.8

CONDA_ROOT="$(conda info --base)"

# anaconda bin location
# TODO: not needed
export PATH="$PATH:${CONDA_PREFIX}/bin:${CONDA_ROOT}/bin"
which anaconda

if [[ $1 == 'check' ]]; then
    echo "Checking build recipe for conda pkg"
    conda build --check "$SCRIPT_DIR/torchani"
    exit 0
fi

echo "Building conda pkg"
conda build \
    -c pytorch \
    -c nvidia \
    -c conda-forge \
    --no-anaconda-upload \
    --no-copy-test-source-files \
    "$SCRIPT_DIR/torchani"

BUILD_FILE="${CONDA_ROOT}/conda-bld/linux-64/${PACKAGE_NAME}-${BUILD_VERSION}-py${PYTHON_VERSION//./}_torch${PYTORCH_VERSION}_cuda${CUDA_VERSION}.tar.bz2"
echo "Build file is: $BUILD_FILE"

if [[ $1 == 'release' ]]; then
    echo "Uploading conda pkg to anaconda.org"
    anaconda -t "$CONDA_TOKEN" upload -u roitberg-group "$BUILD_FILE" --force
elif [[ $1 == 'internal' ]]; then
    echo "Uploading to conda pkg internal group server"
    mkdir -p /release/conda-packages/linux-64
    cp "$BUILD_FILE" /release/conda-packages/linux-64
    rm -rf "${CONDA_ROOT}/conda-bld/*"
    conda index /release/conda-packages
    chown -R 1003:1003 /release/conda-packages
    apt update && apt install rsync -y
    rsync \
        -av \
        --delete \
        -e "ssh -p $SERVER_PORT -o StrictHostKeyChecking=no" \
        /release/conda-packages/ \
        "$SERVER_USERNAME@roitberg.chem.ufl.edu:/home/statics/conda-packages/"
fi
