#!/bin/bash
set -ex

# USAGE:
# Build packages of torchani
# 1. test
# ./build_conda.sh
# 2. upload to group server
# CONDA_TOKEN=TOKEN ./build_conda.sh release-group
# 3. upload to anaconda server
# CONDA_TOKEN=TOKEN ./build_conda.sh release-anaconda

script_dir="$(dirname "$(realpath "$0")")"

# set build version
VERSION_PREFIX=2.2
BUILD_VERSION="${VERSION_PREFIX}.dev$(TZ='America/New_York' date "+%Y%m%d")$VERSION_SUFFIX"
# The following variables are used by meta.yaml
export BUILD_VERSION
export SOURCE_ROOT_DIR="$script_dir/../"
export PACKAGE_NAME=sandbox
export PYTHON_VERSION=3.10
export PYTORCH_VERSION=2.3
export CUDA_VERSION=11.8
export NUMPY_CONSTRAINT=1.24

CONDA="$(conda info --base)"

# anaconda bin location
export PATH="$PATH:${CONDA_PREFIX}/bin:${CONDA}/bin"
which anaconda

# build package
conda build \
    -c pytorch \
    -c nvidia \
    -c conda-forge \
    --no-anaconda-upload \
    --no-copy-test-source-files \
    --python "$PYTHON_VERSION" \
    "$script_dir/torchani"
BUILD_FILE="${CONDA}/conda-bld/linux-64/${PACKAGE_NAME}-${BUILD_VERSION}-py${PYTHON_VERSION//./}_torch${PYTORCH_VERSION}_cuda${CUDA_VERSION}.tar.bz2"
echo "Build file: $BUILD_FILE"

# Upload to anaconda.org if release-anaconda is the arg
if [[ $1 == 'release-anaconda' ]]; then
    anaconda -t "$CONDA_TOKEN" upload -u roitberg-group "$BUILD_FILE" --force
# Upload to roitberg server if release-group is the arg
elif [[ $1 == 'release-group' ]]; then
    mkdir -p /release/conda-packages/linux-64
    cp "$BUILD_FILE" /release/conda-packages/linux-64
    rm -rf "${CONDA}/conda-bld/*"
    conda index /release/conda-packages
    chown -R 1003:1003 /release/conda-packages
    apt update && apt install rsync -y
    rsync -av --delete -e "ssh -p $SERVER_PORT -o StrictHostKeyChecking=no" /release/conda-packages/ "$SERVER_USERNAME@roitberg.chem.ufl.edu:/home/statics/conda-packages/"
fi
