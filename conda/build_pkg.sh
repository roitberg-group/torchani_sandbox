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


# Set build version
# TODO: It should be possible to do this inside meta.yaml
BUILD_VERSION="2.3.dev$(TZ='America/New_York' date "+%Y%m%d")"

# The following variables are used by meta.yaml too
export BUILD_VERSION

# Build conda pkg, --output is used to output file name
# TODO: does this work or is generating this file manually needed?
# Or maybe just put the build file in a new dir as output of conda build, that
# should be better
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
BUILD_FILE="$(conda build --output --numpy 1.24 --no-anaconda-upload --no-copy-test-source-files "$SCRIPT_DIR/recipe")"
echo "Build file is: $BUILD_FILE"

# Upload to anaconda.org
if [[ $1 == 'release' ]]; then
    anaconda --token "$CONDA_TOKEN" upload --user roitberg-group --force "$BUILD_FILE"
# Upload to internal group server
elif [[ $1 == 'internal' ]]; then
    mkdir -p /release/conda-packages/linux-64
    cp "$BUILD_FILE" /release/conda-packages/linux-64
    conda build purge-all
    conda index /release/conda-packages
    chown -R 1003:1003 /release/conda-packages
    rsync --archive --verbose --delete \
        -e "ssh -p $SERVER_PORT -o StrictHostKeyChecking=no" \
        /release/conda-packages/ \
        "$SERVER_USERNAME@roitberg.chem.ufl.edu:/home/statics/conda-packages/"
fi
