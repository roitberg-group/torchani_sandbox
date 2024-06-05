#!/bin/bash
set -ex

# Build torchani conda pkg
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

# Build conda pkg, --output is used to output file name
# TODO: does this work or is generating this file manually needed?
# Or maybe just put the build file in a new dir as output of conda build, that
# should be better
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
OUTPUT_DIR="$SCRIPT_DIR/../conda-pkgs/"
mkdir "$OUTPUT_DIR"
conda build \
    -c pytorch \
    -c nvidia \
    -c conda-forge \
    --no-anaconda-upload \
    --output-folder "${OUTPUT_DIR}" \
    "$SCRIPT_DIR/recipe"

# Upload to anaconda.org
if [[ $1 == 'release' ]]; then
    anaconda --token "$CONDA_TOKEN" upload --user roitberg-group --force "${OUTPUT_DIR}/linux-64/sandbox*"
# Upload to internal group server
elif [[ $1 == 'internal' ]]; then
    chown -R 1003:1003 "${OUTPUT_DIR}"
    rsync --archive --verbose --delete \
        -e "ssh -p $SERVER_PORT -o StrictHostKeyChecking=no" \
        "${OUTPUT_DIR}" \
        "$SERVER_USERNAME@roitberg.chem.ufl.edu:/home/statics/conda-packages/"
fi
