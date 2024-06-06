#!/bin/bash

# This script is used to upload conda packages to prevent github from leaking
# secrets (DOCKER_PVTKEY and MORIA_USERNAME) in the log of the workflow files.
# All secrets should be passed as envvars

if [ "$1" = internal ]; then
    echo "${DOCKER_PVTKEY}"  > ~/.ssh/id_rsa.pub && chmod 400 ~/.ssh/id_rsa
    rsync -av --delete ./conda-pkgs/ "${MORIA_USERNAME}@moria.chem.ufl.edu:/data/conda-pkgs/"
elif [ "$1" = public ]; then
    anaconda \
        --token "${CONDA_TOKEN}" \
        upload \
            --user roitberg-group \
            --force ./conda-pkgs/linux-64/*.tar.gz
fi
