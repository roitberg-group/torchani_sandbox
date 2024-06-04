# This image has ubuntu 22.0, cuda 11.8, cudnn 8.7, python 3.10, pytorch 2.3.0
FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel
WORKDIR /repo

# Set cuda env vars
ENV CUDA_HOME=/usr/local/cuda/
ENV PATH=${CUDA_HOME}/bin:$PATH
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install dependencies to:
# Get the program version from version control (git, needed by setuptools-scm)
# Download test data and maybe CUB (wget, unzip)
# Build C++/CUDA extensions faster (ninja-build)
RUN apt update && apt install -y wget git unzip ninja-build

# Copy recipe and requirements to build conda pkg
COPY conda/ /repo/conda/

# Install requirements to build conda pkg (first activate conda base env)
RUN \
    . /opt/conda/etc/profile.d/conda.sh \
    && conda activate \
    && conda install --file conda/build_pkg_requirements.txt

# pre-render meta.yaml from the recipe to save time when actually building the package
# NOTE: This fixes the versions and build-strings of all packages, but they
# change if the cache is invalidated.
# pre-rendered meta.yaml has to be filtered since the first few lines are
# comments and can't be parsed by conda build
RUN \
    . /opt/conda/etc/profile.d/conda.sh \
    && conda activate \
    && cd conda \
    && conda render ./recipe > rendered-meta.yaml \
    && python filter_rendered_meta.py

# TODO: Maybe installing the environment from rendered-meta.yaml here is
# useful since afterwards conda-build can just use the cache?

# Copy all of the repo files
COPY . /repo

# Overwrite meta.yaml with the rendered meta
RUN mv conda/rendered-meta.yaml conda/meta.yaml


# Init repo from scratch, faster than copying .git
# setuptools-scm needs a Git repo to work properly
RUN \
    git config --global user.email "user@domain.com" \
    && git config --global user.name "User" \
    && git config --global init.defaultBranch "main" \
    && git init > /dev/null \
    && git add . \
    && git commit -m "Initial commit" > /dev/null
