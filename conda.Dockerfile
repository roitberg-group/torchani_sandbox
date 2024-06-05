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

# Copy recipe, scripts and requirements to build conda pkg
COPY conda/recipe/ /repo/conda/recipe/
COPY conda/build_pkg_requirements.txt /repo/conda/

# Install requirements to build conda pkg (first activate conda base env)
#
# Also pre-render meta.yaml from the recipe to save time when actually building
# the package NOTE: This fixes the versions and build-strings of all packages,
# but they change if the cache is invalidated.

# rendered_meta.yaml has to be filtered since the first few lines are
# comments and can't be parsed by conda build as a meta.yaml file
RUN \
    . /opt/conda/etc/profile.d/conda.sh \
    && conda activate \
    && conda update -n base conda \
    && cd conda \
    && conda install --solver libmamba --file build_pkg_requirements.txt

# Copy all of the repo files
COPY . /repo

# Dummy tag
RUN git tag -a "v2.3" -m "Version v2.3"
