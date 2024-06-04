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
COPY conda/*.py /repo/conda/

# Install requirements to build conda pkg (first activate conda base env)
#
# Also pre-render meta.yaml from the recipe to save time when actually building
# the package NOTE: This fixes the versions and build-strings of all packages,
# but they change if the cache is invalidated.
#
# rendered_meta.yaml has to be filtered since the first few lines are
# comments and can't be parsed by conda build as a meta.yaml file
#
# rendered_meta.yaml is converted into a meta_environment.yaml afterwards,
# so that packages can be pre-installed and afterwards found in the cache
#
# Afterwards the environment is updated with the required packages so that later conda
# can find them in the cache
#
# TODO: Make sure that conda-build actually uses the cache of the pre-installed
# environment
RUN \
    . /opt/conda/etc/profile.d/conda.sh \
    && conda activate \
    && cd conda \
    && conda install --solver libmamba --file build_pkg_requirements.txt \
    && conda render -c nvidia -c pytorch -c conda-forge ./recipe > rendered_meta.yaml \
    && python filter_rendered_meta.py \
    && python meta_environment_from_rendered_meta.py \
    && conda env update --solver libmamba --file meta_environment.txt

# Copy all of the repo files
COPY . /repo

# Overwrite meta.yaml with the rendered meta
# NOTE: This must be done after copying the whole repo since that
# copy overwrites meta.yaml again
RUN mv conda/rendered_meta.yaml conda/recipe/meta.yaml

# Init repo from scratch, faster than copying .git
# setuptools-scm needs a Git repo to work properly
RUN \
    git config --global user.email "user@domain.com" \
    && git config --global user.name "User" \
    && git config --global init.defaultBranch "main" \
    && git init > /dev/null \
    && git add . \
    && git commit -m "Initial commit" > /dev/null
