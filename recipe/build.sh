export CUDA_HOME=/usr/local/cuda
python -m pip install --no-deps --no-build-isolation -v .
cd ./torchani/csrc/ || exit 1
mkdir ./build
cmake -DTORCH_CUDA_ARCH_LIST='6.0;6.1;7.0;7.5;8.0;8.6' -S. -B./build/
cmake --build ./build
cmake --install ./build
