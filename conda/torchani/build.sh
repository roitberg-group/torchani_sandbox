pip install --no-deps --no-build-isolation -v .
cd ./torchani/csrc || exit 1
cmake -S. -B./build && cmake --build ./build && cmake --install ./build
