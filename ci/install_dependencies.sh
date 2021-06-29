#!/bin/bash

pip install --upgrade pip
pip install twine wheel
# clean pip cache if it's larger than 4GB
file_size=$(du -bs $(pip cache dir) | cut -f1)
if [[ $file_size -gt 4294967296 ]]; then pip cache purge; fi
pip install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cu111/torch_nightly.html --upgrade --no-deps --force-reinstall
pip install -r test_requirements.txt
pip install -r docs_requirements.txt
