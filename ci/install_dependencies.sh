#!/bin/bash

pip install --upgrade pip
pip install twine wheel
pip install --no-cache-dir --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cu111/torch_nightly.html --upgrade --no-deps --force-reinstall
pip install -r test_requirements.txt
pip install -r docs_requirements.txt
