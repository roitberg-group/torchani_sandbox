#!/bin/bash

pip install --upgrade pip
pip install --no-cache-dir twine wheel
pip install --no-cache-dir --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cu111/torch_nightly.html
pip install --no-cache-dir -r test_requirements.txt
pip install --no-cache-dir -r docs_requirements.txt
