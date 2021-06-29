#!/bin/bash

pip install --upgrade pip
pip install twine wheel
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
pip install -r test_requirements.txt
pip install -r docs_requirements.txt
