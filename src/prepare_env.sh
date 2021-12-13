#!/bin/bash
apt update
apt -y install software-properties-common
add-apt-repository ppa:deadsnakes/ppa -y
apt -y install python3.9 python3.9-distutils
python3.9 -m pip install six
python3.9 -m pip install --upgrade setuptools pip distlib
python3.9 -m pip install pandas nltk nlpaug torch transformers
git config --global user.name "Tianyi Zhang"
git config --global user.email "tianyi.zhang5@mail.mcgill.ca"