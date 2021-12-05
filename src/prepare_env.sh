apt update
apt install software-properties-common
add-apt-repository ppa:deadsnakes/ppa
apt install python3.9
apt install python3.9-distutils
python3.9 -m pip install six
python3.9 -m pip install --upgrade setuptools
python3.9 -m pip install --upgrade pip
python3.9 -m pip install --upgrade distlib
python3.9 -m pip install pandas nltk nlpaug torch transformers