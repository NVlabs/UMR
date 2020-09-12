# -----------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# Code written by Xueting Li.
# -----------------------------------------------------------

#!/bin/bash

echo "Installing python packages"
pip install -r requirements.txt

echo "Installing SoftRas"
cd "./external/SoftRas"
python setup.py install

echo "Installing Neural Mesh Renderer Pytorch"
cd "../"
git clone https://github.com/daniilidis-group/neural_renderer
cd "./neural_renderer"
python setup.py install
cd "../.."

echo "Done"
