#!/bin/bash
cd models/research
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python3 setup.py build
python3 setup.py install
sudo python3 -m pip install .
cd ../..
