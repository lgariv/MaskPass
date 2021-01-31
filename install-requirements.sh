#!/bin/bash
cd models/research
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python setup.py build
python setup.py install
cd ../..
