#!/bin/bash
cd models/research
protoc object_detection/protos/*.proto --python_out=.
python setup.py build
python setup.py installcd ../..
