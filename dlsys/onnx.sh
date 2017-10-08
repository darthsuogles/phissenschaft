#!/bin/bash

docker run -it --rm \
       -v $PWD:/workspace/$USER \
       -w /workspace/$USER \
       onnx/onnx-docker:cpu \
       /bin/bash
