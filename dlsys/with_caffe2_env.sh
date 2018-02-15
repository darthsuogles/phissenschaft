#!/bin/bash

docker run -it        -v "/Users/philip/CodeBase/spinnen-krawl/dlsys":/workspace -w /workspace        caffe2ai/caffe2:c2v0.8.0.cpu.full.ubuntu16.04        /bin/bash
