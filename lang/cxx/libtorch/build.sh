#!/bin/bash

mkdir -p ._cmake_build_tree && pushd $_

cmake .. -DCMAKE_PREFIX_PATH="${PWD}/../libtorch"

make -j4

popd
