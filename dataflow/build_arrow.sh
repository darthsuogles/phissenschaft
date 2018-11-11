#!/bin/bash

set -eu -o pipefail

cat <<_INSTRUCTIONS_EOF_
Please make sure we are under the root directory of arrow
_INSTRUCTIONS_EOF_

conda create -y -f -q -n pyarrow-dev \
      python=3.6 numpy six setuptools cython pandas pytest \
      cmake flatbuffers rapidjson boost-cpp thrift-cpp snappy zlib \
      gflags brotli jemalloc lz4-c zstd -c conda-forge \
      2>/dev/null || true

export ARROW_BUILD_TYPE=release
export ARROW_BUILD_TOOLCHAIN=$CONDA_PREFIX
export ARROW_HOME=$CONDA_PREFIX
export PARQUET_HOME=$CONDA_PREFIX

mkdir -p cpp/build && pushd $_

cmake -DCMAKE_BUILD_TYPE=$ARROW_BUILD_TYPE \
      -DCMAKE_INSTALL_PREFIX=$ARROW_HOME \
      -DARROW_PARQUET=ON \
      -DARROW_PYTHON=ON \
      -DARROW_PLASMA=ON \
      -DARROW_FLIGHT=ON \
      -DARROW_BUILD_TESTS=OFF \
      ..

make -j4
make install
popd
