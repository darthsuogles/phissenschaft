#!/bin/bash

set -euo pipefail

_bsd_="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

function config_cmake {
    local env_root="$(brew --prefix)"
    local src_tree="${PWD}"

    # https://github.com/caffe2/caffe2/issues/1720
    local PY_EXEC="$(which python3)"
    local PY_INC="$(python3 -c 'from distutils import sysconfig; print(sysconfig.get_python_inc())')"
    local PY_LIB="$(find "$(python3-config --prefix)/lib" -name 'libpython3*.dylib' -maxdepth 1 | head -n1)"

    cmake -DCMAKE_INSTALL_PREFIX="${src_tree}/install_dir" \
          -DCMAKE_PREFIX_PATH="${env_root}" \
          -DPYTHON_INCLUDE_DIR="${PY_INC}" \
          -DPYTHON_LIBRARY="${PY_LIB}" \
          -DPYTHON_EXECUTABLE="${PY_EXEC}" \
          -DUSE_CUDA=OFF \
          -DUSE_LEVELDB=OFF \
          ..
}

pushd "${_bsd_}/caffe2"; mkdir -p build-tree && pushd $_
config_cmake
make clean
make -j8
popd; popd
