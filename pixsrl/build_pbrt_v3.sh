#!/bin/bash

_bsd_="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

function cmake_pbrt_v3 {
    local src_root="${_bsd_}/pbrt-v3"
    local env_root="$(brew --prefix)"

    # https://cmake.org/cmake/help/latest/module/FindZLIB.html
    cmake -DCMAKE_INSTALL_PREFIX="${src_root}/install_root" \
          -DCMAKE_PREFIX_PATH="${env_root}" \
          -DZLIB_ROOT="$(brew --prefix zlib)" \
          ..
}

pushd "${_bsd_}/pbrt-v3"
mkdir -p build-tree && pushd $_
cmake_pbrt_v3
make -j8 install
popd; popd
