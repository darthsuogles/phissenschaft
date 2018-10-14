#!/bin/bash

set -eu -o pipefail

_bsd_="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

function install_cmake {
    mkdir -p /tmp/cmake && pushd $_
    local ver=3.12.3
    local pkg="cmake-${ver}"
    if [[ ! -d "${pkg}" ]]; then
	local tarball="${pkg}.tar.gz"
	[[ -f "${tarball}" ]] || wget "https://cmake.org/files/v${ver%.*}/${tarball}"
	tar -zxvf "${tarball}"
    fi
    pushd "${pkg}"
    ./bootstrap && make -j$(nproc) && sudo make install
    popd
    popd
}

function install_dependencies {
    install_cmake
    sudo apt install -y rsync
}

install_dependencies

export UE4_ROOT=/opt/unreal_engine/4.19

pushd "${_bsd_}"
make clean
./Update.sh
make LibCarla
make PythonAPI
make CarlaUE4Editor
make package
pod
