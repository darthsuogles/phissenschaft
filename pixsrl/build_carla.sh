#!/bin/bash

set -eu -o pipefail

_bsd_="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_script_fname="$(readlink -f "${BASH_SOURCE[0]}")"

export UE4_ROOT=/opt/unreal_engine/4.19

if [[ "yes" != "${IS_INSIDE_DOCKER:-no}" ]]; then
    docker ps -a | grep carla-builder-env &>/dev/null || \
	nvidia-docker run -d \
		      -e IS_INSIDE_DOCKER=yes \
		      -v "${_script_fname}":/opt/build_carla.sh \
		      -v "${_bsd_}":/workspace \		      
		      -v /workspace/unreal_engine:/opt/unreal_engine \
		      -e UE4_ROOT=/opt/unreal_engine/4.19 \
		      -w /workspace \
		      --name carla-builder-env \
		      unreal-engine-builder:4.19 \
		      sleep infinity

    docker exec -it carla-builder-env \
	   /usr/local/bin/gosu carla \
	   /opt/build_carla.sh
fi

function install_cmake_impl {
    declare -xr ver="${1}"
    mkdir -p /tmp/cmake && pushd $_
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

function install_cmake {
    local -r ver=3.12.2
    cmake --version | grep "${ver}" || install_cmake_impl "${ver}"
}

function install_dependencies {
    sudo apt install -y --no-install-recommends \
	 rsync \
	 python-dev \
	 python-pip \
	 python3-dev \
	 python3-pip \
	 libboost-python-dev \
	 libtiff5-dev \
	 libpng16-dev \
	 libjpeg-dev \
	 g++-7

    [[ -f /tmp/get-pip.py ]] || \
	curl https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py

    for pcx in python python3; do
	sudo ${pcx} /tmp/get-pip.py
	sudo ${pcx} -m pip install -U pip
	sudo ${pcx} -m pip install -U requests numpy Pillow pygame
    done
    
    install_cmake    
}

install_dependencies

make clean
##############################
make setup
./Update.sh
##############################
make LibCarla
make PythonAPI
make CarlaUE4Editor
##############################
make package
