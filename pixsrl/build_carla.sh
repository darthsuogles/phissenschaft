#!/bin/bash

set -eu -o pipefail

_bsd_="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_script_fname="$(readlink -f "${BASH_SOURCE[0]}")"

BASE_IMAGE=unreal-engine-builder:4.19

function build_carla_docker_image {
    nvidia-docker build "$(mktemp -d)" \
		  --build-arg BASE_IMAGE="${BASE_IMAGE}" \
		  -t carla-builder:unreal-4.19 \
		  -f -<<'_DOCKERFILE_EOF_'
ARG BASE_IMAGE
FROM ${BASE_IMAGE} AS DEPS_PKGS_LAYER

RUN apt-get update -y && apt-get install -y --no-install-recommends \
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

RUN curl https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && \
    for pcx in python python3; do \
	${pcx} /tmp/get-pip.py && \
	${pcx} -m pip install -U pip && \
	${pcx} -m pip install -U \
	  requests \
	  numpy \
	  Pillow \
	  pygame \
	  ; \
    done

WORKDIR /tmp/cmake

RUN set -eux; \
    ver=3.12.2; \
    pkg="cmake-${ver}"; \
    tarball="${pkg}.tar.gz"; \
    curl -fSL -O "https://cmake.org/files/v${ver%.*}/${tarball}" && \
    tar -zxvf "${tarball}" && \
    cd ${pkg} && \
      ./bootstrap && \
      make -j$(nproc) && \
      make install && \
    rm -f /tmp/cmake/${tarball}

FROM DEPS_PKGS_LAYER AS RUNTIME_LAYER

WORKDIR /workspace
ENTRYPOINT ["sleep", "infinity"]
_DOCKERFILE_EOF_
}

if [[ "yes" != "${IS_INSIDE_DOCKER:-no}" ]]; then
    build_carla_docker_image
    
    docker ps -a | grep carla-builder-env &>/dev/null || \
	nvidia-docker run -d \
		      -e IS_INSIDE_DOCKER=yes \
		      -v "${_script_fname}":/opt/build_carla.sh \
		      -v "${_bsd_}":/workspace \
		      -v /workspace/unreal_engine:/opt/unreal_engine \
		      -e UE4_ROOT=/opt/unreal_engine/4.19 \
		      -w /workspace \
		      --name carla-builder-env \
		      carla-builder:unreal-4.19

    docker exec -it carla-builder-env \
	   /usr/local/bin/gosu carla \
	   /opt/build_carla.sh
fi

#make clean
##############################
make setup
./Update.sh
##############################
make LibCarla
make PythonAPI
make CarlaUE4Editor
##############################
make package
