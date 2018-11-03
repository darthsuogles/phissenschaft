#!/bin/bash

set -eu -o pipefail

BASE_IMAGE=carla-simulator:latest
CONTAINER=carla-simulator-client

nvidia-docker build "$(mktemp -d)" \
	     -t "${CONTAINER}:latest" \
	     --build-arg BASE_IMAGE="${BASE_IMAGE}" \
	     -f -<<'_DOCKERFILE_EOF_'
ARG BASE_IMAGE
FROM ${BASE_IMAGE} AS OS_PKGS_LAYER

USER root

RUN apt-get update -y && apt-get install -y --no-install-recommends \
	 rsync \
	 curl \
	 python3-dev \
	 python3-pip \
	 libtiff5-dev \
	 libpng16-dev \
	 libjpeg-dev \
	 && \
	 ln -fsn /usr/bin/python3 /usr/bin/python \
	 && \
	 rm -rf /var/lib/apt/lists/*

FROM OS_PKGS_LAYER AS PYTHON_PKGS_LAYER

RUN curl -fSL -o /tmp/get-pip.py -O https://bootstrap.pypa.io/get-pip.py && \
    python /tmp/get-pip.py && \
    rm -f /tmp/get-pip.py

RUN python -m pip install -U pip && \
    python -m pip install -U \
      requests \
      Pillow \
      numpy \
      pygame

FROM PYTHON_PKGS_LAYER AS RUNTIME_LAYER

USER carla
ENTRYPOINT ["sleep", "infinity"]

_DOCKERFILE_EOF_

docker rm -f "${CONTAINER}" &>/dev/null || true

docker run -d \
       --runtime=nvidia \
       --net=host \
       --ipc=host \
       --privileged \
       -v /tmp/.X11-unix:/tmp/.X11-unix \
       -v $PWD:/workspace \
       -w /workspace \
       -e DISPLAY="${DISPLAY:-1}" \
       -e SDL_VIDEODRIVER=x11 \
       -e NVIDIA_VISIBLE_DEVICES=1 \
       --name "${CONTAINER}" \
       "${CONTAINER}:latest"

docker exec -it "${CONTAINER}" /bin/bash
