#!/bin/bash

set -eu -o pipefail

_bsd_="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DOCKER_BASE_IMAGE=nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

docker build "$(mktemp -d)" \
       --build-arg DOCKER_BASE_IMAGE="${DOCKER_BASE_IMAGE}" \
       --build-arg PYTHON_VERSION=3.6 \
       -t tensor-comprehensions-builder \
       -f -<<'_DOCKERFILE_EOF_'
ARG DOCKER_BASE_IMAGE
FROM ${DOCKER_BASE_IMAGE} AS OS_BASE_LAYER

RUN rm -f /bin/sh && ln -sfn /bin/bash /bin/sh

RUN apt-get update && apt-get install -y --no-install-recommends \
    make \
    git \
    ssh \
    realpath \
    curl \
    wget \
    unzip \
    cmake \
    libncurses5-dev \
    libz-dev \
    libgmp3-dev \
    automake \
    libtool \
    valgrind \
    subversion \
    ca-certificates \
    software-properties-common

##################################################################################
# Conda
##################################################################################
FROM OS_BASE_LAYER as CONDA_LAYER
ARG PYTHON_VERSION
ENV PYTHON_VERSION=${PYTHON_VERSION}

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda config --add channels nicolasvasilache && \
     /opt/conda/bin/conda config --add channels pytorch && \
     /opt/conda/bin/conda config --add channels conda-forge

RUN /opt/conda/bin/conda install -y -c conda-forge \
      python=${PYTHON_VERSION} \
      numpy \
      pyyaml \
      scipy \
      ipython \
      mkl \
      mkl-include \
      cython \
      typing

RUN /opt/conda/bin/conda install -y -c nicolasvasilache \
      llvm-trunk \
      halide

RUN /opt/conda/bin/conda install -y -c pytorch \
      pytorch=0.4.0 \
      torchvision \
      cuda90 \
      magma-cuda90

RUN /opt/conda/bin/conda remove -y cudatoolkit --force && \
    /opt/conda/bin/conda clean -ya

ENV PATH /opt/conda/bin:$PATH

##################################################################################
# CUB
##################################################################################
FROM CONDA_LAYER AS OSS_DEPS_LAYER
RUN mkdir -p /opt/cuda/
RUN git clone --recursive https://github.com/NVlabs/cub.git /opt/cuda/cub

##################################################################################
# Sanity checks
##################################################################################
FROM CONDA_LAYER AS TMP_TESTING_LAYER
RUN test "$(/opt/conda/bin/conda --version | grep 'conda 4.5')" != "" && echo Found conda 4.5.x as expected
RUN test "$(gcc --version | grep 'Ubuntu 5.4.0')" != "" && echo Found gcc-5.4.0 as expected
RUN nvcc --version
RUN test "$(nvcc --version | grep '9.0')" != "" && echo Found nvcc-9.0 as expected

##################################################################################
# Environment
##################################################################################
FROM OSS_DEPS_LAYER AS RUNTIME_LAYER
ENV CC /usr/bin/gcc
ENV CXX /usr/bin/g++
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib/stubs/:/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH
ENV PATH /usr/local/bin:/usr/local/cuda/bin:$PATH

ENV GOSU_VERSION 1.11
RUN set -ex; \
	dpkgArch="$(dpkg --print-architecture | awk -F- '{ print $NF }')"; \
	curl -fsSL -o /usr/local/bin/gosu \
	  -O "https://github.com/tianon/gosu/releases/download/$GOSU_VERSION/gosu-$dpkgArch"; \
	curl -fsSL -o /usr/local/bin/gosu.asc \
	  -O "https://github.com/tianon/gosu/releases/download/$GOSU_VERSION/gosu-$dpkgArch.asc"; \
	\
# verify the signature
	export GNUPGHOME="$(mktemp -d)"; \
	gpg --keyserver ha.pool.sks-keyservers.net --recv-keys B42F6819007F00F88E364FD4036A9C25BF357DD4; \
	gpg --batch --verify /usr/local/bin/gosu.asc /usr/local/bin/gosu; \
	rm -r "$GNUPGHOME" /usr/local/bin/gosu.asc; \
	chmod +x /usr/local/bin/gosu; \
# verify that the binary works
	gosu nobody true

RUN apt-get update && apt-get install -y --no-install-recommends \
    emacs-nox \
    sudo \
    && \
    rm -rf /var/lib/apt/lists/*

_DOCKERFILE_EOF_

##################################################################################
# ENTRYPOINT
##################################################################################
touch ._prepare_and_await.sh && chmod a+x $_

cat <<'_ENTRYPOINT_EOF_' | tee ._prepare_and_await.sh
#!/bin/bash

set -eux -o pipefail

declare -xr CONTAINER_USER_ID
declare -xr CONTAINER_USER_NAME

echo "Starting with UID : ${CONTAINER_USER_ID}"
useradd --shell /bin/bash \
	-u "${CONTAINER_USER_ID}" -o -c "" \
	-m "${CONTAINER_USER_NAME}"

echo "${CONTAINER_USER_NAME}:${CONTAINER_USER_NAME}" | chpasswd
usermod -aG sudo ${CONTAINER_USER_NAME}
mkdir -p /etc/sudoers.d
echo "${CONTAINER_USER_NAME} ALL=(ALL) NOPASSWD: ALL" \
     > "/etc/sudoers.d/${CONTAINER_USER_NAME}"

export HOME=/home/"${CONTAINER_USER_NAME}"
chmod a+w /home/"${CONTAINER_USER_NAME}"
chown "${CONTAINER_USER_NAME}" /home/"${CONTAINER_USER_NAME}"

# exec /usr/local/bin/gosu "${CONTAINER_USER_NAME}" /bin/bash $@
sleep infinity
_ENTRYPOINT_EOF_

##################################################################################
# Create background container and exec into it
##################################################################################
docker rm -f tensor-comprehensions-builder-env &>/dev/null || true

nvidia-docker run -d \
	      --ipc host \
	      --env CONTAINER_USER_NAME=tnsrcmp \
	      --env CONTAINER_USER_ID="$(id -u)" \
	      --volume "${_bsd_}":/workspace \
	      --workdir /workspace \
	      --name tensor-comprehensions-builder-env \
	      tensor-comprehensions-builder \
	      /workspace/._prepare_and_await.sh

printf "Wait for a while till things are settled ... "
sleep 7
printf "done\n"
docker exec -it tensor-comprehensions-builder-env /usr/local/bin/gosu tnsrcmp bash
