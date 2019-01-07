#!/bin/bash

set -eu -o pipefail

# https://github.com/multiarch/qemu-user-static

docker build "$(mktemp -d)" \
       --build-arg HOST_UID="$(id -u)" \
       --build-arg DOCKER_ARCH=arm64v8 \
       --build-arg OS_TARGET_ARCH=arm64 \
       --build-arg TARGET_ARCH=aarch64 \
       -t cross-builder \
       -f -<<'_DOCKERFILE_EOF_'
ARG TARGET_ARCH
ARG DOCKER_ARCH
FROM tensorflow/tensorflow:latest-devel AS TF_CROSS_BUILD_ENV_LAYER

# Running bazel inside a `docker build` command causes trouble, cf:
#   https://github.com/bazelbuild/bazel/issues/134
# The easiest solution is to set up a bazelrc file forcing --batch.
RUN echo "startup --batch" >>/etc/bazel.bazelrc
# Similarly, we need to workaround sandboxing issues:
#   https://github.com/bazelbuild/bazel/issues/418
RUN echo "build --spawn_strategy=standalone --genrule_strategy=standalone" \
    >>/etc/bazel.bazelrc

# Create users
ARG HOST_UID
ENV CONTAINER_USER_ID=${HOST_UID}
ENV CONTAINER_USER_NAME=tensorflow

RUN useradd --shell /bin/bash \
	-u "${CONTAINER_USER_ID}" -o -c "" \
	-m "${CONTAINER_USER_NAME}"
RUN echo "${CONTAINER_USER_NAME}:${CONTAINER_USER_NAME}" | chpasswd
RUN usermod -aG sudo ${CONTAINER_USER_NAME}
RUN mkdir -p /etc/sudoers.d
RUN echo "${CONTAINER_USER_NAME} ALL=(ALL) NOPASSWD: ALL" \
     > "/etc/sudoers.d/${CONTAINER_USER_NAME}"

ENV USER ${CONTAINER_USER_NAME}
ENV HOME /home/"${CONTAINER_USER_NAME}"
RUN chmod a+w /home/"${CONTAINER_USER_NAME}"
RUN chown "${CONTAINER_USER_NAME}" /home/"${CONTAINER_USER_NAME}"

FROM TF_CROSS_BUILD_ENV_LAYER AS TF_CROSS_BUILD_TOOLCHAIN_LAYER

ARG OS_TARGET_ARCH
ENV OS_TARGET_ARCH=${OS_TARGET_ARCH}

RUN add-apt-repository ppa:team-gcc-arm-embedded/ppa
RUN apt-get remove --purge -y gcc-arm-none-eabi || true

RUN dpkg --add-architecture arm64 && dpkg --print-foreign-architectures
RUN touch /etc/apt/sources.list.d/arm.list
RUN echo "deb [arch=$OS_TARGET_ARCH] http://ports.ubuntu.com/ xenial main restricted universe multiverse" \
    | tee -a /etc/apt/sources.list.d/arm.list
RUN echo "deb [arch=$OS_TARGET_ARCH] http://ports.ubuntu.com/ xenial-updates main restricted universe multiverse" \
    | tee -a /etc/apt/sources.list.d/arm.list
RUN echo "deb [arch=$OS_TARGET_ARCH] http://ports.ubuntu.com/ xenial-security main restricted universe multiverse" \
    | tee -a /etc/apt/sources.list.d/arm.list
RUN echo "deb [arch=$OS_TARGET_ARCH] http://ports.ubuntu.com/ xenial-backports main restricted universe multiverse" \
    | tee -a /etc/apt/sources.list.d/arm.list
RUN sed -i 's#deb http://archive.ubuntu.com/ubuntu/#deb [arch=amd64] http://archive.ubuntu.com/ubuntu/#g' /etc/apt/sources.list
RUN sed -i 's#deb http://security.ubuntu.com/ubuntu/#deb [arch=amd64] http://security.ubuntu.com/ubuntu/#g' /etc/apt/sources.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    gfortran \
    crossbuild-essential-${OS_TARGET_ARCH} \
    gcc-arm-embedded \
    gfortran-aarch64-linux-gnu \
    sudo

RUN apt-get update && apt-get install -y --no-install-recommends \
    python-numpy python-pip python-mock \
    python3-numpy python3-pip python3-mock \
    libpython-dev libpython-all-dev libpython3-dev libpython3-all-dev \
    libpython-all-dev:${OS_TARGET_ARCH} \
    libpython3-all-dev:${OS_TARGET_ARCH} \
    zlib1g-dev:${OS_TARGET_ARCH}

RUN rm -rf /var/lib/apt/lists/*

FROM TF_CROSS_BUILD_TOOLCHAIN_LAYER AS THIRD_PARTY_DEPS_LAYER
WORKDIR /opt/third_party
RUN chown -R ${USER} /opt/third_party

USER ${CONTAINER_USER_NAME}
RUN git clone https://github.com/spack/spack.git
RUN git clone https://github.com/xianyi/OpenBLAS openblas

FROM THIRD_PARTY_DEPS_LAYER AS TF_BUILD_LAYER

_DOCKERFILE_EOF_

docker run -it \
       --user="$(id -u)" \
       --privileged \
       --ipc=host \
       -v ${PWD}:/workspace -w /workspace \
       cross-builder \
       /bin/bash
