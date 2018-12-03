#!/bin/bash

set -eu -o pipefail

# https://github.com/multiarch/qemu-user-static

# Register QEMU as a handler for non-x86 targets
# the `--reset` option removes all registered binfmt_misc before
docker container run --rm --privileged multiarch/qemu-user-static:register --reset

docker build "$(mktemp -d)" \
       --build-arg HOST_UID="$(id -u)" \
       --build-arg DOCKER_ARCH=arm64v8 \
       --build-arg TARGET_ARCH=aarch64 \
       -t cross-builder \
       -f -<<'_DOCKERFILE_EOF_'
ARG TARGET_ARCH
ARG DOCKER_ARCH
FROM multiarch/alpine:${TARGET_ARCH}-latest-stable AS BOOTSTRAP
FROM ${DOCKER_ARCH}/ubuntu:16.04 AS SYSROOT
ARG TARGET_ARCH
COPY --from=BOOTSTRAP /usr/bin/qemu-${TARGET_ARCH}-static /usr/bin/qemu-${TARGET_ARCH}-static

RUN apt-get update && apt-get install -y --no-install-recommends \
    lsb-release \
    libarmadillo-dev \
    libcurl4-openssl-dev \
    libeigen3-dev \
    libopenblas-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libopencv-dev \
    curl \
    ca-certificates \
    sudo

RUN apt-get update && apt-get install -y \
    software-properties-common \
    && \
    add-apt-repository ppa:deadsnakes/ppa

RUN apt-get update && apt-get install -y \
    python3.6 \
    python3.6-dev \
    python3.7 \
    python3.7-dev \
    zip \
    build-essential \
    unzip

RUN apt-get purge -y python python-dev python3 python3-dev

RUN rm -rf /var/lib/apt/lists/*

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

RUN add-apt-repository ppa:team-gcc-arm-embedded/ppa
RUN apt-get remove -y gcc-arm-none-eabi || true

RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    gfortran \
    crossbuild-essential-arm64 \
    gcc-arm-embedded \
    gfortran-aarch64-linux-gnu \
    sudo

# # TODO copy toolchain files from other place
# ARG TARGET_ARCH
# ENV CROSS_BUILD_SYSROOT /sysroot/crosstool-${TARGET_ARCH}
# COPY --from=SYSROOT /lib ${CROSS_BUILD_SYSROOT}/lib
# COPY --from=SYSROOT /usr/include ${CROSS_BUILD_SYSROOT}/usr/include
# COPY --from=SYSROOT /usr/lib ${CROSS_BUILD_SYSROOT}/usr/lib
# COPY --from=SYSROOT /usr/share/pkgconfig ${CROSS_BUILD_SYSROOT}/usr/share/pkgconfig
# COPY --from=SYSROOT /usr/local ${CROSS_BUILD_SYSROOT}/usr/local
# COPY --from=SYSROOT /opt ${CROSS_BUILD_SYSROOT}/opt
# COPY --from=SYSROOT /etc/alternatives ${CROSS_BUILD_SYSROOT}/etc/alternatives

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
