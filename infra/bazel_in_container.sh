#!/bin/bash

set -eu -o pipefail

docker build . -t linux-bazel -f -<<'_DOCKERFILE_EOF_'
FROM ubuntu:18.04

ARG PYTHON=python3.6

RUN apt-get update \
    && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        ${PYTHON} \
        python3-distutils \
        libcurl3-dev \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libpng-dev \
        libzmq3-dev \
        libssl-dev \
        pkg-config \
        rsync \
        unzip \
        zip \
        zlib1g-dev \
        openjdk-8-jdk \
        openjdk-8-jre-headless

RUN curl -fsSL -O https://bootstrap.pypa.io/get-pip.py && \
    ${PYTHON} get-pip.py && \
    rm get-pip.py

# Set up Bazel.

# Running bazel inside a `docker build` command causes trouble, cf:
#   https://github.com/bazelbuild/bazel/issues/134
# The easiest solution is to set up a bazelrc file forcing --batch.
RUN echo "startup --batch" >>/etc/bazel.bazelrc
# Similarly, we need to workaround sandboxing issues:
#   https://github.com/bazelbuild/bazel/issues/418
RUN echo "build --spawn_strategy=standalone --genrule_strategy=standalone" \
    >>/etc/bazel.bazelrc
# Install the most recent bazel release.
ENV BAZEL_VERSION=0.17.2
WORKDIR /
RUN mkdir /bazel && \
    cd /bazel && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -o /bazel/LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE && \
    chmod +x bazel-*.sh && \
    ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    cd / && \
    rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh

RUN echo "source /usr/local/lib/bazel/bin/bazel-complete.bash" >> ~/.bashrc
_DOCKERFILE_EOF_

repo_root="$(git rev-parse --show-toplevel)"
repo_workdir="$(git rev-parse --show-prefix)"

docker rm -f linux-bazel-env || true

docker run -d \
       -v /tmp/bazel/:/tmp/bazel \
       -v "${repo_root}":/workspace \
       -w "/workspace/${repo_workdir}" \
       --name linux-bazel-env \
       linux-bazel \
       sleep infinity

function docker_exec {
    docker exec -it linux-bazel-env /bin/bash -c "$@"
}

function bazel_build {
    target="${1:-...}"
    docker_exec "bazel --output_base=/tmp/bazel/build build ${target}"
}

bazel_build
