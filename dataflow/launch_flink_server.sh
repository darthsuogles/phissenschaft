#!/bin/bash

flink_version=1.6.2

_bsd_="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

docker build "$(mktemp -d)" \
       -t flink-runner \
       -f -<<'_DOCKERFILE_EOF_'
FROM ubuntu:bionic

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
      software-properties-common \
      && \
    add-apt-repository ppa:webupd8team/java && \
    apt-get update -y && \
    apt install -y --no-install-recommends \
      openjdk-8-jdk \
      && \
    rm -rf /var/lib/apt/lists/*

_DOCKERFILE_EOF_

docker run -it \
       --net=host \
       -v "${_bsd_}/prebuilt/flink/${flink_version}":/workspace \
       -w /workspace \
       flink-runner \
       /bin/bash
