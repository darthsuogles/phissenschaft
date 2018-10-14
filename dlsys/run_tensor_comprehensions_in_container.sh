#!/bin/bash

docker build "$(mktemp -d)" \
       -t tensor-comprehensions-builder \
       -f -<<'_DOCKERFILE_EOF_'
FROM continuumio/miniconda

RUN /opt/conda/bin/conda install -y -c pytorch -c tensorcomp tensor_comprehensions

_DOCKERFILE_EOF_

docker run -it \
       tensor-comprehensions-builder \
       /bin/bash
