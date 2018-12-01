#!/bin/bash

set -eu -o pipefail

_bsd_="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TAG=onnx2trt
TENSORRT_ROOT=/workspace/pkgs/TensorRT-5.0.2.6

nvidia-docker build "${TENSORRT_ROOT}" \
              -t ${TAG}-builder:base \
              -f -<<'_DOCKERFILE_EOF_'
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04 AS OS_BASE_LAYER

RUN apt-get update && apt-get install -y --no-install-recommends \
        autoconf \
        automake \
        libtool \
        pkg-config \
        ca-certificates \
        sudo \
        curl \
        wget \
        git \
        libprotobuf-dev \
        protobuf-compiler \
        cmake \
        swig \
    && rm -rf /var/lib/apt/lists/*

FROM OS_BASE_LAYER AS PYTHON_PKGS_LAYER

ARG PYTHON_VERSION=3.7
ENV PYTHON_VERSION=${PYTHON_VERSION}

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y -c conda-forge \
       python=$PYTHON_VERSION \
       numpy \
       pyyaml \
       scipy \
       mkl \
       mkl-include \
       cython \
       typing \
       && \
     /opt/conda/bin/conda clean -ya

ENV PATH /opt/conda/bin:$PATH

RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN pip --no-cache-dir install \
        Pillow \
        h5py \
        keras_applications \
        keras_preprocessing \
        mock \
        numpy \
        scipy \
        pandas \
        protobuf \
        pyyaml \
        typing

FROM PYTHON_PKGS_LAYER AS TENSORRT_LAYER
COPY . /opt/tensorrt
ENV TENSORRT_ROOT=/opt/tensortt
RUN find /opt/tensorrt -name '*.whl' -exec python -m pip install {} \;

FROM TENSORRT_LAYER AS RUNTIME_LAYER

ENV GOSU_VERSION 1.11
RUN set -eux; \
# save list of currently installed packages for later so we can clean up
	\
	dpkgArch="$(dpkg --print-architecture | awk -F- '{ print $NF }')"; \
	curl -fsSL -o /usr/local/bin/gosu \
      -O "https://github.com/tianon/gosu/releases/download/$GOSU_VERSION/gosu-$dpkgArch"; \
	curl -fsSL -o /usr/local/bin/gosu.asc \
      -O "https://github.com/tianon/gosu/releases/download/$GOSU_VERSION/gosu-$dpkgArch.asc"; \
	\
# verify the signature
	export GNUPGHOME="$(mktemp -d)"; \
# for flaky keyservers, consider https://github.com/tianon/pgp-happy-eyeballs, ala https://github.com/docker-library/php/pull/666
	gpg --keyserver ha.pool.sks-keyservers.net --recv-keys B42F6819007F00F88E364FD4036A9C25BF357DD4; \
	gpg --batch --verify /usr/local/bin/gosu.asc /usr/local/bin/gosu; \
	command -v gpgconf && gpgconf --kill all || :; \
	rm -rf "$GNUPGHOME" /usr/local/bin/gosu.asc; \
	\
	chmod +x /usr/local/bin/gosu; \
# verify that the binary works
	gosu --version; \
	gosu nobody true

_DOCKERFILE_EOF_

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

chown -R "${CONTAINER_USER_NAME}" /opt/conda

# exec /usr/local/bin/gosu "${CONTAINER_USER_NAME}" /bin/bash $@
sleep infinity
_ENTRYPOINT_EOF_

chmod a+x ._prepare_and_await.sh

nvidia-docker run -d \
	      --env CONTAINER_USER_NAME=${TAG} \
	      --env CONTAINER_USER_ID="$(id -u)" \
	      --volume "${_bsd_}":/workspace \
	      --workdir /workspace \
	      --name ${TAG}-builder-env \
	      ${TAG}-builder:base \
	      /workspace/._prepare_and_await.sh

sleep 7
docker exec -it ${TAG}-builder-env /usr/local/bin/gosu ${TAG} /bin/bash
