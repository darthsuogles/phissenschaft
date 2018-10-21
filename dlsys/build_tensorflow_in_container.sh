#!/bin/bash

set -eu -o pipefail

_bsd_="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cat <<_SCRIPT_LOCATION_WARNING_EOF_
==========================================
Please make sure this script is located
under the root directory of tensorflow.
==========================================
_SCRIPT_LOCATION_WARNING_EOF_

BASE_CONTAINER_IMAGE=tensorflow/tensorflow:latest-devel-gpu-py3

#docker pull "${BASE_CONTAINER_IMAGE}"

nvidia-docker build "$(mktemp -d)" \
	      --build-arg BASE_CONTAINER_IMAGE="${BASE_CONTAINER_IMAGE}" \
	      -t tensorflow-builder:base \
	      -f -<<'_DOCKERFILE_EOF_'
ARG BASE_CONTAINER_IMAGE
FROM ${BASE_CONTAINER_IMAGE}

FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-10-0 \
        cuda-cublas-dev-10-0 \
        cuda-cudart-dev-10-0 \
        cuda-cufft-dev-10-0 \
        cuda-curand-dev-10-0 \
        cuda-cusolver-dev-10-0 \
        cuda-cusparse-dev-10-0 \
        curl \
        git \
        libnccl2 \
        libnccl-dev \
        libcurl3-dev \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libpng-dev \
        libzmq3-dev \
	openjdk-8-jdk \
	python3-dev \
        pkg-config \
        rsync \
        software-properties-common \
	swig \
        unzip \
        zip \
        zlib1g-dev \
        wget \
        && \
    find /usr/local/cuda-10.0/lib64/ -type f -name 'lib*_static.a' -not -name 'libcudart_static.a' -delete && \
    rm /usr/lib/x86_64-linux-gnu/libcudnn_static_v7.a

RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

RUN apt-get update && \
        apt-get install nvinfer-runtime-trt-repo-ubuntu1804-5.0.0-rc-cuda10.0 \
        && apt-get update \
        && apt-get install -y --no-install-recommends libnvinfer5=5.0.0-1+cuda10.0 \
        && apt-get install libnvinfer-dev=5.0.0-1+cuda10.0 \
        && rm -rf /var/lib/apt/lists/*

# Link NCCL libray and header where the build script expects them.
RUN mkdir /usr/local/cuda-10.0/lib &&  \
    ln -s /usr/lib/x86_64-linux-gnu/libnccl.so.2 /usr/local/cuda/lib/libnccl.so.2 && \
    ln -s /usr/include/nccl.h /usr/local/cuda/include/nccl.h

# Install bazel
RUN echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list && \
    curl https://bazel.build/bazel-release.pub.gpg | apt-key add - && \
    apt-get update && \
    apt-get install -y bazel

# Running bazel inside a `docker build` command causes trouble, cf:
#   https://github.com/bazelbuild/bazel/issues/134
# The easiest solution is to set up a bazelrc file forcing --batch.
RUN echo "startup --batch" >>/etc/bazel.bazelrc
# Similarly, we need to workaround sandboxing issues:
#   https://github.com/bazelbuild/bazel/issues/418
RUN echo "build --spawn_strategy=standalone --genrule_strategy=standalone" \
    >>/etc/bazel.bazelrc

# Configure the build for our CUDA configuration.
ENV CI_BUILD_PYTHON python3
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ENV TF_NEED_CUDA 1
ENV TF_NEED_TENSORRT 1
ENV TF_CUDA_COMPUTE_CAPABILITIES=3.5,5.2,6.0,6.1,7.0
ENV TF_CUDA_VERSION=10.0
ENV TF_CUDNN_VERSION=7

ENV GOSU_VERSION 1.11
RUN set -eux; \
# save list of currently installed packages for later so we can clean up
	apt-get update; \
	apt-get install -y --no-install-recommends ca-certificates wget; \
	if ! command -v gpg; then \
		apt-get install -y --no-install-recommends gnupg2 dirmngr; \
	fi; \
	rm -rf /var/lib/apt/lists/*; \
	\
	dpkgArch="$(dpkg --print-architecture | awk -F- '{ print $NF }')"; \
	wget -O /usr/local/bin/gosu "https://github.com/tianon/gosu/releases/download/$GOSU_VERSION/gosu-$dpkgArch"; \
	\
	chmod +x /usr/local/bin/gosu; \
# verify that the binary works
	gosu --version; \
	gosu nobody true

RUN apt-get update -y && apt-get install -y --no-install-recommends \
        curl \
        git \
        rsync \
        wget \
	sudo \
        && rm -rf /var/lib/apt/lists/*

_DOCKERFILE_EOF_

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

# Create build directories shared by host and container
sudo mkdir -p /tf_build/cache /tf_build/output
sudo chown -R "${USER}" /tf_build

docker rm -f tensorflow-builder-env &>/dev/null || true

nvidia-docker run -d \
	      --env CONTAINER_USER_NAME=tensorflow \
	      --env CONTAINER_USER_ID="$(id -u)" \
	      --volume "${_bsd_}":/workspace \
	      --volume /tf_build:/tf_build \
	      --workdir /workspace \
	      --name tensorflow-builder-env \
	      tensorflow-builder:base \
	      /workspace/._prepare_and_await.sh


touch ._build_tensorflow_impl.sh && chmod +x $_

cat <<'_BUILD_TENSORFLOW_EOF_' | tee ._build_tensorflow_impl.sh
#!/bin/bash

set -eu -o pipefail

sudo ln -fsn /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:"${LD_LIBRARY_PATH:-.}"

tensorflow/tools/ci_build/builds/configured GPU

function bazel_build {
    bazel \
	--output_base=/tf_build/output \
	build \
	-c opt \
	--copt=-mavx \
	--config=cuda \
	--disk_cache=/tf_build/cache \
	--cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" \
	$@
}

bazel_build tensorflow/tools/pip_package:build_pip_package

sudo rm /usr/local/cuda/lib64/stubs/libcuda.so.1

bazel-bin/tensorflow/tools/pip_package/build_pip_package /workspace/pip
pip --no-cache-dir install --upgrade /workspace/pip/tensorflow-*.whl
rm -rf /root/.cache

_BUILD_TENSORFLOW_EOF_

cat <<_RUN_BUILD_INST_EOF_
==========================================
Please run your build with this command

docker exec -it tensorflow-builder-env /usr/local/bin/gosu tensorflow bash
==========================================
_RUN_BUILD_INST_EOF_
