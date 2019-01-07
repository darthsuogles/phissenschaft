#!/bin/bash

set -eu -o pipefail

UNREAL_ENGINE_VERSION=4.19
UNREAL_ENGINE_ROOT=/workspace/unreal_engine
UNREAL_ENGINE_PATH="${UNREAL_ENGINE_ROOT}/${UNREAL_ENGINE_VERSION}"

[[ -d "${UNREAL_ENGINE_PATH}" ]] || \
    git clone --depth=1 \
	    -b "${UNREAL_ENGINE_VERSION}" \
	    https://github.com/EpicGames/UnrealEngine.git \
	    "${UNREAL_ENGINE_PATH}"

function rebuild_unreal_engine_image {

    nvidia-docker build "$(mktemp -d)" \
		          -t unreal-engine-builder:base \
		          -f -<<'_DOCKERFILE_EOF_'

FROM nvidia/opengl:1.0-glvnd-devel-ubuntu16.04

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    sed \
    sudo \
    curl \
    wget \
    unzip \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN add-apt-repository ppa:ubuntu-toolchain-r/test

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    clang-5.0 \
    lld-5.0 \
    g++-7 \
    ninja-build \
    tzdata \
    python \
    python-pip \
    python-dev \
    autoconf \
    libtool \
    apt-transport-https \
    && update-alternatives --install /usr/bin/clang++ clang++ /usr/lib/llvm-5.0/bin/clang++ 101 \
    && update-alternatives --install /usr/bin/clang clang /usr/lib/llvm-5.0/bin/clang 101 \
    && rm -rf /var/lib/apt/lists/*

ENV GOSU_VERSION 1.10
RUN set -ex; \
	dpkgArch="$(dpkg --print-architecture | awk -F- '{ print $NF }')"; \
	wget -O /usr/local/bin/gosu "https://github.com/tianon/gosu/releases/download/$GOSU_VERSION/gosu-$dpkgArch"; \
	wget -O /usr/local/bin/gosu.asc "https://github.com/tianon/gosu/releases/download/$GOSU_VERSION/gosu-$dpkgArch.asc"; \
	\
# verify the signature
	export GNUPGHOME="$(mktemp -d)"; \
	gpg --keyserver ha.pool.sks-keyservers.net --recv-keys B42F6819007F00F88E364FD4036A9C25BF357DD4; \
	gpg --batch --verify /usr/local/bin/gosu.asc /usr/local/bin/gosu; \
	rm -r "$GNUPGHOME" /usr/local/bin/gosu.asc; \
	\
	chmod +x /usr/local/bin/gosu; \
# verify that the binary works
	gosu nobody true

RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 3FA7E0328081BFF6A14DA29AA6A19B38D3D831EF \
    && echo "deb https://download.mono-project.com/repo/ubuntu stable-xenial main" \
    | tee /etc/apt/sources.list.d/mono-official-stable.list \
    && apt-get update -y && apt-get install -y --no-install-recommends \
       shared-mime-info

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

sleep infinity

_ENTRYPOINT_EOF_

    docker rm -f unreal-engine-prebuilder >&/dev/null || true

    nvidia-docker run -d \
		          --name unreal-engine-prebuilder \
		          --env CONTAINER_USER_ID="$(id -u)" \
		          --env CONTAINER_USER_NAME=carla \
		          --volume "${UNREAL_ENGINE_ROOT}":/opt/unreal_engine \
		          --volume "${PWD}":/workspace \
		          unreal-engine-builder:base \
		          /workspace/._prepare_and_await.sh

    printf "Wait for a while prepared ... "
    sleep 7
    printf "done\n"

    nvidia-docker exec -it \
		          unreal-engine-prebuilder \
		          /usr/local/bin/gosu carla /bin/bash \
		          "/opt/unreal_engine/${UNREAL_ENGINE_VERSION}/Setup.sh"

    docker commit unreal-engine-prebuilder \
	       "unreal-engine-builder:${UNREAL_ENGINE_VERSION}"

    docker rm -f unreal-engine-prebuilder >&/dev/null || true
}

function init_unreal_engine_builder {
    docker rm -f unreal-engine-builder-env >&/dev/null || true

    nvidia-docker run -d \
		          --name unreal-engine-builder-env \
		          --volume "${UNREAL_ENGINE_ROOT}":/opt/unreal_engine \
		          --volume "${PWD}":/workspace \
		          --workdir "/opt/unreal_engine/${UNREAL_ENGINE_VERSION}" \
		          "unreal-engine-builder:${UNREAL_ENGINE_VERSION}" \
		          sleep infinity
}

[[ "yes" == "${REBUILD_IMAGE:-no}" ]] && rebuild_unreal_engine_image

init_unreal_engine_builder

function unreal_exec {
    nvidia-docker exec -it unreal-engine-builder-env \
		          /usr/local/bin/gosu carla $@
}

unreal_exec ./GenerateProjectFiles.sh

unreal_exec make
