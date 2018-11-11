#!/bin/bash

BASE_IMAGE=ros:melodic-perception-bionic

TMP_BUILD_CONTEXT="$(mktemp -d)"

cat <<'_ENTRYPOINT_EOF_' | tee ${TMP_BUILD_CONTEXT}/entrypoint.sh
#!/bin/bash

set -eux -o pipefail

declare -xr CONTAINER_USER_ID
declare -xr CONTAINER_USER_NAME
declare -xr ROS_DISTRO

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

chmod a+x ${TMP_BUILD_CONTEXT}/entrypoint.sh

nvidia-docker build "${TMP_BUILD_CONTEXT}" \
	      --build-arg BASE_IMAGE="${BASE_IMAGE}" \
	      -t ros-perception \
	      -f -<<'_DOCKERFILE_EOF_'
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    gnupg2 \
    dirmngr \
    ca-certificates \
    curl \
    aria2 \
    wget \
    sudo \
    && \
    rm -rf /var/lib/apt/lists/*

ENV GOSU_VERSION 1.11
RUN set -eux; \
	dpkgArch="$(dpkg --print-architecture | awk -F- '{ print $NF }')"; \
	wget -O /usr/local/bin/gosu "https://github.com/tianon/gosu/releases/download/$GOSU_VERSION/gosu-$dpkgArch"; \
	wget -O /usr/local/bin/gosu.asc "https://github.com/tianon/gosu/releases/download/$GOSU_VERSION/gosu-$dpkgArch.asc"; \
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

RUN touch /etc/bashrc && \
    echo source "/opt/ros/$ROS_DISTRO/setup.bash" >> /etc/bashrc

COPY entrypoint.sh /opt/service_entrypoint.sh
ENTRYPOINT ["/opt/service_entrypoint.sh"]

_DOCKERFILE_EOF_

docker rm -f ros-perception-env &>/dev/null || true

nvidia-docker run -d \
	      -v $PWD:/workspace \
	      -w /workspace \
	      -e CONTAINER_USER_NAME="baroque" \
	      -e CONTAINER_USER_ID="$(id -u)" \
	      --name ros-perception-env \
	      ros-perception

printf "Container launched, wait for a while till things are settled ... "
sleep 4
printf "done\n"

nvidia-docker exec -it \
	      ros-perception-env \
	      /usr/local/bin/gosu baroque \
	      /bin/bash $@
	      
