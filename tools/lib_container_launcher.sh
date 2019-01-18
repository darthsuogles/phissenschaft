#!/bin/bash

set -eu -o pipefail

# We mostly follow Google's bash script guide.
# https://google.github.io/styleguide/shell.xml

_DEFAULT_ROOTDIR=/workspace
_DEFAULT_CONTAINER_USER=xpilot
_DEFAULT_WORKSPACE_TAG="git-$(git rev-parse --short HEAD)"

function _derive_envar {

    _lib_container_launcher_bsd_="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    # Find the working directory inside the container
    # We will always launch the
    caller_codebase_root="$(git rev-parse --show-toplevel)"
    caller_codebase_workdir="$(git rev-parse --show-prefix)"

    container_user="${_DEFAULT_CONTAINER_USER}"
    # http://web.archive.org/web/20170606145050/http://wiki.bash-hackers.org/commands/builtin/caller
    _caller_script_name_="$(caller 1 | awk '{print $NF}' | xargs basename)"

    _maybe_valid_prefix="${_caller_script_name_%%-*}"
    [[ "launch" == "${_maybe_valid_prefix}" ]] || \
	    quit_with "launcher script must have a format 'launch-<your-container-name>'"

    # We will use the name of the script to determine the function name
    # The script should be named "launch-XXX" where "XXX" indicates the container's name.
    container_name="${_caller_script_name_#launch-}"

    # TODO: find a better way to derive the true host name
    launching_host="$(hostname -s)"

    cat << _DEBUG_EOF_
LIBRARY DIR: ${_lib_container_launcher_bsd_}
REL WORKDIR: ${caller_codebase_workdir}
CODEBASE: ${caller_codebase_root}
LAUNCHER: ${_caller_script_name_}
LAUNCHED FROM: ${launching_host}
CONTAINER: ${container_name}
_DEBUG_EOF_

}

WORKSPACE_TAG="${_DEFAULT_WORKSPACE_TAG}"
VISIBLE_GPUS="0"

function _parse_opts {
    POSITIONAL=()
    while [[ $# -gt 0 ]]; do
	    local key="$1"

        # Notice that the `shift 2` command will skip the value and the argument
	    case "${key}" in
	        --registry)
		        DOCKER_REGISTRY="$2"
                shift 2;;
	        --image)
		        DOCKER_IMAGE="$2"
                shift 2;;
	        --gpus)
		        VISIBLE_GPUS="$2"
                shift 2;;
	        --workspace)
		        WORKSPACE_TAG="$2"
                shift 2;;
	        --tag)
		        IMAGE_TAG="$2"
                shift 2;;
	        *) # unknown
		        POSITIONAL+=("$1")
                shift 2;;
	    esac
    done

    # Restore positional parameters
    if [[ ${#POSITIONAL[@]} -gt 0 ]]; then
	    set -- "${POSITIONAL[@]}"
    fi
}

function quit_with { >&2 echo "ERROR: $@"; exit 1; }

function _find_free_port {
    local readonly _INIT_PORT=40000
    local readonly _FINI_PORT=65535

    # Find available port
    for port in $(seq ${_INIT_PORT} ${_FINI_PORT}); do
	    if ! (echo >/dev/tcp/0.0.0.0/${port})> /dev/null 2>&1; then
	        unique_ssh_port="${port}"
	        break
        fi
    done
}

function launch_container {
    declare -r DATASETS_ROOT="${_DEFAULT_ROOTDIR}/datasets"
    declare -r MODELS_ROOT="${_DEFAULT_ROOTDIR}/models"
    declare -r PERSISTENT_WORKSPACES_ROOT="${_DEFAULT_ROOTDIR}/workspaces"

    _parse_opts $@

    _derive_envar

    _find_free_port

    local image_name="${DOCKER_REGISTRY}/${DOCKER_IMAGE}:${IMAGE_TAG}"

    docker pull "${image_name}"

    local host_container_state="${PERSISTENT_WORKSPACES_ROOT}/${WORKSPACE_TAG}/${container_name}"
    mkdir -p "${host_container_state}"

    local _tmp_etc_fname="$(mktemp)"
    cat <<_SYS_ENVAR_EOF_ | tee "${_tmp_etc_fname}"

PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
# Added for container
CONTAINER_NAME="${container_name}"
HOST_CODEBASE_WORKDIR="${caller_codebase_workdir}"
HOST_CODEBASE="${caller_codebase_root}"
HOST_DATASETS_ROOT="${DATASETS_ROOT}"
HOST_MODELS_ROOT="${MODELS_ROOT}"
HOST_CONTAINER_STATE="${host_container_state}"

_SYS_ENVAR_EOF_

    nvidia-docker run -it \
                  --pid host \
                  --ipc host \
                  --restart always \
                  -d -e RUN_AS_SERVICE=yes \
                  -e NVIDIA_VISIBLE_DEVICES="${VISIBLE_GPUS}" \
                  -p "${unique_ssh_port}":19284 \
                  -v "${_tmp_etc_fname}":/etc/environment \
                  -v ~/.ssh:"/home/${container_user}/.ssh":ro \
                  -v "${caller_codebase_root}":/host/workspace \
                  -v "${DATASETS_ROOT}/stable":/host/datasets/stable:ro \
                  -v "${DATASETS_ROOT}/incubator":/host/datasets/incubator:rw \
                  -v "${MODELS_ROOT}/incubator":/host/models/incubator:rw \
                  -v "${MODELS_ROOT}/stable":/host/models/stable:ro \
                  -v "${host_container_state}":/host/container_state:rw \
                  -w "/host/workspace/${caller_codebase_workdir}" \
                  --name "${container_name}" \
                  "${image_name}"

    cat <<_INST_EOF_
------------------------------------

Please run the following command to connect

Host inception-${container_name}
  HostName 127.0.0.1
  Port ${unique_ssh_port}
  User ${_DEFAULT_CONTAINER_USER}
  ProxyJump ${launching_host}
  ForwardAgent yes
  StrictHostKeyChecking no
  UserKnownHostsFile /dev/null
  LocalForward 8888 localhost:8888
  LocalForward 6006 localhost:6006
  Compression yes

------------------------------------
_INST_EOF_

}
