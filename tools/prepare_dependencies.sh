#!/bin/bash

set -eu -o pipefail

function init_conda {
    mkdir -p /workspace/conda && pushd $_
    [[ -f miniconda.sh ]] || \
	curl -o miniconda.sh \
	     -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    chmod +x miniconda.sh && \
	./miniconda.sh -u -b -p "${PWD}"
    popd
}

function add_cuda_compute_repo {
    local distro_name="$(lsb_release -si | tr '[:upper:]' '[:lower:]')"
    local distro_version="$(lsb_release -sr | tr -d '.')"
    local arch="$(uname -m)"
    local repo_tag="${distro_name}${distro_version}/${arch}"
    local repo_url_prefix=https://developer.download.nvidia.com/compute/machine-learning/repos
    local repo_url="${repo_url_prefix}/${repo_tag}"
    
    sudo apt-key adv --fetch-keys "${repo_url}/7fa2af80.pub"
    local pkgs=(
	nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
	nvinfer-runtime-trt-repo-ubuntu1804-5.0.0-rc-cuda10.0_1-1_amd64.deb
    )
    for pkg in ${pkgs[@]}; do
	[[ -f "/tmp/${pkg}" ]] || \
	    curl -fSL -o "/tmp/${pkg}" -O "${repo_url}/${pkg}"
	sudo dpkg -i "/tmp/${pkg}"
    done
}

init_conda
#add_cuda_compute_repo
