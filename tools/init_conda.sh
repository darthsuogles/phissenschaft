#!/bin/bash

repo_root="$(git rev-parse --show-toplevel)"
conda_pkg_url=https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh

[[ -f install_miniconda.sh ]] ||
    curl -fsSL "${conda_pkg_url}" -o install_miniconda.sh
chmod +x install_miniconda.sh
./install_miniconda.sh -b -u -p "${repo_root}/._conda"
source "${repo_root}/._conda/bin/activate"
conda create -n phi9t python=3.7
