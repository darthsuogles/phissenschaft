#!/bin/bash

set -euo pipefail

function quit_with { >&2 echo "ERROR: $@"; exit; }

[[ $# == 1 ]] || quit_with "Error: must provide problem ID"

problem_id="$1"
src_fname="leetcode_${problem_id}.cxx"
[[ -f $src_fname ]] || quit_with "Error: cannot find source code ${src_fname}."

exec_name="_leet_${problem_id}"
g++ -std=c++1z -o "${exec_name}" -g -O3 "${src_fname}" \
    && /usr/bin/time "./${exec_name}" \
    && rm -f "./${exec_name}"
