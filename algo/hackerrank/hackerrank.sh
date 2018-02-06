#!/bin/bash

set -euo pipefail

problem_fname="${1}"

g++ -std=c++1z -O2 -g "${problem_fname}" && ./a.out < input.txt
