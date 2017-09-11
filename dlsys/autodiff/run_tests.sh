#!/bin/bash

_bsd_="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pushd ${_bsd_}
PYTHONPATH=$PWD python3 -m nose -v test/test_autodiff.py
popd
