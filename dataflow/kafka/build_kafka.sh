#!/bin/bash

set -eu -o pipefail

ver=2.1.0

_bsd_="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

_KAFKA_PREBUILT_ROOT="${_bsd_}/._kafka_actual"

mkdir -p "${_KAFKA_PREBUILT_ROOT}" && pushd $_

tarball="kafka_2.11-${ver}.tgz"
[[ -f "${tarball}" ]] || \
    wget "http://mirror.reverse.net/pub/apache/kafka/${ver}/${tarball}"

mkdir "${ver}"
tar -zxvf "${tarball}" -C "${ver}" --strip-components=1

popd
