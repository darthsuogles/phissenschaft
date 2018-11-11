#!/bin/bash

set -eu -o pipefail

flink_version=1.6.2
flink_tarball="flink-${flink_version}-bin-scala_2.11.tgz"

function fetch_flink_prebuilt {
    local -r apache_mirror_cgi="https://www.apache.org/dyn/closer.lua"
    local -r flink_rel_path="flink/flink-${flink_version}/${flink_tarball}"
    local -r flink_url="${apache_mirror_cgi}?path=${flink_rel_path}"

    curl --silent --location "${flink_url}&asjson=1" \
	| python <(cat <<_PY_JSON_PROC_EOF_
############################################################
import sys, json
pkg_info = json.load(sys.stdin)
print("{}/${flink_rel_path}".format(pkg_info["preferred"]))
############################################################
_PY_JSON_PROC_EOF_
		  ) | xargs curl -fSL -O
}

mkdir -p prebuilt && pushd $_
[[ -f "${flink_tarball}" ]] || fetch_flink_prebuilt
mkdir -p "flink/${flink_version}"
tar -zxvf "${flink_tarball}" \
    -C "flink/${flink_version}" --strip-components=1
popd
