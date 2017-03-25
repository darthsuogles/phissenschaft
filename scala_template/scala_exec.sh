#!/bin/bash

_bsd_="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

sbt_ver=0.13.13

pushd "${_bsd_}"

function fetch_sbt() {
    local tarball="sbt-${sbt_ver}.tgz"
    [[ -f "${tarball}" ]] || (
        local url="https://dl.bintray.com/sbt/native-packages/sbt/${sbt_ver}/${tarball}"
        wget "${url}" || curl -L "${url}"
    )
    mkdir -p "${sbt_ver}" && tar -zxvf "${tarball}" --strip-components=1 -C "$_"
}

[[ -d sbt/latest ]] || (
    mkdir -p sbt && cd $_
    [[ -d "${sbt_ver}" ]] || fetch_sbt
    ln -nfs "${sbt_ver}" latest    
)

while getopts ":b" OPT_CMD; do
      case "${OPT_CMD}" in
          b) CMD_TYPE=sbt
             ;;
          \?) ;;
      esac
done

shift $((OPTIND - 1))

if [[ "${CMD_TYPE}" == "sbt" ]]; then
    exec sbt/latest/bin/sbt $@
fi

popd
