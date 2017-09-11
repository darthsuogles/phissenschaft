#!/bin/bash

function fetch_lucene {
    local ver=6.5.0
    local tarball="lucene-${ver}.tgz"
    local url="http://download.nextag.com/apache/lucene/java/${ver}/${tarball}"
    [[ -f "${tarball}" ]] || wget "${url}" || curl -LO "${url}"
    tar -zxvf "${tarball}" && rm -f "${tarball}"
}

[[ -d ._lucene ]] || (
    mkdir -p ._lucene && cd $_
    fetch_lucene
)
