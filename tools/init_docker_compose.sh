#!/bin/bash

version=1.23.0
url_prefix="https://github.com/docker/compose/releases/download/${version}"

[[ -f "/tmp/docker-compose-${version}" ]] || \
    curl -fSL \
	 -o "/tmp/docker-compose-${version}" \
	 -O "${url_prefix}/docker-compose-$(uname -s)-$(uname -m)"

chmod a+x "/tmp/docker-compose-${version}"
sudo cp "/tmp/docker-compose-${version}" /usr/local/bin/
sudo ln -s \
     "/usr/local/bin/docker-compose-${version}" \
     "/usr/local/bin/docker-compose"
