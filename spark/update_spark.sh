#!/bin/bash

set -eu

spark_ver=2.2.0
spark_tarball="spark-${spark_ver}-bin-hadoop2.7.tgz"

apache_mirror_cgi="https://www.apache.org/dyn/closer.lua"
spark_rel_path="spark/spark-${spark_ver}/${spark_tarball}"
spark_url="${apache_mirror_cgi}?path=${spark_rel_path}"

python -- << __PY_SCRIPT_EOF__ | xargs wget
import sys, json
json_str = """$(curl --silent --location "${spark_url}&asjson=1")"""
pkg_info = json.loads(json_str)
print("{}/${spark_rel_path}".format(pkg_info["preferred"]))
__PY_SCRIPT_EOF__
