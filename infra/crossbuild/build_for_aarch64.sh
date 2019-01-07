#!/usr/bin/env bash
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
set -e

_bsd_="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TF_REPO_ROOT="${_bsd_}/tensorflow"
pushd "${TF_REPO_ROOT}"

# The system-installed OpenSSL headers get pulled in by the latest BoringSSL
# release on this configuration, so move them before we build:
if [ -d /usr/include/openssl ]; then
    sudo mv /usr/include/openssl /usr/include/openssl.original
fi

cat <<'_TF_CONFIGURE_EOF_' | tee .tf_configure.bazelrc
build --action_env PYTHON_BIN_PATH="/usr/bin/python3"
build --action_env PYTHON_LIB_PATH="/usr/local/lib/python3.5/dist-packages"
build --python_path="/usr/bin/python3"

build:xla --define with_xla_support=true
build --config=xla

build --config=monolithic

build --define with_jemalloc=true
build --define with_gcp_support=false
build --define with_hdfs_support=false
build --define with_s3_support=false

build --action_env TF_NEED_OPENCL_SYCL="0"
build --action_env TF_NEED_AWS="0"
build --action_env TF_NEED_KAFKA="0"
build --action_env TF_NEED_ROCM="0"
build --action_env TF_NEED_CUDA="0"
build --action_env TF_DOWNLOAD_CLANG="0"
build --action_env TF_NEED_GCP="0"
build --action_env TF_NEED_HDFS="0"
build --action_env TF_NEED_S3="0"
build --action_env TF_NEED_IGNITE="0"

build:opt --copt=-marchh=armv8-a
build:opt --copt=-marchh=-O3
build:opt --host_copt=-march=native
build:opt --define with_default_optimizations=true
build:v2 --define=tf_api_version=2
_TF_CONFIGURE_EOF_

bazel build \
      --cpu=aarch64 \
      -c opt \
      --crosstool_top=//tools/aarch64_compiler:toolchain \
      --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
      --verbose_failures \
      //tensorflow:libtensorflow.so \
      //tensorflow:libtensorflow_framework.so

popd
