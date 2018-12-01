licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = 'python/ops/_zero_out_ops.so',
    srcs = [
        "cc/kernels/zero_out_kernels.cc",
        "cc/ops/zero_out_ops.cc",
    ],
    linkshared = 1,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
    copts = ["-pthread", "-std=c++11", "-D_GLIBCXX_USE_CXX11_ABI=0"]
)

cc_binary(
    name = 'python/ops/_matrix_add_ops.so',
    srcs = [
        "cc/kernels/matrix_add_kernels.cc",
        "cc/ops/matrix_add_ops.cc",
    ],
    linkshared = 1,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
    copts = ["-pthread", "-std=c++11", "-D_GLIBCXX_USE_CXX11_ABI=0"]
)

py_library(
    name = "custom_ops_lib_py",
    srcs = ([
        "python/ops/custom_ops_lib.py",
    ]),
    data = [
        ":python/ops/custom_ops.so"
    ],
    srcs_version = "PY3ONLY",
)

py_test(
    name = "zero_out_ops_py_test",
    srcs = [
        "python/ops/zero_out_ops_test.py"
    ],
    main = "python/ops/zero_out_ops_test.py",
    deps = [
        ":zero_out_ops_py",
    ],
    srcs_version = "PY3ONLY",
)

py_library(
    name = "zero_out_py",
    srcs = ([
        "__init__.py",
        "python/__init__.py",
        "python/ops/__init__.py",
    ]),
    deps = [
        ":zero_out_ops_py"
    ],
    srcs_version = "PY3ONLY",
)
