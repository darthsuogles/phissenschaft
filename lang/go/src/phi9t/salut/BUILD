genrule(
    name = "extern_buildifier",
    srcs = ["@golang_bazel_buildtools//buildifier"],
    outs = ["buildifier"],
    cmd = "cp $(location @golang_bazel_buildtools//buildifier) $@",
    executable = True,
    output_to_bindir = True,
)
