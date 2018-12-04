# https://gflags.github.io/gflags/#bazel

cc_binary(
    name = "convert_from_velodyne",
    srcs = ["convert_from_velodyne.cc"],
    deps = [
        "@com_github_gflags_gflags//:gflags",
        "@boost//:filesystem",
	"@point_cloud_library//:pcl",
    ],
)
