# load("//tools:protos.bzl", "py_protos")

package(
    default_visibility = ["//visibility:public"],
)

# py_protos(
#     name = "py_proto_ensemble",
#     protos = glob(["object_detection/protos/*.proto"]),
# )

load("@rules_foreign_cc//tools/build_defs:cmake.bzl", "cmake_external")

cmake_external(
   name = "openblas",
   # Values to be passed as -Dkey=value on the CMake command line;
   # here are serving to provide some CMake script configuration options
   cache_entries = {
       "NOFORTRAN": "on",
       "BUILD_WITHOUT_LAPACK": "no",
   },
   lib_source = "@openblas//:all",

   # We are selecting the resulting static library to be passed in C/C++ provider
   # as the result of the build;
   # However, the cmake_external dependants could use other artefacts provided by the build,
   # according to their CMake script
   static_libraries = ["libopenblas.a"],
)

cmake_external(
   name = "eigen",
   # These options help CMake to find prebuilt OpenBLAS, which will be copied into
   # $EXT_BUILD_DEPS/openblas by the cmake_external script
   cache_entries = {
       "BLA_VENDOR": "OpenBLAS",
       "BLAS_LIBRARIES": "$EXT_BUILD_DEPS/openblas/lib/libopenblas.a",
   },
   headers_only = True,
   lib_source = "@eigen//:all",
   # Dependency on other cmake_external rule; can also depend on cc_import, cc_library rules
   deps = [":openblas"],
)
