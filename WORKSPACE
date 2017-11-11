git_repository(
    name = "io_bazel_rules_pex",
    remote = "https://github.com/benley/bazel_rules_pex.git",
    tag = "0.3.0",
)
load("@io_bazel_rules_pex//pex:pex_rules.bzl", "pex_repositories")
pex_repositories()

rules_scala_version="031e73c02e0d8bfcd06c6e4086cdfc7f3a3061a8" # update this as needed

http_archive(
    name = "io_bazel_rules_scala",
    url = "https://github.com/bazelbuild/rules_scala/archive/%s.zip"%rules_scala_version,
    type = "zip",
    strip_prefix= "rules_scala-%s" % rules_scala_version
)

load("@io_bazel_rules_scala//scala:scala.bzl", "scala_repositories")
scala_repositories()


git_repository(
    name = "io_bazel_rules_docker",
    remote = "https://github.com/bazelbuild/rules_docker.git",
    tag = "v0.1.0",
)

load(
    "@io_bazel_rules_docker//docker:docker.bzl",
    "docker_repositories", "docker_pull"
)
docker_repositories()

docker_pull(
    name = "java_base",
    registry = "gcr.io",
    repository = "distroless/java",
    # 'tag' is also supported, but digest is encouraged for reproducibility.
  tag = "latest",
)

docker_pull(
    name = "official_ubuntu",
    registry = "index.docker.io",
    repository = "library/ubuntu",
    tag = "16.04",
)

maven_jar(
    name = "google_guava",
    artifact = "com.google.guava:guava:18.0",
)

maven_jar(
    name = "beust_jcommander",
    artifact = "com.beust:jcommander:1.72",
)
