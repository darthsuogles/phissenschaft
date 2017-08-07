load("@io_bazel_rules_docker//docker:docker.bzl", "docker_build")

java_binary(
    name = "runner",
    main_class = "ArgsMain",
    srcs = glob(["java_cli/*.java"]),
    deps = ["@google_guava//jar", "@beust_jcommander//jar"],
)

docker_build(
    name = "app",
    # References docker_pull from WORKSPACE (above)
    base = "@java_base//image",
    files = [":runner_deploy.jar"],
    cmd = ["/runner_deploy.jar"],
)
