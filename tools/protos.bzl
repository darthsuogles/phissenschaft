# Build protobuf

def py_protos(name, protos, visibility = None):
    """\
    Generate all python targets from a protobuf package
    """
    outs = [fname[:-len(".proto")] + "_pb2.py" for fname in protos]
    base_dir = "/".join(outs[0].split("/")[:-1])
    cmds = [
        "mkdir -p {}".format(base_dir),
        "cp $(SRCS) $(@D)/{}".format(base_dir),
        "$(location @com_google_protobuf//:protoc) -I$(@D) $(@D)/{}/*.proto --python_out=$(@D)".format(base_dir),
        "rm -f $(@D)/{}/*.proto".format(base_dir),
        "touch $(@D)/{}/__init__.py".format(base_dir),
    ]
    native.genrule(
        name = name,
        srcs = protos,
        outs = outs,
        cmd = ";".join(cmds),
        tools = ["@com_google_protobuf//:protoc"],
    )
