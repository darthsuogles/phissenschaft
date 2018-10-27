import sys
import os
import ctypes
from random import randint

from PIL import Image
import numpy as np
import tensorflow as tf

import pycuda.driver as cuda
import pycuda.autoinit

import tensorrt as trt
import graphsurgeon as gs
import uff

# ../common.py
sys.path.insert(1,
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        os.pardir
    )
)
import common

# lenet5.py
import lenet5


# MNIST dataset metadata
MNIST_IMAGE_SIZE = 28
MNIST_CHANNELS = 1
MNIST_CLASSES = 10

# Path where clip plugin library will be built (check README.md)
CLIP_PLUGIN_LIBRARY = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'build/libclipplugin.so'
)

# Path to which trained model will be saved (check README.md)
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'models/trained_lenet5.pb'
)

# Define global logger object (it should be a singleton,
# available for TensorRT from anywhere in code).
# You can set the logger severity higher to suppress messages
# (or lower to display more messages)
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# Define some global constants about the model.
class ModelData(object):
    INPUT_NAME = "InputLayer"
    INPUT_SHAPE = (MNIST_CHANNELS, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE)
    RELU6_NAME = "ReLU6"
    OUTPUT_NAME = "OutputLayer/Softmax"
    OUTPUT_SHAPE = (MNIST_IMAGE_SIZE, )
    DATA_TYPE = trt.float32


# Generates mappings from unsupported TensorFlow operations to TensorRT plugins
def prepare_namespace_plugin_map():
    # In this sample, the only operation that is not supported by TensorRT
    # is tf.nn.relu6, so we create a new node which will tell UffParser which
    # plugin to run and with which arguments in place of tf.nn.relu6.


    # The "clipMin" and "clipMax" fields of this TensorFlow node will be parsed by createPlugin,
    # and used to create a CustomClipPlugin with the appropriate parameters.
    trt_relu6 = gs.create_plugin_node(name="trt_relu6", op="CustomClipPlugin", clipMin=0.0, clipMax=6.0)

    # In order to use TensorRT plugins, you need to create a mapping from
    # TensorFlow node name to the plugin node. If you wish to map an entire
    # namespace to a plugin, provide the TensorFlow namespace name instead.
    namespace_plugin_map = {
        ModelData.RELU6_NAME: trt_relu6
    }
    return namespace_plugin_map

# Transforms model path to uff path (e.g. /a/b/c/d.pb -> /a/b/c/d.uff)
def model_path_to_uff_path(model_path):
    uff_path = os.path.splitext(model_path)[0] + ".uff"
    return uff_path

# Converts the TensorFlow frozen graphdef to UFF format using the UFF converter
def model_to_uff(model_path):
    # Transform graph using graphsurgeon to map unsupported TensorFlow
    # operations to appropriate TensorRT custom layer plugins
    dynamic_graph = gs.DynamicGraph(model_path)
    dynamic_graph.collapse_namespaces(prepare_namespace_plugin_map())
    # Save resulting graph to UFF file
    output_uff_path = model_path_to_uff_path(model_path)
    uff.from_tensorflow(
        dynamic_graph.as_graph_def(),
        [ModelData.OUTPUT_NAME],
        output_filename=output_uff_path,
        text=True,
        quiet=True
    )
    return output_uff_path

# Builds TensorRT Engine
def build_engine(model_path):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
        builder.max_workspace_size = common.GiB(1)

        uff_path = model_to_uff(model_path)
        parser.register_input(ModelData.INPUT_NAME, ModelData.INPUT_SHAPE)
        parser.register_output(ModelData.OUTPUT_NAME)
        parser.parse(uff_path, network)

        return builder.build_cuda_engine(network)

# Loads a test case into the provided pagelocked_buffer. Returns loaded test case label.
def load_normalized_test_case(pagelocked_buffer):
    _, _, x_test, y_test = lenet5.load_data()
    num_test = len(x_test)
    case_num = randint(0, num_test-1)
    img = x_test[case_num].ravel()
    np.copyto(pagelocked_buffer, img)
    return y_test[case_num]

def main():
    # Load the shared object file containing the Clip plugin implementation.
    # By doing this, you will also register the Clip plugin with the TensorRT
    # PluginRegistry through use of the macro REGISTER_TENSORRT_PLUGIN present
    # in the plugin implementation. Refer to plugin/clipPlugin.cpp for more details.
    if not os.path.isfile(CLIP_PLUGIN_LIBRARY):
        raise IOError("\n{}\n{}\n{}\n".format(
            "Failed to load library ({}).".format(CLIP_PLUGIN_LIBRARY),
            "Please build the Clip sample plugin.",
            "For more information, see the included README.md"
        ))
    ctypes.CDLL(CLIP_PLUGIN_LIBRARY)

    # Load pretrained model
    if not os.path.isfile(MODEL_PATH):
        raise IOError("\n{}\n{}\n{}\n".format(
            "Failed to load model file ({}).".format(MODEL_PATH),
            "Please use 'python lenet5.py' to train and save the model.",
            "For more information, see the included README.md"
        ))

    # Build an engine and retrieve the image mean from the model.
    with build_engine(MODEL_PATH) as engine:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        with engine.create_execution_context() as context:
            print("\n=== Testing ===")
            test_case = load_normalized_test_case(inputs[0].host)
            print("Loading Test Case: " + str(test_case))
            # The common do_inference function will return a list of outputs - we only have one in this case.
            [pred] = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            print("Prediction: " + str(np.argmax(pred)))


if __name__ == "__main__":
    main()
