# This sample uses a Caffe model along with a custom plugin to create a TensorRT engine.
from random import randint
from PIL import Image
import numpy as np

import pycuda.driver as cuda
import pycuda.autoinit

import tensorrt as trt

try:
    from build import fcplugin
except ImportError as err:
    raise ImportError("""ERROR: Failed to import module ({})
Please build the FullyConnected sample plugin.
For more information, see the included README.md
Note that Python 2 requires the presence of `__init__.py` in the build folder""".format(err))

# Allows us to import from common.
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import common

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Define some global constants about the model.
class ModelData(object):
    INPUT_NAME = "input"
    INPUT_SHAPE = (1, 28, 28)
    OUTPUT_NAME = "prob"
    OUTPUT_SHAPE = (10, )
    DTYPE = trt.float32

# Uses a parser to retrieve mean data from a binary_proto.
def retrieve_mean(mean_proto):
    with trt.CaffeParser() as parser:
        return parser.parse_binary_proto(mean_proto)

# For more information on TRT basics, refer to the introductory parser samples.
def build_engine(deploy_file, model_file):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.CaffeParser() as parser:
        builder.max_workspace_size = common.GiB(1)

        # Set the parser's plugin factory. Note that we bind the factory to a reference so
        # that we can destroy it later. (parser.plugin_factory_ext is a write-only attribute)
        fc_factory = fcplugin.FCPluginFactory()
        parser.plugin_factory_ext = fc_factory

        # Parse the model and build the engine.
        model_tensors = parser.parse(deploy=deploy_file, model=model_file, network=network, dtype=ModelData.DTYPE)

        # After parsing, we destroy the plugin. This function is exposed through the binding code in plugin/pyFullyConnected.cpp.
        # The plugin is automatically copied into the engine, and therefore we can safely destroy our copy.
        fc_factory.destroy_plugin()

        network.mark_output(model_tensors.find(ModelData.OUTPUT_NAME))
        return builder.build_cuda_engine(network)

# Loads a test case into the provided pagelocked_buffer.
def load_normalized_test_case(data_path, pagelocked_buffer, mean, case_num=randint(0, 9)):
    test_case_path = os.path.join(data_path, str(case_num) + ".pgm")
    # Flatten the image into a 1D array, normalize, and copy to pagelocked memory.
    img = np.array(Image.open(test_case_path)).ravel()
    np.copyto(pagelocked_buffer, img - mean)
    return case_num

def main():
    # Get data files for the model.
    data_path, [deploy_file, model_file, mean_proto] = common.find_sample_data(description="Runs an MNIST network using a Caffe model file", subfolder="mnist", find_files=["mnist.prototxt", "mnist.caffemodel", "mnist_mean.binaryproto"])

    with build_engine(deploy_file, model_file) as engine:
        # Build an engine, allocate buffers and create a stream.
        # For more information on buffer allocation, refer to the introductory samples.
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        mean = retrieve_mean(mean_proto)
        with engine.create_execution_context() as context:
            case_num = load_normalized_test_case(data_path, inputs[0].host, mean)
            # For more information on performing inference, refer to the introductory samples.
            # The common.do_inference function will return a list of outputs - we only have one in this case.
            [output] = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            pred = np.argmax(output)
            print("Test Case: " + str(case_num))
            print("Prediction: " + str(pred))

if __name__ == "__main__":
    main()
