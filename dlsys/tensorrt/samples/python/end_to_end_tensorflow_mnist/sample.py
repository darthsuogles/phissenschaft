# This sample uses a UFF MNIST model to create a TensorRT Inference Engine
from pathlib import Path
from random import randint
from PIL import Image
import numpy as np
import timeit

import pycuda.driver as cuda
# This import causes pycuda to automatically manage CUDA context creation and cleanup.
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

import tensorrt as trt

import sys, os
import dlsys.tensorrt.samples.python.common as trt_common

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class ModelData(object):
    #MODEL_FILE = os.path.join(os.path.dirname(__file__), "models/lenet5.uff")
    INPUT_NAME = "input_1"
    INPUT_SHAPE = (1, 28, 28)
    OUTPUT_NAME = "dense_1/Softmax"


def build_engine(model_file):
    # For more information on TRT basics, refer to the introductory samples.
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network() as network, \
         trt.UffParser() as parser:

        builder.max_workspace_size = trt_common.GiB(1)
        # Parse the Uff Network
        parser.register_input(ModelData.INPUT_NAME, ModelData.INPUT_SHAPE)
        parser.register_output(ModelData.OUTPUT_NAME)
        parser.parse(model_file, network)
        # Build and return an engine.
        return builder.build_cuda_engine(network)


def test_simple_pycuda():
    kernel_src = SourceModule("""
    __global__ void multiply_them(float *dest, float *a, float *b)
    {
      const int i = threadIdx.x;
      dest[i] = a[i] * b[i];
    }
    """)
    multiply_them = kernel_src.get_function("multiply_them")

    a = np.random.randn(400).astype(np.float32)
    b = np.random.randn(400).astype(np.float32)

    dest = np.zeros_like(a)
    multiply_them(
            drv.Out(dest), drv.In(a), drv.In(b),
            block=(400,1,1), grid=(1,1))

    np.testing.assert_allclose(dest, a * b)


    
# Loads a test case into the provided pagelocked_buffer.
def load_normalized_test_case(data_fpath: Path,
                              pagelocked_buffer,
                              case_num=randint(0, 9)):
    test_case_path = str(data_fpath / "{}.pgm".format(case_num))
    # Flatten the image into a 1D array, normalize, and copy to pagelocked memory.
    img = np.array(Image.open(test_case_path)).ravel()
    np.copyto(pagelocked_buffer, 1.0 - img / 255.0)
    return case_num


def main():
    data_fpath = Path('/workspace/pkgs/TensorRT-5.0.0.10/data') / 'mnist'
    with build_engine('/tmp/lenet5.pb.uff') as engine:
        # Build an engine, allocate buffers and create a stream.
        # For more information on buffer allocation, refer to the introductory samples.
        inputs, outputs, bindings, stream = trt_common.allocate_buffers(engine)
        with engine.create_execution_context() as context:
            case_num = load_normalized_test_case(
                data_fpath, pagelocked_buffer=inputs[0].host)
            # For more information on performing inference, refer to the introductory samples.
            # The common.do_inference function will return a list of outputs - we only have one in this case.
            [output] = trt_common.do_inference(
                context,
                bindings=bindings,
                inputs=inputs,
                outputs=outputs,
                stream=stream)

            pred = np.argmax(output)
            print("Test Case: " + str(case_num))
            print("Prediction: " + str(pred))

            print("Begin timing")
            def perf():
                trt_common.do_inference(
                    context,
                    bindings=bindings,
                    inputs=inputs,
                    outputs=outputs,
                    stream=stream)

            elapsed = timeit.timeit(perf)
            print('timeit: {} ms'.format(min(elapsed * 1000)))
            


if __name__ == '__main__':
    main()
