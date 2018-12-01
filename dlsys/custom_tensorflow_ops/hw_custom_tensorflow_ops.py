import tensorflow as tf
import subprocess
from pathlib import Path

# from universe.experimental.aialgo.hack import *
# _THIS_MODULE = xpilot_repo_root() / 'universe' / 'experimental' / 'custom_tensorflow_ops'

# def compiling():
#     tf.sysconfig.get_include()
#     tf.sysconfig.get_lib()

#     tf_cxx_flags = " ".join(tf.sysconfig.get_compile_flags())
#     tf_lib_flags = " ".join(tf.sysconfig.get_link_flags())

#     template_nvcc_command = """
#     nvcc -std=c++11 -c -o {custom_op_name}.cu.o {custom_op_name}.cu.cc
#       {tf_cxx_flags} -I/usr/local -D NDEBUG -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
#     """.replace('\n', '')

#     template_gcc_command = """
#     g++ -std=c++11 -shared -o {custom_op_name}.so
#       {custom_op_name}.cc
#       {custom_op_name}.cu.o
#       {tf_cxx_flags} -fPIC
#       -L/usr/local/cuda/lib64 -lcudart {tf_lib_flags}
#     """.replace('\n', '')

#     subprocess.check_output(template_nvcc_command.format(
#         custom_op_name='kernel_example',
#         tf_cxx_flags=tf_cxx_flags,
#         tf_lib_flags=tf_lib_flags).split(), cwd=_THIS_MODULE)

#     subprocess.check_output(template_gcc_command.format(
#         custom_op_name='kernel_example',
#         tf_cxx_flags=tf_cxx_flags,
#         tf_lib_flags=tf_lib_flags).split(), cwd=_THIS_MODULE)


custom_ops_module = tf.load_op_library('custom_ops/python/ops/custom_ops.so')
