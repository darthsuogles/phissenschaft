PYTHON_BIN_PATH = python
PYTEST_BIN = python -m pytest

KERNEL_SRCS = $(wildcard custom_ops/cc/kernels/*.cc)
CUDA_SRCS = $(wildcard custom_ops/cc/kernels/*.cu.cc)
SRCS = $(filter-out $(CUDA_SRCS),$(KERNEL_SRCS)) $(wildcard custom_ops/cc/ops/*.cc)

TF_CFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

CUDA_CFLAGS = -std=c++11 -O2 $(TF_CFLAGS) -I/usr/local --expt-relaxed-constexpr -ftz=true -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
LDFLAGS = -fPIC -L/usr/local/cuda/lib64 -lcudart $(TF_LFLAGS)

TARGET_LIB = custom_ops/python/ops/custom_ops.so
TARGET_CUDA_OBJ = custom_ops/python/ops/custom_ops.cu.o

.PHONY: op
op: $(TARGET_LIB)

$(TARGET_LIB): $(TARGET_CUDA_OBJ) $(SRCS)
	g++ -std=c++11 -shared -o $@ $^ $(TF_CFLAGS) $(LDFLAGS)

$(TARGET_CUDA_OBJ): $(CUDA_SRCS)
	nvcc -c -o $@ $^ $(CUDA_CFLAGS)

test: custom_ops/python/ops/custom_ops_tests.py custom_ops/python/ops/custom_ops_lib.py $(TARGET_LIB)
	$(PYTEST_BIN) $<

pip_pkg: $(TARGET_LIB)
	./build_pip_pkg.sh make artifacts


.PHONY: clean
clean:
	rm -f $(TARGET_LIB) $(TARGET_CUDA_OBJ)
