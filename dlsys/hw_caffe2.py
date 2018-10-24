import onnx
import caffe2.python.onnx.frontend
from caffe2.proto import caffe2_pb2
from caffe2.python import workspace, model_helper
import numpy as np

# Create random tensor of three dimensions
x = np.random.rand(4, 3, 2)
print(x)
print(x.shape)

workspace.FeedBlob("my_x", x)
x2 = workspace.FetchBlob("my_x")
print(x2)

# Data in Caffe2 is organized as blobs.
# A blob is just a named chunk of data in memory.
# A Workspace stores all the blobs.
# Workspaces initialize themselves the moment you start using them.
workspace.Blobs()


# Create the input data
data = np.random.rand(16, 100).astype(np.float32)

# Create labels for the data as integers [0, 9].
label = (np.random.rand(16) * 10).astype(np.int32)

workspace.FeedBlob("data", data)
workspace.FeedBlob("label", label)

# Create model using a model helper
model = model_helper.ModelHelper(name="test_net")

# These are like TensorFlow's tensors
weight = model.param_init_net.XavierFill([], 'fc_w', shape=[10, 100])
bias = model.param_init_net.ConstantFill([], 'fc_b', shape=[10, ])

fc_1 = model.net.FC(["data", "fc_w", "fc_b"], "fc1")
pred = model.net.Sigmoid(fc_1, "pred")
softmax, loss = model.net.SoftmaxWithLoss([pred, "label"], ["softmax", "loss"])

print(model.net.Proto())
print(model.param_init_net.Proto())

model.AddGradientOperators([loss])
print(model.net.Proto())

workspace.RunNetOnce(model.param_init_net)
workspace.CreateNet(model.net)

for _ in range(100):
    data = np.random.rand(16, 100).astype(np.float32)
    label = (np.random.rand(16) * 10).astype(np.int32)

    workspace.FeedBlob("data", data)
    workspace.FeedBlob("label", label)

    workspace.RunNet(model.name, 10)   # run for 10 times

workspace.FetchBlob("pred")
