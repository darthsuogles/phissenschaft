"""
Transfering a model from PyTorch to Caffe2 and Mobile using ONNX
================================================================

In this tutorial, we describe how to use ONNX to convert a model defined
in PyTorch into the ONNX format and then load it into Caffe2. Once in
Caffe2, we can run the model to double-check it was exported correctly,
and we then show how to use Caffe2 features such as mobile exporter for
executing the model on mobile devices.

For this tutorial, you will need to install `onnx <https://github.com/onnx/onnx>`__,
`onnx-caffe2 <https://github.com/onnx/onnx-caffe2>`__ and `Caffe2 <https://caffe2.ai/>`__.
You can get binary builds of onnx and onnx-caffe2 with
``conda install -c ezyang onnx onnx-caffe2``.

``NOTE``: This tutorial needs PyTorch master branch which can be installed by following
the instructions `here <https://github.com/pytorch/pytorch#from-source>`__

"""

# Some standard imports
import io
import numpy as np
import PIL.Image

from torch import nn
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo
import torch.onnx
from torchvision.transforms import ToTensor

######################################################################
# Super-resolution is a way of increasing the resolution of images, videos
# and is widely used in image processing or video editing. For this
# tutorial, we will first use a small super-resolution model with a dummy
# input.
#
# First, let's create a SuperResolution model in PyTorch. `This
# model <https://github.com/pytorch/examples/blob/master/super_resolution/model.py>`__
# comes directly from PyTorch's examples without modification:
#

# Super Resolution model definition in PyTorch
import torch.nn as nn
import torch.nn.init as init


class SuperResolutionNet(nn.Module):
    def __init__(self, upscale_factor, inplace=False):
        super(SuperResolutionNet, self).__init__()

        self.relu = nn.ReLU(inplace=inplace)
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal(self.conv4.weight)

# Create the super-resolution model by using the above model definition.
torch_model = SuperResolutionNet(upscale_factor=3)


######################################################################
# Ordinarily, you would now train this model; however, for this tutorial,
# we will instead download some pre-trained weights. Note that this model
# was not trained fully for good accuracy and is used here for
# demonstration purposes only.
#

# Load pretrained model weights
model_url = 'https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth'
batch_size = 1    # just a random number

# Initialize model with the pretrained weights
if torch.cuda.is_available():
    map_location = None
else:
    def map_location(storage, loc): return storage

torch_model.load_state_dict(model_zoo.load_url(model_url, map_location=map_location))

# set the train mode to false since we will only run the forward pass.
torch_model.train(False)


######################################################################
# Exporting a model in PyTorch works via tracing. To export a model, you
# call the ``torch.onnx._export()`` function. This will execute the model,
# recording a trace of what operators are used to compute the outputs.
# Because ``_export`` runs the model, we need provide an input tensor
# ``x``. The values in this tensor are not important; it can be an image
# or a random tensor as long as it is the right size.
#
# To learn more details about PyTorch's export interface, check out the
# `torch.onnx documentation <http://pytorch.org/docs/master/onnx.html>`__.
#

# Input to the model
orig_img = PIL.Image.open('pallas_cat.jpg').convert('YCbCr')
y, cb, cr = orig_img.split()
x = Variable(ToTensor()(y)).view(1, -1, y.size[1], y.size[0])

# Export the model
torch_out = torch.onnx._export(torch_model,
                               # model input (or a tuple for multiple inputs)
                               x,
                               # where to save the model (can be a file or file-like object)
                               "super_resolution.onnx",
                               # store the trained parameter weights inside the model file
                               export_params=True)


######################################################################
# ``torch_out`` is the output after executing the model. Normally you can
# ignore this output, but here we will use it to verify that the model we
# exported computes the same values when run in Caffe2.
#
# Now let's take the ONNX representation and use it in Caffe2. This part
# can normally be done in a separate process or on another machine, but we
# will continue in the same process so that we can verify that Caffe2 and
# PyTorch are computing the same value for the network:
#

import onnx
import onnx_caffe2.backend

# Load the ONNX GraphProto object. Graph is a standard Python protobuf object
graph = onnx.load("super_resolution.onnx")

# prepare the caffe2 backend for executing the model this converts the ONNX graph into a
# Caffe2 NetDef that can execute it. Other ONNX backends, like one for CNTK will be
# availiable soon.
prepared_backend = onnx_caffe2.backend.prepare(graph)

# run the model in Caffe2

# Construct a map from input names to Tensor data.
# The graph itself contains inputs for all weight parameters, followed by the input image.
# Since the weights are already embedded, we just need to pass the input image.
# last input the graph
W = {graph.input[-1]: x.data.numpy()}

# Run the Caffe2 net:
c2_out = prepared_backend.run(W)[0]

# Verify the numerical correctness upto 3 decimal places
tnsr = torch_out.data.cpu().numpy()
#out_img_y = tnsr_out.data[0].numpy()
np.testing.assert_almost_equal(tnsr, c2_out, decimal=3)

def to_image(out_img_arr):
    out_img_y = out_img_arr[0] * 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = PIL.Image.fromarray(np.uint8(out_img_y[0]), mode='L')

    out_img_cb = cb.resize(out_img_y.size, PIL.Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, PIL.Image.BICUBIC)
    out_img = PIL.Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
    return out_img

to_image(tnsr).save('torch_sr.jpg')
to_image(c2_out).save('caffe2_sr.jpg')
