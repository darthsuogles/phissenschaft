import torch
from torch import nn
from tensorboardX import SummaryWriter
import numpy as np


class HParams(object):
    def __init__(self, input_channels=3):
        self.blocks = []
        self.pooling_ops = {
            'max': nn.MaxPool2d,
            'avg': nn.AvgPool2d,
        }
        self.in_channels = input_channels
        self.receptive_field = 1
        self.pixel_delta = 1

    def _update_stats(self, **kwargs):
        """
        Compute receptive field
        """
        kernel_size = kwargs.get("kernel_size") or 1
        stride = kwargs.get("stride") or 1
        self.receptive_field += (kernel_size - 1) * self.pixel_delta
        self.pixel_delta *= stride

    def conv2d_bn(self, out_channels, **kwargs):
        """
        Apply 2D convolution followed by batch normalization and ReLU
        """
        self.blocks.append(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=out_channels,
                **kwargs))
        self.blocks.append(nn.BatchNorm2d(out_channels))
        self.blocks.append(nn.ReLU(inplace=True))
        self._update_stats(**kwargs)

    def conv2d(self, out_channels, **kwargs):
        """
        Apply 2D convolution followed by a ReLU
        """
        self.blocks.append(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=out_channels,
                **kwargs))
        self.blocks.append(nn.ReLU(inplace=True))
        self.in_channels = out_channels
        self._update_stats(**kwargs)

    def pooling(self, op_type, **kwargs):
        if op_type not in self.pooling_ops:
            raise TypeError("[pooling] does not support {}".format(op_type))
        self.blocks.append(self.pooling_ops[op_type](**kwargs))
        self._update_stats(**kwargs)

    def stacked(self):
        return nn.Sequential(*self.blocks)


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        hparams = HParams()

        hparams.conv2d(64, kernel_size=11, stride=4, padding=2, dilation=1)
        hparams.pooling("max", kernel_size=3, stride=2)

        hparams.conv2d(192, kernel_size=5, stride=1, padding=2, dilation=1)
        hparams.pooling("max", kernel_size=3, stride=2)

        for out_channels in [384, 256, 256]:
            hparams.conv2d(
                out_channels, kernel_size=3, stride=1, padding=1, dilation=1)
        hparams.pooling("max", kernel_size=3, stride=2)

        self.hparams = hparams
        self.features = hparams.stacked()

        self.linear_dim = hparams.in_channels * 6 * 6

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.linear_dim, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, input_tensor):
        spatial_feats = self.features(input_tensor)
        feats = spatial_feats.view(spatial_feats.shape[0], -1)
        pre_logits = self.classifier(feats)
        return pre_logits


def test_alex_net():
    net = AlexNet()

    input_tensor = torch.zeros(2, 3, 227, 227)
    pre_logits = net.forward(input_tensor)
    feats = net.features(input_tensor)

    with SummaryWriter(comment="AlexPyTorch") as writer:
        writer.add_graph(net, input_tensor, True)


class Inception(nn.Module):
    def __init__(self):
        pass

    def forward(self, input_tensor):
        pass


input_tensor = torch.randn(20, 3, 32, 64)
batch_norm_fn = nn.BatchNorm2d(3, affine=False, momentum=0)

arr = input_tensor.numpy()

spatial_mean = arr.mean(axis=(0, 1))
spatial_var = arr.var(axis=(0, 1))
epsilon = batch_norm_fn.eps
arr_normed = (arr - spatial_mean) / np.sqrt(spatial_var + epsilon)

ss = nn.BatchNorm2d(3)(input_tensor)

arr_expected = ss.detach().numpy()

np.abs(arr_expected - arr_normed)
