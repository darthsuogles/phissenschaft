import torch
from torch import nn
import numpy as np
import math
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from tensorboardX import SummaryWriter


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def _update_receptive_fields(self, receptive_field, pixel_delta):
        for m in self.modules():
            if not isinstance(m, nn.Conv2d) and \
               not isinstance(m, nn.MaxPool2d):
                continue

            kernel_size = np.array(m.kernel_size)
            if np.all(kernel_size == 1):
                continue
            stride = np.array(m.stride)
            receptive_field += (kernel_size - 1) * pixel_delta
            pixel_delta *= stride

        return receptive_field, pixel_delta

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def _update_receptive_fields(self, receptive_field: np.ndarray,
                                 pixel_delta: np.ndarray):
        for m in self.modules():
            if not isinstance(m, nn.Conv2d) and \
               not isinstance(m, nn.MaxPool2d):
                continue

            kernel_size = np.array(m.kernel_size)
            stride = np.array(m.stride)
            receptive_field += (kernel_size - 1) * pixel_delta
            pixel_delta *= stride

        return receptive_field, pixel_delta

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetInitBlock(nn.Module):
    def __init__(self, inplanes):
        super(ResNetInitBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                3, inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.receptive_field = np.array([11, 11])
        self.pixel_delta = np.array([4, 4])

    def forward(self, input_tensor):
        return self.block(input_tensor)


class ResNet(nn.Module):
    def __init__(self, block, layer_sizes):
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.init_block = ResNetInitBlock(self.inplanes)
        self.receptive_field = self.init_block.receptive_field
        self.pixel_delta = self.init_block.pixel_delta

        self.layer1 = self._make_layer(block, 64,
                                       layer_sizes[0])
        self.layer2 = self._make_layer(block, 128,
                                       layer_sizes[1], stride=2)
        self.layer3 = self._make_layer(block, 256,
                                       layer_sizes[2], stride=2)
        self.layer4 = self._make_layer(block, 512,
                                       layer_sizes[3], stride=2)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Initialize with Xavier
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        layers += [block(self.inplanes, planes) for _ in range(1, num_blocks)]

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.init_block(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def count_parameters(model: nn.Module, trainable_only=True):
    return sum(p.numel() for p in model.parameters()
               if p.requires_grad or not trainable_only)



def compute_stats(model: nn.Module):
    receptive_field = 1
    pixel_detla = 1
    for m in model.modules():
        if type(m) not in [nn.Conv2d, nn.MaxPool2d]:
            continue
        k = m.kernel_size
        if isinstance(k, tuple):
            k = k[0]
        if 1 == k:
            # Spacial layers conv1x1 are mainly used for
            # spatial and filter-wise subsampling
            continue

        s = m.stride
        if isinstance(s, tuple):
            s = s[0]

        receptive_field += (k - 1) * pixel_detla
        pixel_detla *= s
        print(type(m), k, s, receptive_field, pixel_detla)

    return receptive_field, pixel_detla


model = ResNet(Bottleneck, [3, 4, 6, 3])
count_parameters(model)
compute_stats(model)

input_tensor = torch.randn(2, 3, 224, 224)
ss = model.forward(input_tensor)
init_block = ResNetInitBlock(64)

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

# state_dict = model_zoo.load_url(model_urls['resnet18'])
# model.load_state_dict(state_dict)
# # feats = model.forward(input_tensor)

resnet18_reference = models.resnet18(pretrained=True)
count_parameters(resnet18_reference)
compute_stats(resnet18_reference)

#

# with SummaryWriter(comment="ResNet18") as writer:
#     writer.add_graph(resnet18, input_tensor)

# sum(p.numel() for p in resnet18.parameters() if p.requires_grad)

# pp = resnet18.forward(input_tensor)
