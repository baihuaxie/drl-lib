"""
Convolutional Networks
"""

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

from networks_util import register


# pretrained models
model_urls = {

}

# 5x5 conv filter
def conv5x5(in_planes, out_planes, stride=1, groups=1):
    """
    5x5-conv filter
    - preserve fmap dimensions if stride=1
    - exactly halves fmap dimensions if stride=2
        - requires padding=2, dilation=1
    - becomes depthwise conv filter when in_planes = out_planes = groups
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride, groups=groups,
                     padding=2, dilation=1, bias=False)


# 3x3 conv filter
def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """
    3x3-conv filter
    - preserve fmap dimensions if stride=1
    - exactly halves fmap dimensions if stride=2
        - requires padding=1, dilation=1
    - becomes depthwise conv filter when in_planes = out_planes = groups
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, groups=groups,
                     padding=1, dilation=1, bias=False)


# 1x1 conv filter
def conv1x1(in_planes, out_planes, stride=1, groups=1):
    """
    1x1-conv filter
    - preserve fmap dimensions if stride=1
    - exactly halves fmap dimensions if stride=2
        - requires padding=0, dilation=arbitrary
    - becomes depthwise conv filter when in_planes = out_planes = groups
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, groups=groups,
                     padding=0, dilation=1, bias=False)


class BasicBlock(nn.Module):
    """
    Basic convolutional block

    - conv-3x3 -> bn -> relu
    """
    def __init__(self, inplanes, outplanes, kernel_size=3, stride=1, groups=1, dilation=1,
                 base_width=64, downsample=None, norm_layer=None, act_layer=None):
        """
        Constructor

        Args:
            inplanes:       (int) number of input channels
            outplanes:      (int) number of output channels
            stride:         (int) stride
            kernel_size:    (int) conv filter size
            groups:         (int) number of groups for grouped conv filters
            dilation:       (int) for dilated conv filters
            base_width:     (int) BasicBlock only supports base_width=64
            downsample:     (nn.Module) if specified, downsamples output fmaps for skip connection
                            not supported in BasicBlock
            norm_layer:     (nn.Module) normalization layer; default = nn.BatchNorm2d
            act_layer:      (nn.Module) activation layer; default = nn.ReLU
        """
        super(BasicBlock, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU

        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock class only supports groups=1 and base_width=64')
        if dilation > 1:
            raise ValueError('BasicBlock class only supports dilation=1')
        self.downsample = downsample

        if kernel_size == 3:
            conv = conv3x3
        if kernel_size == 5:
            conv = conv5x5

        self.conv1 = conv(inplanes, outplanes, stride=stride)
        self.bn1 = norm_layer(outplanes)

        self.relu = act_layer(inplace=True)

    def forward(self, x):
        """
        forward method
        """
        out = self.bn1(self.conv1(x))
        out = self.relu(out)

        return out


class ConvNet(nn.Module):
    """
    Basic CNN architecture
    """

    def __init__(self, block, layers, latent_dim=512, norm_layer=None, act_layer=None):
        """
        Constructor

        Args:
            block:      (nn.Module) building block; e.g., BasicBlock
            layers:     (list of int) a list of integers specifying number of blocks per stack
            latent_dim: (int) dimension of latent space at network output; default = 512
            norm_layer: (nn.Module) normalization layer; default = nn.BatchNorm2d
            act_layer:  (nn.Module) activation layer; default = nn.ReLU
        """
        super(ConvNet, self).__init__()

        self._norm_layer = norm_layer
        self._act_layer = act_layer

        self.conv1 = conv3x3(3, 16, stride=2)
        self.bn1 = norm_layer(16)

        self.stack1 = self._make_stack(block=block, num_layers=layers[0], inplanes=16, outplanes=32,
                                       kernel_size=3, stride=2)
        self.stack2 = self._make_stack(block=block, num_layers=layers[1], inplanes=32, outplanes=64,
                                       kernel_size=3, stride=2)
        self.stack3 = self._make_stack(block=block, num_layers=layers[2], inplanes=64, outplanes=128,
                                       kernel_size=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, latent_dim)

        self.relu = self._act_layer(inplace=True)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_stack(self, block, num_layers, inplanes, outplanes, kernel_size=3, stride=1):
        """
        Build a stack of blocks
        - first block in stack can have stride > 1 + downsample for skip connection (if applicable)
        - other blocks in stack have stride=1, inplanes=outplanes (if applicable)

        Args:
            block:          (nn.Module) building block
            num_layers:     (int) number of blocks in the stack
            inplanes:       (int) number of input channels to the stack
            outplanes:      (int) number of output channels to the stack
            kernel_size:    (int) conv filter size; 1x1, 3x3, 5x5
            stride:         (int) number of stride for conv filter in first block;
                            for other blocks stride=1

        Returns:
            (nn.Module) a stack of blocks; returned by calling nn.Sequential()
        """

        norm_layer = self._norm_layer
        act_layer = self._act_layer
        downsample = None

        # if stride > 1
        # or if block inplanes != block outplanes (only possible for first block in the stack)
        # apply downsample for possible skip connections by 1x1-conv filters
        if stride != 1 or inplanes != outplanes:
            downsample = nn.Sequential(
                conv1x1(inplanes, outplanes, stride=stride),
                norm_layer(outplanes)
            )

        # initialize layers
        layers = []

        # first block in stack can have stride > 1
        layers.append(block(inplanes, outplanes, kernel_size=kernel_size, stride=stride,
                            downsample=downsample, norm_layer=norm_layer, act_layer=act_layer))

        # other blocks in stack
        # inplanes=outplanes, stride=1, downsample=None
        for _ in range(1, num_layers):
            layers.append(block(outplanes, outplanes, kernel_size=kernel_size, stride=1,
                                downsample=None, norm_layer=norm_layer, act_layer=act_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        forward method
        """
        out = self.bn1(self.conv1(x))
        out = self.relu(out)

        out = self.stack1(out)
        out = self.stack2(out)
        out = self.stack3(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out


def _convnet(arch, block, layers, latent_dim=512, pretrained=False, progress=False, **kwargs):
    """
    Common interface for building ConvNet object

    Args:
        arch:           (str) architecture of pretrained model
        block:          (nn.Module) building block
        layers:         (list of int) a list of integers specifying number of blocks per stack
        latent_dim:     (int) dimension of latent space of network output
        pretrained:     (bool) if true download pretrained model
        progress:       (bool) if true display download progress
        **kwargs:       pointer to additional arguments

    Returns:
        (ConvNet) object for ConvNet class
    """
    model = ConvNet(block=block, layers=layers, latent_dim =latent_dim, **kwargs)

    # load pretrained model if specified & available
    if pretrained:
        if arch in model_urls.keys():
            state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
            model.load_state_dict(state_dict)

    return model

@register(name='simplecnn-k3s4')
def convnet_simplecnn_k3s4(pretrained=False, progress=False, **kwargs):
    """
    Build simple CNN network
    - kernel_size = 3 (default)
    - stacks = 4
        each stack contains 1 block with stride=2
    - latent_dim = 512
    """
    return _convnet('simplecnn', block=BasicBlock, layers=[1, 1, 1, 1], latent_dim=512,
                    pretrained=pretrained, progress=progress, **kwargs)
