import torch
import torch.nn as nn
from spikingjelly.clock_driven import functional
try:
    from torchvision.models.utils import load_state_dict_from_url
except ImportError:
    from torchvision._internally_replaced_utils import load_state_dict_from_url
    
__all__ = ['MultiStepMSResNet', 'multi_step_sew_resnet18']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',

}

# modified by https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

def sew_function(x: torch.Tensor, y: torch.Tensor, cnf:str):
    if cnf == 'ADD':
        return x + y
    elif cnf == 'AND':
        return x * y
    elif cnf == 'IAND':
        return x * (1. - y)
    else:
        raise NotImplementedError



def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, cnf: str = None, single_step_neuron: callable = None, **kwargs):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.sn1 = single_step_neuron(**kwargs)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.sn2 = single_step_neuron(**kwargs)
        self.downsample = downsample
        if downsample is not None:
            self.downsample_sn = single_step_neuron(**kwargs)
        self.stride = stride
        self.cnf = cnf

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.sn2(out)

        if self.downsample is not None:
            identity = self.downsample_sn(self.downsample(x))

        out = sew_function(identity, out, self.cnf)

        return out

    def extra_repr(self) -> str:
        return super().extra_repr() + f'cnf={self.cnf}'



# class MultiStepBasicBlock(BasicBlock):
#     def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
#                  base_width=64, dilation=1, norm_layer=None, cnf: str = None, multi_step_neuron: callable = None, **kwargs):
#         super().__init__(inplanes, planes, stride, downsample, groups,
#                  base_width, dilation, norm_layer, cnf, multi_step_neuron, **kwargs)
#
#     def forward(self, x_seq):
#         identity = x_seq
#
#         out = self.sn1(x_seq)
#         out = functional.seq_to_ann_forward(x_seq, [self.conv1, self.bn1])
#
#         out = self.sn2(out)
#         out = functional.seq_to_ann_forward(out, [self.conv2, self.bn2])
#
#
#         if self.downsample is not None:
#             identity = functional.seq_to_ann_forward(x_seq, self.downsample)
#
#         out = identity + out
#
#         return out

class MultiStepBasicBlock(BasicBlock):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, cnf: str = None, multi_step_neuron: callable = None, **kwargs):
        super().__init__(inplanes, planes, stride, downsample, groups,
                 base_width, dilation, norm_layer, cnf, multi_step_neuron, **kwargs)

    def forward(self, x_seq):
        identity = x_seq

        out = functional.seq_to_ann_forward(x_seq, [self.conv1, self.bn1])
        out = self.sn1(out)

        out = functional.seq_to_ann_forward(out, [self.conv2, self.bn2])
        out = self.sn2(out)

        if self.downsample is not None:
            identity = self.downsample_sn(functional.seq_to_ann_forward(x_seq, self.downsample))

        out = sew_function(identity, out, self.cnf)

        return out

class MultiStepMSResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, T:int=None, cnf: str=None, multi_step_neuron: callable = None, **kwargs):
        super().__init__()
        self.T = T
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.sn1 = multi_step_neuron(**kwargs)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(block, 64, layers[0], cnf=cnf, multi_step_neuron=multi_step_neuron, **kwargs)
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0], cnf=cnf,
                                      multi_step_neuron=multi_step_neuron, **kwargs)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1], cnf=cnf,
                                      multi_step_neuron=multi_step_neuron, **kwargs)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], cnf=cnf,
                                      multi_step_neuron=multi_step_neuron, **kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def make_layer(self, block, planes, blocks, stride=1, dilate=False, cnf: str = None, multi_step_neuron: callable = None, **kwargs):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, cnf, multi_step_neuron, **kwargs))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, cnf=cnf, multi_step_neuron=multi_step_neuron, **kwargs))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: torch.Tensor):
        # See note [TorchScript super()]
        x_seq = None
        if x.dim() == 5:
            # x.shape = [T, N, C, H, W]
            x_seq = functional.seq_to_ann_forward(x, [self.conv1, self.bn1])
        else:
            assert self.T is not None, 'When x.shape is [N, C, H, W], self.T can not be None.'
            # x.shape = [N, C, H, W]
            x = self.conv1(x)
            x = self.bn1(x)
            x.unsqueeze_(0)
            x_seq = x.repeat(self.T, 1, 1, 1, 1)

        x_seq = functional.seq_to_ann_forward(x_seq, self.maxpool)

        x_seq = self.layer1(x_seq)
        x_seq = self.layer2(x_seq)
        x_seq = self.layer3(x_seq)
        x_seq = self.layer4(x_seq)

        x_seq = functional.seq_to_ann_forward(x_seq, self.avgpool)
        x_seq = self.sn1(x_seq)
        x_seq = torch.flatten(x_seq, 2)
        # x_seq = self.fc(x_seq.mean(0))
        x_seq = functional.seq_to_ann_forward(x_seq, self.fc)
    
        return x_seq

    def forward(self, x):
        """
        :param x: the input with `shape=[N, C, H, W]` or `[*, N, C, H, W]`
        :type x: torch.Tensor
        :return: output
        :rtype: torch.Tensor
        """
        return self._forward_impl(x)




def _multi_step_sew_resnet(arch, block, layers, pretrained, progress, T, cnf, multi_step_neuron, **kwargs):
    model = MultiStepMSResNet(block, layers, T=T, cnf=cnf, multi_step_neuron=multi_step_neuron, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def multi_step_sew_resnet18(pretrained=False, progress=True, T: int = None, cnf: str = 'ADD', multi_step_neuron: callable=None, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param T: total time-steps
    :type T: int
    :param cnf: the name of spike-element-wise function
    :type cnf: str
    :param multi_step_neuron: a multi-step neuron
    :type multi_step_neuron: callable
    :param kwargs: kwargs for `multi_step_neuron`
    :type kwargs: dict
    :return: Spiking ResNet-18
    :rtype: torch.nn.Module

    The multi-step spike-element-wise ResNet-18 `"Deep Residual Learning in Spiking Neural Networks" <https://arxiv.org/abs/2102.04159>`_
    modified by the ResNet-18 model from `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """

    return _multi_step_sew_resnet('resnet18', MultiStepBasicBlock, [2, 2, 2, 2], pretrained, progress, T, cnf, multi_step_neuron, **kwargs)
