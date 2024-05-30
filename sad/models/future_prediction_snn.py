import torch
import torch.nn as nn
import copy

from sad.layers.convolutions import SpikingDeepLabHead
from sad.layers.temporal_snn import SpatialGRU, Dual_LIF_temporal_mixer, BiGRU


from sad.models.module.MS_ResNet import multi_step_sew_resnet18
from spikingjelly.clock_driven.neuron import (
    MultiStepParametricLIFNode,
    MultiStepLIFNode,
)


class FuturePrediction(nn.Module):
    def __init__(self, in_channels, latent_dim, n_future, mixture=True):
        super(FuturePrediction, self).__init__()
        # self.n_spatial_gru = n_gru_blocks

        backbone = multi_step_sew_resnet18(pretrained=False, multi_step_neuron=MultiStepLIFNode)

        gru_in_channels = latent_dim
        self.layer1 = backbone.layer1
        self.layer1[0].conv1 = torch.nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                               bias=False)
        self.layer1[0].bn1 = torch.nn.BatchNorm2d(in_channels)
        self.layer1[0].conv2 = torch.nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                               bias=False)
        self.layer1[0].bn2 = torch.nn.BatchNorm2d(in_channels)

        self.layer1[1].conv1 = torch.nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                               bias=False)
        self.layer1[1].bn1 = torch.nn.BatchNorm2d(in_channels)
        self.layer1[1].conv2 = torch.nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                               bias=False)
        self.layer1[1].bn2 = torch.nn.BatchNorm2d(in_channels)

        # self.layer2 = backbone.layer2
        # self.layer3 = backbone.layer3
        # self.layer3[1].conv2 = torch.nn.Conv2d(256, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        # self.layer3[1].bn2 = torch.nn.BatchNorm2d(64)

        self.layer2 = copy.deepcopy(backbone.layer1)
        self.layer2[0].conv1 = torch.nn.Conv2d(in_channels, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer2[0].bn1 = torch.nn.BatchNorm2d(384)
        self.layer2[0].conv2 = torch.nn.Conv2d(384, in_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer2[0].bn2 = torch.nn.BatchNorm2d(in_channels)

        self.layer2[1].conv1 = torch.nn.Conv2d(in_channels, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer2[1].bn1 = torch.nn.BatchNorm2d(384)
        self.layer2[1].conv2 = torch.nn.Conv2d(384, in_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer2[1].bn2 = torch.nn.BatchNorm2d(in_channels)

        self.layer3 = copy.deepcopy(self.layer2)

        # 修改第1个MultiStepBasicBlock中的BatchNorm2d层的特征数



        self.dual_lif = Dual_LIF_temporal_mixer(gru_in_channels, in_channels, n_future=n_future, mixture=mixture)
        # self.res_blocks1 = nn.Sequential(*[Block(in_channels) for _ in range(n_res_layers)])
        #
        # self.spatial_grus = []
        # self.res_blocks = []
        # for i in range(self.n_spatial_gru):
        #     self.spatial_grus.append(SpatialGRU(in_channels, in_channels))
        #     if i < self.n_spatial_gru - 1:
        #         self.res_blocks.append(nn.Sequential(*[Block(in_channels) for _ in range(n_res_layers)]))
        #     else:
        #         self.res_blocks.append(DeepLabHead(in_channels, in_channels, 128))
        #
        # self.spatial_grus = torch.nn.ModuleList(self.spatial_grus)
        # self.res_blocks = torch.nn.ModuleList(self.res_blocks)

    def forward(self, x, state):
        # x has shape (b, 1, c, h, w), state: torch.Tensor [b, n_present, hidden_size, h, w]
        x = self.dual_lif(x, state)


        # b, n_future, c, h, w = x.shape  # 预测未来情况，这时候就已经有未来的feature了
        # x = self.res_blocks1(x.view(b * n_future, c, h, w))  # 过ResNet Block，此时没有未来feature
        # x = x.view(b, n_future, c, h, w)

        x = torch.cat([state, x], dim=1)  # 回正后把未来feature和当前的状况做一个融合
        b, s, c, h, w = x.shape
        x = x.reshape(s, b, c, h, w)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.reshape(b, s, c, h, w)

        # hidden_state = x[:, 0]
        # for i in range(self.n_spatial_gru):
        #     x = self.spatial_grus[i](x, hidden_state)  # 使用Spatial GRU，正常计算
        #
        #     b, s, c, h, w = x.shape
        #     x = self.res_blocks[i](x.view(b * s, c, h, w))  # 过Res Blocks，特征提取，这一块可以被整合进入RWKV
        #     x = x.view(b, s, c, h, w)

        return x