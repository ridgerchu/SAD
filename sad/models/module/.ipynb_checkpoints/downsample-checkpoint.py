import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode, MultiStepLIFNode
from spikingjelly.clock_driven import layer

def MS_conv_unit(in_channels, out_channels,kernel_size=1,padding=0,groups=1):
    return nn.Sequential(
        layer.SeqToANNContainer(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, groups=groups,bias=True),
           nn.BatchNorm2d(out_channels)  # 这里可以进行改进 ?
        )
    )
class MS_ConvBlock(nn.Module):
    def __init__(self, dim,
        mlp_ratio=4.0):
        super().__init__()
        self.neuron1 = MultiStepLIFNode(tau=2.0, detach_reset=True,backend='torch')
        self.conv1 = MS_conv_unit(dim, dim, 3, 1)

        self.neuron2 = MultiStepLIFNode(tau=2.0, detach_reset=True,backend='torch')
        self.conv2 = MS_conv_unit(dim, dim * mlp_ratio, 3, 1)

        self.neuron3 = MultiStepLIFNode(tau=2.0, detach_reset=True,backend='torch')

        self.conv3 = MS_conv_unit(dim*mlp_ratio, dim, 3, 1)


    def forward(self, x, hook=None, num=None):
        short_cut1 = x
        x = self.neuron1(x)
        if hook is not None:
            hook[self._get_name() + num + "_lif1"] = x.detach()
        x = self.conv1(x)+short_cut1
        short_cut2 = x
        x = self.neuron2(x)
        if hook is not None:
            hook[self._get_name() + num + "_lif2"] = x.detach()

        x = self.conv2(x)
        x = self.neuron3(x)
        if hook is not None:
            hook[self._get_name() + num + "_lif3"] = x.detach()

        x = self.conv3(x)
        x = x + short_cut2
        return x
class MS_DownSampling(nn.Module):
    def __init__(
        self,
        in_channels=2,
        embed_dims=256,
        kernel_size=3,
        stride=2,
        padding=1,
        first_layer=True,
    ):
        super().__init__()

        self.encode_conv = nn.Conv2d(
            in_channels,
            embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.encode_bn = nn.BatchNorm2d(embed_dims)
        if not first_layer:
            self.encode_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

    def forward(self, x, hook=None, num=None):

        T, B, _, _, _ = x.shape

        if hasattr(self, "encode_lif"):
            x = self.encode_lif(x)
            if hook is not None:
                hook[self._get_name() + num + "_lif"] = x.detach()
        x = self.encode_conv(x.flatten(0, 1))
        _, _, H, W = x.shape
        x = self.encode_bn(x).reshape(T, B, -1, H, W).contiguous()
        return x
class Sps_downsampling(nn.Module):
    def __init__(
            self,
            in_channels=3,
            mlp_ratios=4,
            embed_dim=[64,128,256],
            pooling_stat="1111",
            spike_mode="lif",
    ):
        super().__init__()
        self.downsample1_1 = MS_DownSampling(
            in_channels=in_channels,
            embed_dims=embed_dim[0] // 2,
            kernel_size=7,
            stride=2,
            padding=3,
            first_layer=True,
        )

        self.ConvBlock1_1 = nn.ModuleList(
            [MS_ConvBlock(dim=embed_dim[0] // 2, mlp_ratio=mlp_ratios)]
        )

        self.downsample1_2 = MS_DownSampling(
            in_channels=embed_dim[0] // 2,
            embed_dims=embed_dim[0],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
        )

        self.ConvBlock1_2 = nn.ModuleList(
            [MS_ConvBlock(dim=embed_dim[0], mlp_ratio=mlp_ratios)]
        )

        self.downsample2 = MS_DownSampling(
            in_channels=embed_dim[0],
            embed_dims=embed_dim[1],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
        )

        self.ConvBlock2_1 = nn.ModuleList(
            [MS_ConvBlock(dim=embed_dim[1], mlp_ratio=mlp_ratios)]
        )

        self.ConvBlock2_2 = nn.ModuleList(
            [MS_ConvBlock(dim=embed_dim[1], mlp_ratio=mlp_ratios)]
        )

        self.downsample3 = MS_DownSampling(
            in_channels=embed_dim[1],
            embed_dims=embed_dim[2],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
        )
    def forward(self,x, hook=None):
        x = self.downsample1_1(x,hook,"1")
        for blk in self.ConvBlock1_1:
            x = blk(x,hook,"1_1") # 112
        x = self.downsample1_2(x)
        for blk in self.ConvBlock1_2:
            x = blk(x,hook,"1_2") #56


        x = self.downsample2(x,hook,"2")

        for blk in self.ConvBlock2_1:
            x = blk(x,hook,"2_1") #28

        for blk in self.ConvBlock2_2:
            output_1 = blk(x,hook,"2_2") #28
        
        
        output_2 = self.downsample3(x,hook,"3")
        
        return output_1, output_2
# model=Sps_downsampling()
# x=torch.randn(1,1,3,224,224)
# print(model(x).shape)
