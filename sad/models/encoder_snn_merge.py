from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.helpers import clean_state_dict
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from spikingjelly.clock_driven.neuron import (
    MultiStepLIFNode,
    MultiStepParametricLIFNode,
)
from sad.models.module import *
from sad.layers.convolutions import SpikingUpsamplingConcat, SpikingDeepLabHead, DeepLabHead, MergeUpsamplingConcat
from timm.models import (
    create_model,
    safe_model_name,
    resume_checkpoint,
    load_checkpoint,
    convert_splitbn_model,
    model_parameters,
)

class CoinMLP(nn.Module):
    def __init__(
        self,
        img_size_h=128,
        img_size_w=128,
        patch_size=16,
        in_channels=2,
        num_classes=11,
        embed_dims=512,
        num_heads=8,
        mlp_ratios=4,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        depths=[6, 8, 6],
        sr_ratios=[8, 4, 2],
        T=4,
        pooling_stat="1111",
        attn_mode="direct_xor",
        spike_mode="lif",
        get_embed=False,
        dvs_mode=False,
        TET=False,
        cml=False,
        pretrained=False,
        pretrained_cfg=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        self.T = T
        self.TET = TET
        self.dvs = dvs_mode
        self.D = 48
        self.C = 64
        self.depth_layer_1 = SpikingDeepLabHead(384, 384, hidden_channel=64)
        self.depth_layer_2 = MergeUpsamplingConcat(192 + 384, self.D)

        self.feature_layer_1 = SpikingDeepLabHead(384, 384, hidden_channel=64)
        self.feature_layer_2 = MergeUpsamplingConcat(192 + 384, self.C)



        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depths)
        ]  # stochastic depth decay rule

        patch_embed = Sps_downsampling(
            embed_dim=[int(embed_dims/4),int(embed_dims/2),embed_dims],
            pooling_stat=pooling_stat,
            spike_mode=spike_mode,
        )


        setattr(self, f"patch_embed", patch_embed)

        # classification head
        self.head_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")
        self.head_lif_2 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, hook=None):
        patch_embed = getattr(self, f"patch_embed")

        x_1, x_2 = patch_embed(x, hook=hook)
#         import pdb
#         pdb.set_trace()

        return x_1, x_2

    def forward(self, x, hook=None):
        x_1, x_2 = self.forward_features(x, hook=hook)
        
        x_1 = self.head_lif(x_1)
        x_2 = self.head_lif_2(x_2)


        # x_2 = torch.mean(x_2, dim=0)

        x_1 = torch.mean(x_1, dim=0)
        feature = self.feature_layer_1(x_2)
        feature = torch.mean(feature, dim=0)
        feature = self.feature_layer_2(feature, x_1)


        depth = self.depth_layer_1(x_2)
        depth = torch.mean(depth, dim=0)
        depth = self.depth_layer_2(depth, x_1)

        if hook is not None:
            hook["head_lif"] = x.detach()

        # feature = torch.mean(feature, dim=0)
        # depth = torch.mean(depth, dim=0)

        return feature, depth


@register_model
def sdt(**kwargs):
    model = CoinMLP(
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model

def main():
    encoder = create_model(
            "sdt",
            T = 3,  # 默认时间步长
            pretrained = False,  # 默认情况下不使用预训练模型
            drop_rate = 0.0,  # 默认dropout率
            drop_path_rate = 0.2,  # 默认drop path率
            drop_block_rate = None,  # 默认drop block率，未指定
            num_heads = 8,  # 默认头数
            num_classes = 1000,  # 默认类别数
            pooling_stat = "1111",  # 默认池化状态
            img_size_h = 480,  # 默认图像高度，未指定
            img_size_w = 224,  # 默认图像宽度，未指定
            patch_size = None,  # 默认patch大小，未指定
            embed_dims = 384,  # `args.dim`没有在您提供的参数解析器中直接列出，请指定一个默认值或确认是否有误
            mlp_ratios = 4,  # 默认MLP比率
            in_channels = 3,  # 默认输入通道数
            qkv_bias = False,  # qkv偏置，默认未指定，这里设为False
            depths = 6,  # 默认层数
            sr_ratios = 1,  # 默认sr比率，未在参数解析器直接列出，这里设为1
            spike_mode = "lif",  # 默认脉冲模式
            dvs_mode = False,  # `args.dvs_mode`没有在您提供的参数解析器中直接列出，请指定一个默认值或确认是否有误
            TET = False,  # 默认TET设置

            )
    checkpoint = torch.load("/vol5/Coin-MLP-back/Pure-MLP-SNN/18M-with-downsample-conv/checkpoint-308.pth.tar", map_location="cpu")
    state_dict = clean_state_dict(checkpoint["state_dict"])
    encoder.load_state_dict(state_dict, strict=False)
    x = torch.randn(3, 3, 3, 224, 480)
    encoder = encoder.cuda()
    y_1,y_2 = encoder(x.cuda())
    print(y_1.shape)
    print(y_2.shape)
        

if __name__ == "__main__":
    main()