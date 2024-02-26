"""
IMPALA:
    Paper: Scalable Distributed Deep-RL with Importance Weighted Actor
    Reference code: https://github.com/facebookresearch/torchbeast

Nauture CNN:
    Code: https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/common/policies.py
"""


import numpy as np
import torch.nn as nn, torch, torch.nn.functional as F
from torch import distributions as torchd
import math
from maniskill2_learn.utils.torch import ExtendedModule, no_grad, ExtendedSequential

from ..builder import BACKBONES
from ..modules import build_norm_layer, need_bias, build_activation_layer
from ...networks import utils_dreamer


class CNNBase(ExtendedModule):
    @no_grad
    def preprocess(self, inputs):
        # assert inputs are channel-first; output is channel-first
        if isinstance(inputs, dict):
            feature = []
            if "rgb" in inputs:
                # inputs images must not have been normalized before
                feature.append(inputs["rgb"] / 255.0)
            if "depth" in inputs:
                depth = inputs["depth"]
                if isinstance(depth, torch.Tensor):
                    feature.append(depth.float())
                elif isinstance(depth, np.ndarray):
                    feature.append(depth.astype(np.float32))
                else:
                    raise NotImplementedError()
            if "seg" in inputs:
                feature.append(inputs["seg"])
            feature = torch.cat(feature, dim=1)
        else:
            feature = inputs
        return feature


# @BACKBONES.register_module()
# class IMPALA(CNNBase):


@BACKBONES.register_module()
class IMPALA(CNNBase):
    def __init__(self, in_channel, image_size, out_feature_size=256, out_channel=None):
        super(IMPALA, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.feat_convs = []
        self.resnet1 = []
        self.resnet2 = []
        self.convs = []
        in_channel = in_channel
        fcs = [64, 64, 64]

        self.stem = nn.Conv2d(in_channel, fcs[0], kernel_size=4, stride=4)
        in_channel = fcs[0]

        for num_ch in fcs:
            feats_convs = []
            feats_convs.append(
                nn.Conv2d(
                    in_channels=in_channel,
                    out_channels=num_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            feats_convs.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            self.feat_convs.append(nn.Sequential(*feats_convs))
            in_channel = num_ch
            for i in range(2):
                resnet_block = []
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=in_channel,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=in_channel,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                if i == 0:
                    self.resnet1.append(nn.Sequential(*resnet_block))
                else:
                    self.resnet2.append(nn.Sequential(*resnet_block))

        self.feat_convs = nn.ModuleList(self.feat_convs)
        self.resnet1 = nn.ModuleList(self.resnet1)
        self.resnet2 = nn.ModuleList(self.resnet2)
        self.img_feat_size = (
            math.ceil(image_size[0] / (2 ** len(fcs) * 4)) * math.ceil(image_size[1] / (2 ** len(fcs) * 4)) * fcs[-1]
        )

        self.fc = nn.Linear(self.img_feat_size, out_feature_size)
        self.final = nn.Linear(out_feature_size, self.out_channel) if out_channel else None

    def forward(self, inputs, **kwargs):
        feature = self.preprocess(inputs)

        x = self.stem(feature)
        # x = feature
        res_input = None
        for i, fconv in enumerate(self.feat_convs):
            x = fconv(x)
            res_input = x
            x = self.resnet1[i](x)
            x += res_input
            res_input = x
            x = self.resnet2[i](x)
            x += res_input

        x = F.relu(x)
        x = x.reshape(x.shape[0], self.img_feat_size)
        x = F.relu(self.fc(x))

        if self.final:
            x = self.final(x)

        return x


@BACKBONES.register_module()
class NatureCNN(CNNBase):
    # DQN
    def __init__(
        self,
        in_channel,
        image_size,
        mlp_spec=[32, 64, 64],
        out_channel=None,
        norm_cfg=dict(type="LN2d"),
        act_cfg=dict(type="ReLU"),
        **kwargs,
    ):
        super(NatureCNN, self).__init__()
        assert len(mlp_spec) == 3, "Nature Net only contain 3 layers"
        with_bias = need_bias(norm_cfg)
        self.net = ExtendedSequential(
            *[
                nn.Conv2d(in_channel, mlp_spec[0], 8, 4, bias=with_bias),
                build_norm_layer(norm_cfg, mlp_spec[0])[1],
                build_activation_layer(act_cfg),
                nn.Conv2d(mlp_spec[0], mlp_spec[1], 4, 2, bias=with_bias),
                build_norm_layer(norm_cfg, mlp_spec[1])[1],
                build_activation_layer(act_cfg),
                nn.Conv2d(mlp_spec[1], mlp_spec[2], 3, 1, bias=with_bias),
                build_norm_layer(norm_cfg, mlp_spec[2])[1],
                build_activation_layer(act_cfg),
                nn.Flatten(1),
            ]
        )
        with torch.no_grad():
            image = torch.zeros([1, in_channel] + list(image_size), device=self.device)
        feature_size = self.net(image).shape[-1]
        self.net.append_list(
            [
                nn.Linear(feature_size, 512),
                build_activation_layer(act_cfg),
            ]
        )
        if out_channel is not None:
            self.net.append_list([nn.Linear(512, 256), build_activation_layer(act_cfg), nn.Linear(256, out_channel)])

        if "conv_init_cfg" in kwargs:
            self.init_weights(self.convs, kwargs["conv_init_cfg"])

    def forward(self, inputs, **kwargs):
        feature = self.preprocess(inputs)
        return self.net(feature)

    def init_weights(self, conv_init_cfg=None):
        if conv_init_cfg is not None:
            init = conv_init_cfg(conv_init_cfg)
            init(self.convs)


# From GitHub implementation of PyTorch Dreamer
@BACKBONES.register_module()
class ConvEncoder(nn.Module):
    def __init__(self, channels, 
                 depth=32, 
                 act_cfg=dict(type="ReLU"),  
                 kernels=(4, 3, 4, 4), 
                 stride=(1,2,2,2),
                 norm_cfg=dict(type="LN2d"), # NOTE (Lukas): Add layer normalization
                 ):
        super(ConvEncoder, self).__init__()
        self._act_cfg = act_cfg
        self._depth = depth
        self._kernels = kernels
        self._stride = stride

        """ Processing Size Visualization for image shape (6, 32, 32):
        IMG_SIZE: 32x32 --> 29x29 --> 14x14 --> 6x6 --> 2x2 
        CHANNELS:     6 -->   2^5 -->   2^6 --> 2^7 --> 2^8 
        """
        with_bias = need_bias(norm_cfg)
        # NOTE (Lukas): Layer normalization does NOT need bias.
        layers = []
        for i, (kernel, stride) in enumerate(zip(self._kernels, self._stride)):
            if i == 0:
                inp_dim = channels
            else:
                inp_dim = 2 ** (i - 1) * self._depth
            depth = 2**i * self._depth
            layers.append(nn.Conv2d(inp_dim, depth, kernel, stride, bias=with_bias)) 
            layers.append(build_norm_layer(norm_cfg, depth)[1])
            layers.append(build_activation_layer(act_cfg))
        self.layers = nn.Sequential(*layers)

    def __call__(self, obs):
        # Supports batches.
        x = obs["image"].reshape((-1,) + tuple(obs["image"].shape[-3:]))
        # NOTE (Lukas): There is no need to permute here, channel is already at front.
        # x = x.permute(0, 3, 1, 2) 
        x = self.layers(x)
        x = x.reshape([x.shape[0], np.prod(x.shape[1:])])
        shape = list(obs["image"].shape[:-3]) + [x.shape[-1]]
        return x.reshape(shape)


@BACKBONES.register_module()
class ConvDecoder(nn.Module):
    def __init__(
        self, 
        inp_depth, 
        depth=32, 
        act_cfg=dict(type="ReLU"), 
        norm_cfg=dict(type="LN2d"), # NOTE (Lukas): Add layer normalization
        shape=(6, 32, 32), 
        kernels=(3, 3, 3, 4), 
        stride=(1,2,2,2), 
        thin=True,
    ):
        super(ConvDecoder, self).__init__()
        self._inp_depth = inp_depth
        self._act_cfg = act_cfg
        self._depth = depth
        self._shape = shape
        self._kernels = kernels
        self._stride = stride
        self._thin = thin

        if self._thin:
            self._linear_layer = nn.Linear(inp_depth, 32 * self._depth)
        else:
            self._linear_layer = nn.Linear(inp_depth, 128 * self._depth)
        inp_dim = 32 * self._depth
        """ Embedding Size Visualization for image shape (6, 32, 32):
        IMG_SIZE: Feature Size -->  1x1 --> 3x3 --> 7x7 --> 15x15 --> 32x32 
        CHANNELS:            6 --> 2^10 --> 2^7 --> 2^6 --> 2^5   --> 6 
        """
        cnnt_layers = []
        for i, (kernel, stride) in enumerate(zip(self._kernels, self._stride)):
            depth = 2 ** (len(self._kernels) - i - 2) * self._depth
            act_cfg = self._act_cfg
            if i == len(self._kernels) - 1:
                # depth = self._shape[-1]
                depth = self._shape[0]
                act_cfg = None
            if i != 0:
                inp_dim = 2 ** (len(self._kernels) - (i - 1) - 2) * self._depth
            cnnt_layers.append(nn.ConvTranspose2d(inp_dim, depth, kernel, stride))
            if norm_cfg is not None:
                cnnt_layers.append(build_norm_layer(norm_cfg, depth)[1])
            if act_cfg is not None:
                cnnt_layers.append(build_activation_layer(act_cfg))
        self._cnnt_layers = nn.Sequential(*cnnt_layers)

    def __call__(self, features, dtype=None):
        if self._thin:
            x = self._linear_layer(features)
            x = x.reshape([-1, 1, 1, 32 * self._depth])
            # NOTE: In contrast to Encoder, we definitely need to permute here because of the reshape above!
            x = x.permute(0, 3, 1, 2)
        else:
            x = self._linear_layer(features)
            x = x.reshape([-1, 2, 2, 32 * self._depth])
            x = x.permute(0, 3, 1, 2)
        x = self._cnnt_layers(x)
        # x.shape = (traj_len, channel, width, height)
        mean = x.reshape(features.shape[:-1] + self._shape)
        # TODO (Lukas): In our case: features.shape=(200,250). But the author must have expected 3 dims.
        # Check if we set features.shape correctly. 
        # NOTE: Permuting in the following is false, because we also didn't permute in encoder.
        """
        if len(mean.shape) == 5:
            # NOTE: Check if this breakpoint is ever hit. 
            mean = mean.permute(0, 1, 3, 4, 2) 
        elif len(mean.shape) == 4:
            mean = mean.permute(0, 2, 3, 1)
        """
        return utils_dreamer.ContDist(torchd.independent.Independent(torchd.normal.Normal(mean, 1), len(self._shape)))
