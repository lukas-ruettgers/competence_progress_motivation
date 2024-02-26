from maniskill2_learn.networks.modules.norm import need_bias
import torch.nn as nn, torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from einops.layers.torch import Rearrange
import torch
from torch import distributions as torchd
import numpy as np

from maniskill2_learn.utils.meta import get_logger
from maniskill2_learn.utils.torch import load_checkpoint, ExtendedModule

from ..builder import BACKBONES
from ..modules.block_utils import ConvModule, MLP, SharedMLP
from ..modules.weight_init import build_init
from ..modules.activation import build_activation_layer
from ..modules.norm import build_norm_layer

from ...networks import utils_dreamer

BACKBONES.register_module(name="MLP", module=MLP)
BACKBONES.register_module(name="SharedMLP", module=SharedMLP)


@BACKBONES.register_module()
class LinearMLP(ExtendedModule):
    def __init__(
        self,
        mlp_spec,
        norm_cfg=dict(type="LN1d"),  # Change BN -> LN
        bias="auto",
        act_cfg=dict(type="ReLU"),
        inactivated_output=True,
        zero_init_output=False,
        pretrained=None,
        linear_init_cfg=None,
        norm_init_cfg=None,
    ):
        super(LinearMLP, self).__init__()
        self.mlp = nn.Sequential()
        for i in range(len(mlp_spec) - 1):
            if i == len(mlp_spec) - 2 and inactivated_output:
                act_cfg = None
                norm_cfg = None
            bias_i = need_bias(norm_cfg) if bias == "auto" else bias
            self.mlp.add_module(f"linear{i}", nn.Linear(mlp_spec[i], mlp_spec[i + 1], bias=bias_i))
            if norm_cfg:
                self.mlp.add_module(f"norm{i}", build_norm_layer(norm_cfg, mlp_spec[i + 1])[1])
            if act_cfg:
                self.mlp.add_module(f"act{i}", build_activation_layer(act_cfg))
        self.init_weights(pretrained, linear_init_cfg, norm_init_cfg)
        if zero_init_output:
            last_linear = self.last_linear
            if last_linear is not None:
                nn.init.zeros_(last_linear.bias)
                last_linear.weight.data.copy_(0.01 * last_linear.weight.data)

    @property
    def last_linear(self):
        last_linear = None
        for m in self.modules():
            if isinstance(m, nn.Linear):
                last_linear = m
        return last_linear

    def forward(self, input, **kwargs):
        return self.mlp(input)

    def init_weights(self, pretrained=None, linear_init_cfg=None, norm_init_cfg=None):
        if isinstance(pretrained, str):
            logger = get_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            linear_init = build_init(linear_init_cfg) if linear_init_cfg else None
            norm_init = build_init(norm_init_cfg) if norm_init_cfg else None

            for m in self.modules():
                if isinstance(m, nn.Linear) and linear_init:
                    linear_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)) and norm_init:
                    norm_init(m)
        else:
            raise TypeError("pretrained must be a str or None")


@BACKBONES.register_module()
class ConvMLP(ExtendedModule):
    def __init__(
        self,
        mlp_spec,
        norm_cfg=dict(type="BN1d"),
        bias="auto",
        act_cfg=dict(type="ReLU"),
        inactivated_output=True,
        pretrained=None,
        conv_init_cfg=None,
        norm_init_cfg=None,
    ):
        super(ConvMLP, self).__init__()
        self.mlp = nn.Sequential()
        for i in range(len(mlp_spec) - 1):
            if i == len(mlp_spec) - 2 and inactivated_output:
                act_cfg = None
                norm_cfg = None
            bias_i = need_bias(norm_cfg) if bias == "auto" else bias
            if norm_cfg is not None and norm_cfg.get("type") == "LN":
                self.mlp.add_module(f"conv{i}", nn.Conv1d(mlp_spec[i], mlp_spec[i + 1], 1, bias=bias_i))
                self.mlp.add_module(f"tranpose{i}-1", Rearrange("b c n -> b n c"))
                self.mlp.add_module(f"ln{i}", build_norm_layer(norm_cfg, num_features=mlp_spec[i + 1])[1])
                self.mlp.add_module(f"tranpose{i}-2", Rearrange("b n c -> b c n"))
                self.mlp.add_module(f"act{i}", build_activation_layer(act_cfg))
            else:
                self.mlp.add_module(
                    f"layer{i}",
                    ConvModule(
                        mlp_spec[i],
                        mlp_spec[i + 1],
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        dilation=1,
                        groups=1,
                        bias=bias_i,
                        conv_cfg=dict(type="Conv1d"),
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        inplace=True,
                        with_spectral_norm=False,
                        padding_mode="zeros",
                        order=("dense", "norm", "act"),
                    ),
                )
        self.init_weights(pretrained, conv_init_cfg, norm_init_cfg)

    def forward(self, input, **kwargs):
        return self.mlp(input)

    def init_weights(self, pretrained=None, conv_init_cfg=None, norm_init_cfg=None):
        if isinstance(pretrained, str):
            logger = get_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            conv_init = build_init(conv_init_cfg) if conv_init_cfg else None
            norm_init = build_init(norm_init_cfg) if norm_init_cfg else None

            for m in self.modules():
                if isinstance(m, nn.Conv1d) and conv_init:
                    conv_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)) and norm_init:
                    norm_init(m)
        else:
            raise TypeError("pretrained must be a str or None")


@BACKBONES.register_module()
class DenseHead(ExtendedModule):
    def __init__(self, inp_dim, shape, layers, units, 
                 act_cfg=dict(type="ELU"), 
                 dist="normal", 
                 std="learned",
                 norm_cfg=dict(type="BN1d"), # NOTE (Lukas): I added this from ConvMLP.
                 bias="auto"
                ):
        # NOTE (Lukas): I replaced std=1.0 by std="learned", to see whether learning the std improves the result.
        super(DenseHead, self).__init__()
        self._inp_dim = inp_dim
        self._shape = (shape,) if isinstance(shape, int) else shape
        if len(self._shape) == 0:
            self._shape = (1,)
        self._layers = layers
        self._units = units
        self._act_cfg = act_cfg
        self._dist = dist
        self._std = std
        
        # NOTE (Lukas): Batch normalization does NOT need bias.
        bias_i = need_bias(norm_cfg) if bias == "auto" else bias
        self._mean_layers = nn.Sequential()
        for index in range(self._layers):
            self._mean_layers.add_module(f"linear{index}", nn.Linear(inp_dim, self._units, bias=bias_i))
            self._mean_layers.add_module(f"ln{index}", build_norm_layer(norm_cfg, num_features=self._units)[1])
            self._mean_layers.add_module(f"act{index}", build_activation_layer(act_cfg))
            # Adjust dimension for consistency
            if index == 0:
                inp_dim = self._units
        # NOTE (Lukas): In last layer, there is no act and norm.
        self._mean_layers.add_module(f"linear_{self._layers}", nn.Linear(inp_dim, np.prod(self._shape), bias=True))


        if self._std == "learned":
            self._std_layer = nn.Linear(self._inp_dim, np.prod(self._shape))

    def __call__(self, features, dtype=None):
        x = features
        mean = self._mean_layers(x)
        if self._std == "learned":
            std = self._std_layer(x)
            std = F.softplus(std) + 0.01
        else:
            std = self._std
        if self._dist == "normal":
            return utils_dreamer.ContDist(torchd.independent.Independent(torchd.normal.Normal(mean, std), len(self._shape)))
        if self._dist == "huber":
            return utils_dreamer.ContDist(
                torchd.independent.Independent(utils_dreamer.UnnormalizedHuber(mean, std, 1.0), len(self._shape))
            )
        if self._dist == "binary":
            return utils_dreamer.Bernoulli(
                torchd.independent.Independent(torchd.bernoulli.Bernoulli(logits=mean), len(self._shape))
            )
        raise NotImplementedError(self._dist)


@BACKBONES.register_module()
class ActionHead(ExtendedModule):
    def __init__(
        self,
        inp_dim,
        size,
        layers,
        units,
        act_cfg=dict(type="ELU"),
        norm_cfg=dict(type="BN1d"), # NOTE (Lukas): I added this from ConvMLP.
        bias="auto",
        dist="trunc_normal",
        init_std=0.0,
        min_std=0.1,
        action_disc=5,
        temp=0.1,
        outscale=0,
    ):
        super(ActionHead, self).__init__()
        self._size = size
        self._layers = layers
        self._units = units
        self._dist = dist
        self._act_cfg = act_cfg
        self._norm_cfg = norm_cfg
        self._min_std = min_std
        self._init_std = init_std
        self._action_disc = action_disc
        self._temp = temp() if callable(temp) else temp
        self._outscale = outscale
        
        bias_i = need_bias(norm_cfg) if bias == "auto" else bias
        self._pre_layers = nn.Sequential()
        for index in range(self._layers):
            self._pre_layers.add_module(f"linear{index}", nn.Linear(inp_dim, self._units, bias=bias_i))
            if norm_cfg is not None:
                self._pre_layers.add_module(f"ln{index}", build_norm_layer(norm_cfg, num_features=self._units)[1])
            if act_cfg is not None:
                self._pre_layers.add_module(f"act{index}", build_activation_layer(self._act_cfg))
            # Adjust dimension for consistency
            if index == 0:
                inp_dim = self._units

        if self._dist in ["tanh_normal", "tanh_normal_5", "normal", "trunc_normal"]:
            self._dist_layer = nn.Linear(self._units, 2 * self._size)
        elif self._dist in ["normal_1", "onehot", "onehot_gumbel"]:
            self._dist_layer = nn.Linear(self._units, self._size)

    def __call__(self, features, dtype=None):
        x = features
        x = self._pre_layers(x)
        if self._dist == "tanh_normal":
            x = self._dist_layer(x)
            mean, std = torch.split(x, 2, -1)
            mean = torch.tanh(mean)
            std = F.softplus(std + self._init_std) + self._min_std
            dist = torchd.normal.Normal(mean, std)
            dist = torchd.transformed_distribution.TransformedDistribution(dist, utils_dreamer.TanhBijector())
            dist = torchd.independent.Independent(dist, 1)
            dist = utils_dreamer.SampleDist(dist)
        elif self._dist == "tanh_normal_5":
            x = self._dist_layer(x)
            mean, std = torch.split(x, 2, -1)
            mean = 5 * torch.tanh(mean / 5)
            std = F.softplus(std + 5) + 5
            dist = torchd.normal.Normal(mean, std)
            dist = torchd.transformed_distribution.TransformedDistribution(dist, utils_dreamer.TanhBijector())
            dist = torchd.independent.Independent(dist, 1)
            dist = utils_dreamer.SampleDist(dist)
        elif self._dist == "normal":
            x = self._dist_layer(x)
            mean, std = torch.split(x, 2, -1)
            std = F.softplus(std + self._init_std) + self._min_std
            dist = torchd.normal.Normal(mean, std)
            dist = utils_dreamer.ContDist(torchd.independent.Independent(dist, 1))
        elif self._dist == "normal_1":
            x = self._dist_layer(x)
            dist = torchd.normal.Normal(mean, 1)
            dist = utils_dreamer.ContDist(torchd.independent.Independent(dist, 1))
        elif self._dist == "trunc_normal":
            x = self._dist_layer(x)
            mean, std = torch.split(x, [self._size] * 2, -1)
            mean = torch.tanh(mean)
            std = 2 * torch.sigmoid(std / 2) + self._min_std
            dist = utils_dreamer.SafeTruncatedNormal(mean, std, -1, 1)
            dist = utils_dreamer.ContDist(torchd.independent.Independent(dist, 1))
        elif self._dist == "onehot":
            x = self._dist_layer(x)
            dist = utils_dreamer.OneHotDist(x)
        elif self._dist == "onehot_gumble":
            x = self._dist_layer(x)
            temp = self._temp
            dist = utils_dreamer.ContDist(torchd.gumbel.Gumbel(x, 1 / temp))
        else:
            raise NotImplementedError(self._dist)
        return dist
