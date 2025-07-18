# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Common modules
"""

import json
import math
import platform
import warnings
from collections import OrderedDict, namedtuple
from copy import copy
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torchvision
import yaml
from PIL import Image
from torch.cuda import amp

from utils.dataloaders import exif_transpose, letterbox
from utils.general import (LOGGER, check_requirements, check_suffix, check_version, colorstr, increment_path,
                           make_divisible, non_max_suppression, scale_coords, xywh2xyxy, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import copy_attr, time_sync
from models.SwinTransformer import SwinTransformerLayer
import torch.nn.functional as F

from models.CFPNET import *
from models.ODConv import ODConv2d
from models.BiFormer import BiFormerBlock


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


# 基础卷积 CBL = Conv-BN-ReLU(LeakyRelu | SiLu)
class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, dilation=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False, dilation=dilation)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DeConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=False):
        super().__init__()
        self.conv = nn.ConvTranspose2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)

    def forward(self, x):
        return self.bn(self.conv(x))


# class DWConv(Conv):
#     # Depth-wise convolution class
#     def __init__(self, c1, c2, k=1, s=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
#         super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    # Depth-wise transpose convolution class
    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


# Bottleneck 残差单元 CBL + CBL + add
class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


# CSP1_x模块
class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class CrossConv(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        # ch_in, ch_out, kernel, stride, groups, expansion, shortcut
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


# SPD conv
class SPD_conv(nn.Module):
    def __init__(self, dimension=1):
        super(SPD_conv, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)


# Concat_BiFPN
class Concat_BiFPN(nn.Module):
    def __init__(self, c1, c2):
        super(Concat_BiFPN, self).__init__()
        self.w1_weight = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w2_weight = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.conv = Conv(c1, c2, 1, 1, 0)
        self.act = nn.SiLU()

    def forward(self, x):
        if len(x) == 2:
            w = F.relu(self.w1_weight)
            weight = w / (torch.sum(w, dim=0) + self.epsilon)
            x = self.conv(self.act(weight[0] * x[0] + weight[1] * x[1]))
        elif len(x) == 3:
            w = F.relu(self.w2_weight)
            weight = w / (torch.sum(w, dim=0) + self.epsilon)
            x = self.conv(self.act(weight[0] * x[0] + weight[1] * x[1] + weight[2] * x[2]))

        return x


# 改用yolox中的decoupled head 解耦头
class DecoupledHead(nn.Module):
    def __init__(self, ch=256, nc=80, anchors=()):
        super(DecoupledHead, self).__init__()
        self.nc = nc  # 类别数量
        self.nl = len(anchors)  # 检测层的数量
        self.na = len(anchors[0]) // 2  # anchor的数量
        self.merge = Conv(ch, 256, 1, 1)
        self.cls_convs1 = Conv(256, 256, 3, 1, 1)
        self.cls_convs2 = Conv(256, 256, 3, 1, 1)
        self.reg_convs1 = Conv(256, 256, 3, 1, 1)
        self.reg_convs2 = Conv(256, 256, 3, 1, 1)

        self.cls_preds = nn.Conv2d(256, self.nc * self.na, 1)
        self.reg_preds = nn.Conv2d(256, 4 * self.na, 1)
        self.obj_preds = nn.Conv2d(256, 1 * self.na, 1)

    def forward(self, x):
        x = self.merge(x)

        x_cls = self.cls_convs1(x)
        x_cls = self.cls_convs2(x_cls)
        x_cls = self.cls_preds(x_cls)

        x_reg = self.reg_convs1(x)
        x_reg = self.reg_convs2(x_reg)
        x_reg1 = self.reg_preds(x_reg)
        x_obj = self.obj_preds(x_reg)
        out = torch.cat([x_reg1, x_obj, x_cls], dim=1)

        return out


class LightHead(nn.Module):
    def __init__(self, ch=256, nc=80, anchors=()):
        super(LightHead, self).__init__()
        self.nc = nc  # 类别数量
        self.nl = len(anchors)  # 检测层的数量
        self.na = len(anchors[0]) // 2  # anchor的数量
        self.cls_conv = Conv(ch, ch, 3, 1, 1)
        self.reg_conv = Conv(ch, ch, 3, 1, 1)

        self.cls_preds = nn.Conv2d(ch, self.nc * self.na, 1)
        self.reg_preds = nn.Conv2d(ch, 4 * self.na, 1)
        self.obj_preds = nn.Conv2d(ch, 1 * self.na, 1)

    def forward(self, x):
        x_cls = self.cls_conv(x)
        x_cls = self.cls_preds(x_cls)
        b, _, h, w = x_cls.shape
        x_cls = x_cls.view((-1, self.na, self.nc, h, w)).contiguous()

        x_reg = self.reg_conv(x)
        x_reg1 = self.reg_preds(x_reg)
        x_obj = self.obj_preds(x_reg)

        x_reg1 = x_reg1.view((-1, self.na, 4, h, w)).contiguous()
        x_obj = x_obj.view((-1, self.na, 1, h, w)).contiguous()

        out = torch.cat([x_reg1, x_obj, x_cls], dim=2)
        out = out.view((b, -1, h, w)).contiguous()

        return out


# 通道注意力模块
class ChannelsAttention(nn.Module):
    def __init__(self, c1, reduction=16):
        super(ChannelsAttention, self).__init__()
        mid_channel = c1 // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_channels=c1, out_channels=mid_channel, kernel_size=1, stride=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=mid_channel, out_channels=c1, kernel_size=1, stride=1)
        )

        self.act = nn.Sigmoid()

    def forward(self, x):
        # avgout = self.shared_MLP(self.avg_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)
        # maxout = self.shared_MLP(self.max_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)

        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.act(avgout + maxout)


# 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.act = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat((avgout, maxout), dim=1)
        out = self.act(self.conv2d(out))
        return out


#  CBAM 注意力模块
class CBAM(nn.Module):
    def __init__(self, c1, c2):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelsAttention(c1)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out

        return out
    

#  PCBAM 注意力模块
class PCBAM(nn.Module):
    def __init__(self, c1, c2):
        super(PCBAM, self).__init__()
        self.channel_attention = ChannelsAttention(c1)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        out_c = self.channel_attention(x) * x
        out_s = self.spatial_attention(x) * x

        return x + out_c + out_s


#  坐标注意力模块coordinate attention
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CA(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CA, self).__init__()

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        pool_h = nn.AdaptiveAvgPool2d((h, 1))
        pool_w = nn.AdaptiveAvgPool2d((1, w))
        x_h = pool_h(x)
        x_w = pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class RFACAConv(nn.Module):
    def __init__(self, c1, K=2):
        super(RFACAConv, self).__init__()
        self.K = K
        self.group_conv = Conv(c1, self.K * self.K * c1, k=3, g=4)
        self.conv1 = Conv(c1, c1, 1, 1)

        self.conv_h = nn.Conv2d(c1, c1, kernel_size=1, stride=1)
        self.conv_w = nn.Conv2d(c1, c1, kernel_size=1, stride=1)

        self.conv2 = Conv(c1, c1, k=3, s=self.K)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.group_conv(x)
        x = x.view((B, C, self.K * H, self.K * W))  # adjust shape
        pool_h = nn.AdaptiveAvgPool2d((self.K * H, 1))
        pool_w = nn.AdaptiveAvgPool2d((1, self.K * W))
        x_h = pool_h(x)
        x_w = pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        x_h, x_w = torch.split(y, [self.K * H, self.K * W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        x_h = self.conv_h(x_h).sigmoid()
        x_w = self.conv_w(x_w).sigmoid()

        out = x * x_h * x_w
        out = self.conv2(out)

        return out


class CoordAttn(nn.Module):
    def __init__(self, dim=512, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.key_embed = nn.Sequential(
            Conv(dim, dim, k=kernel_size, p=kernel_size // 2, g=4, act=False),
            nn.ReLU()
        )
        self.value_embed = nn.Sequential(
            Conv(dim, dim, k=1, act=False)
        )
        factor = 4
        self.attention_embed = nn.Sequential(
            Conv(2 * dim, 2 * dim // factor, k=1, act=False),
            nn.ReLU(),
            nn.Conv2d(2 * dim // factor, kernel_size * kernel_size * dim, 1)
        )

    def forward(self, x):
        bs, c, h, w = x.shape
        k1 = self.key_embed(x)  # bs,c,h,w
        v = self.value_embed(x).view(bs, c, -1)  # bs,c,h,w
        y = torch.cat([k1, x], dim=1)  # bs,2c,h,w
        att = self.attention_embed(y)  # bs,c*k*k,h,w
        att = att.reshape(bs, c, self.kernel_size * self.kernel_size, h, w)
        att = att.mean(2, keepdim=False).view(bs, c, -1)  # bs,c,h*w
        k2 = F.softmax(att, dim=-1) * v
        k2 = k2.view(bs, c, h, w)
        return k1 + k2


# CSP1_3模块
class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3_CBAM(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(C3_CBAM, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        self.cbam = CBAM(c_, c_)

    def forward(self, x):
        out1 = self.cv2(x)
        out2 = self.m(self.cv1(x))
        out2 = self.cbam(out2)
        out = torch.cat((out2, out1), dim=1)
        return self.cv3(torch.cat((self.cbam(self.m(self.cv1(x))), self.cv2(x)), dim=1))


class C3_MLP_PCBAM(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(C3_MLP_PCBAM, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.mlp = LightMLPBlock(c_, c_)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        self.pcbam = PCBAM(c2, c2)

    def forward(self, x):
        return self.cv3(self.pcbam(torch.cat((self.m(self.cv1(x)), self.mlp(self.cv2(x))), dim=1)))


class C3_MLP(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(C3_MLP, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.mlp = LightMLPBlock(c_, c_)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.mlp(self.cv2(x))), dim=1))


class C3_PCBAM(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(C3_PCBAM, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        self.pcbam = PCBAM(c2, c2)

    def forward(self, x):
        return self.cv3(self.pcbam(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1)))


class gnconv(nn.Module):
    def __init__(self, dim, c2, k=1, stride=1, order=3, gflayer=None, h=14, w=8, s=1.0):
        super().__init__()
        self.order = order
        self.dims = [dim // 2 ** i for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv2d(dim, 2 * dim, 1)

        if gflayer is None:
            self.dwconv = get_dwconv(sum(self.dims), 7, True)  # DWConv(sum(self.dims), sum(self.dims), 7, 1)
        else:
            self.dwconv = gflayer(sum(self.dims), h=h, w=w)

        self.proj_out = nn.Conv2d(dim, c2, kernel_size=k, stride=stride, padding=k // 2)

        self.pws = nn.ModuleList(
            [nn.Conv2d(self.dims[i], self.dims[i + 1], 1) for i in range(order - 1)]
        )
        self.scale = s
        self.act = nn.SiLU(inplace=True)
        self.bn = nn.BatchNorm2d(c2)

    def forward(self, x, mask=None, dummy=False):
        # B, C, H, W = x.shape gnconv [512]by iscyy/air
        fused_x = self.proj_in(x)
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)
        dw_abc = self.dwconv(abc) * self.scale
        dw_list = torch.split(dw_abc, self.dims, dim=1)
        x = pwa * dw_list[0]
        for i in range(self.order - 1):
            x = self.pws[i](x) * dw_list[i + 1]
        x = self.proj_out(x)
        x = self.act(x)

        return x


def get_dwconv(dim, kernel, bias):
    return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel - 1) // 2, bias=bias, groups=dim)


class HorLayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).# https://ar5iv.labs.arxiv.org/html/2207.14284
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError  # by iscyy/air
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


# def drop_path_f(x, drop_prob: float = 0., training: bool = False):
#     """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
#     This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
#     the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
#     See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
#     changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
#     'survival rate' as the argument.
#     """
#     if drop_prob == 0. or not training:
#         return x
#     keep_prob = 1 - drop_prob
#     shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
#     random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
#     random_tensor.floor_()  # binarize
#     output = x.div(keep_prob) * random_tensor
#     return output
#
# class DropPath(nn.Module):
#     """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
#     """
#
#     def __init__(self, drop_prob=None):
#         super(DropPath, self).__init__()
#         self.drop_prob = drop_prob
#
#     def forward(self, x):
#         return drop_path_f(x, self.drop_prob, self.training)

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class HorBlock(nn.Module):
    r""" HorNet block
    """

    def __init__(self, dim, shortcut=False, layer_scale_init_value=1e-6):
        super().__init__()
        self.shortcut = shortcut
        self.norm1 = HorLayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.gnconv = gnconv(dim, dim)
        self.norm2 = HorLayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 2 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(2 * dim, dim)

        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                   requires_grad=True) if layer_scale_init_value > 0 else None

        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                   requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        B, C, H, W = x.shape  # [512]
        if self.gamma1 is not None:
            gamma1 = self.gamma1.view(C, 1, 1)
        else:
            gamma1 = 1
        x = (x + gamma1 * self.gnconv(self.norm1(x))) if self.shortcut else gamma1 * self.gnconv(self.norm1(x))

        input = x
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma2 is not None:
            x = self.gamma2 * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = (input + x) if self.shortcut else x
        return x


class C3HB(nn.Module):
    # CSP HorBlock with 3 convolutions by iscyy/yoloair
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = gnconv(c1, c_, 1, 1, 5)
        self.cv2 = gnconv(c1, c_, 1, 1, 5)
        self.cv3 = gnconv(2 * c_, c2, 1, 1, 5)
        self.m = nn.Sequential(*(HorBlock(c_) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class C3x(C3):
    # C3 module with cross-convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)))


# Swin Transform
class SwinTransformerBlock(nn.Module):
    def __init__(self, c1, c2, num_heads, num_layers, window_size=8):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)

        # remove input_resolution
        self.blocks = nn.Sequential(*[SwinTransformerLayer(dim=c2, num_heads=num_heads, window_size=window_size,
                                                           shift_size=0 if (i % 2 == 0) else window_size // 2) for i in
                                      range(num_layers)])

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        x = self.blocks(x)
        return x


class C3STR(C3):
    # C3 module with SwinTransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        num_heads = c_ // 32
        self.m = SwinTransformerBlock(c_, c_, num_heads, n)


class C3SPP(C3):
    # C3 module with SPP()
    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SPP(c_, c_, k)


class C3Ghost(C3):
    # C3 module with GhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


# 空间金字塔池化层次
class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

# Focus 模块
class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
                                      Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


def channel_shuffle(x, groups):
    N, C, H, W = x.shape
    channels_per_group = C // groups
    x = x.view(N, groups, channels_per_group, H, W)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(N, -1, H, W)
    return x


class GhostPoolConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.c_ = c_
        self.rep_num = math.ceil(c2 / c1)
        self.in_channels = c1
        self.out_channels = c2
        self.stride = s
        self.cv1 = Conv(c1, c_, k, s, p, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)
        self.cv3 = Conv(c1, c2)
        self.pool = nn.AvgPool2d(3, s, padding=1)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(channel_shuffle(x1, self.c_ // 2))
        x3 = self.pool(x)
        if self.in_channels != self.out_channels:
            x3 = x3.repeat((1, self.rep_num, 1, 1))
            x3 = x3[:, :self.out_channels, :, :]
            # print(x3.shape, x.shape, x1.shape, x2.shape)
            # x4 = torch.cat((x1, x2), dim=1)
            # print(x4.shape)

        return torch.cat((x1, x2), dim=1) + x3


class DenseGhostPoolBottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super(DenseGhostPoolBottleneck, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = GhostPoolConv(c1, c_, 3, 1)
        self.cv2 = GhostPoolConv(c_, c_, 1, 1)
        self.cv3 = GhostPoolConv(c_, c_, 3, 1)
        self.cv4 = GhostPoolConv(c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x + x1)
        x3 = self.cv3(x + x1 + x2)
        x4 = self.cv4(x + x1 + x2 + x3)

        return x4


class C3_DenseGhost_MLP(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(C3_DenseGhost_MLP, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.mlp = LightMLPBlock(c_, c_)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(DenseGhostPoolBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.mlp(self.cv2(x))), dim=1))


#  漏斗形Ghost
class GhostConvP(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.c_ = c_
        self.rep_num = math.ceil(c2 / c1)
        self.in_channels = c1
        self.out_channels = c2
        self.stride = s
        self.cv1 = Conv(c1, c_, k, s, p, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)
        self.cv3 = Conv(c_, c2, 1, 1, p, g, act)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(channel_shuffle(x1, self.c_ // 2))
        x3 = self.cv3(x1)

        return torch.cat((x1, x2), dim=1) + x3


class GhostPBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConvP(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConvP(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
                                      Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class GhostPBottleneck1(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super(GhostPBottleneck1, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = GhostPBottleneck(c1, c_, 3, 1)
        self.cv2 = GhostPBottleneck(c_, c_, 3, 1)
        self.cv3 = GhostPBottleneck(c_, c_, 3, 1)
        self.cv4 = GhostPBottleneck(c_, c2, 3, 1)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x + x1)
        x3 = self.cv3(x + x1 + x2)
        x4 = self.cv4(x + x1 + x2 + x3)

        return x4


class C3_Ghost_MLP(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(C3_Ghost_MLP, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.mlp = LightMLPBlock(c_, c_)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(GhostPBottleneck1(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.mlp(self.cv2(x))), dim=1))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert (h / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(b, s, s, c // s ** 2, h, w)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(b, c // s ** 2, h * s, w * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride=1, p=None):
        super().__init__()
        self.dconv = Conv(in_channels, in_channels, k=ksize, s=stride, p=p, g=in_channels)
        self.pconv = Conv(in_channels, out_channels, k=1, s=1, g=1)

    def forward(self, x):
        return self.pconv(self.dconv(x))


#  LightMLPBlock
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, res_conv=False, groups=1):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        expansion = 4
        c = out_channels // expansion

        self.conv1 = Conv(in_channels, c, k=1, s=1, p=0, act=nn.ReLU(inplace=True))

        self.conv2 = Conv(c, c, k=3, s=stride, p=1, g=groups, act=nn.ReLU(inplace=True))

        self.conv3 = Conv(c, out_channels, k=1, s=1, p=0, act=False)
        self.act3 = nn.ReLU(inplace=True)

        if res_conv:
            self.residual_conv = Conv(in_channels, out_channels, k=1, s=1, p=0, act=False)

        self.res_conv = res_conv

    def forward(self, x, return_x_2=True):
        residual = x
        x = self.conv3(self.conv2(self.conv1(x)))

        if self.res_conv:
            residual = self.residual_conv(residual)

        x += residual
        x = self.act3(x)

        if return_x_2:
            return x
        else:
            return x


class LightMLPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=1, stride=1, mlp_ratio=4.,
                 drop=0., use_layer_scale=True, layer_scale_init_value=1.0, drop_path=0.):
        super().__init__()
        self.dw = DWConv(in_channels, out_channels, ksize=1, stride=1)
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(1, in_channels)
        self.norm2 = nn.GroupNorm(1, in_channels)

        mlp_hidden_dim = int(in_channels * mlp_ratio)
        self.mlp = MLP(in_features=in_channels, hidden_features=mlp_hidden_dim, drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((out_channels)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((out_channels)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.dw(self.norm1(x)))
            x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.dw(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


#  LVCBlock
class LVCBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_codes, channel_ratio=0.25, base_channel=64):
        super(LVCBlock, self).__init__()
        self.out_channels = out_channels
        self.num_codes = num_codes
        num_codes = 64

        self.conv_1 = ConvBlock(in_channels=in_channels, out_channels=in_channels, res_conv=True, stride=1)

        self.LVC = nn.Sequential(
            Conv(in_channels, in_channels, k=1, act=False),
            nn.ReLU(inplace=True),
            CBAM(in_channels, in_channels),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            Conv(in_channels, in_channels, 1, 1, act=False)
        )

    def forward(self, x):
        x = self.conv_1(x, return_x_2=False)
        en = self.LVC(x)
        gam = self.fc(en)
        x = F.relu_(x + x * gam)
        return x


#  EVCBlock
class EVCBlock(nn.Module):
    def __init__(self, in_channels, out_channels, channel_ratio=4, base_channel=16):
        super().__init__()
        expansion = 2
        ch = out_channels * expansion
        # Stem stage: get the feature maps by conv block (copied form resnet.py) 进入conformer框架之前的处理
        self.conv1 = Conv(in_channels, in_channels, k=7, s=1, p=3, act=False)
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # 1 / 4 [56, 56]

        # LVC
        self.lvc = LVCBlock(in_channels=in_channels, out_channels=out_channels, num_codes=64)  # c1值暂时未定

        # self.lvc = nn.Conv2d(in_channels, out_channels, 1, 1, bias=False)# LightMLPBlock

        self.l_MLP = LightMLPBlock(in_channels, out_channels, ksize=1, stride=1,
                                   mlp_ratio=4., drop=0.,
                                   use_layer_scale=True, layer_scale_init_value=1e-5, drop_path=0.)
        self.cnv1 = nn.Conv2d(ch, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.maxpool(self.act1(self.conv1(x)))
        # LVCBlock
        x_lvc = self.lvc(x1)
        # LightMLPBlock
        x_lmlp = self.l_MLP(x1)
        # concat
        x = torch.cat((x_lvc, x_lmlp), dim=1)
        x = self.cnv1(x)
        return x


# RepVGG
class RepConv(nn.Module):
    # Represented convolution
    # https://arxiv.org/abs/2101.03697

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True, deploy=False):
        super(RepConv, self).__init__()

        self.deploy = deploy
        self.groups = g
        self.in_channels = c1
        self.out_channels = c2

        assert k == 3
        assert autopad(k, p) == 1

        padding_11 = autopad(k, p) - k // 2

        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

        if deploy:
            self.rbr_reparam = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=True)

        else:
            # if s == 1 and c1 == c2:
            #     self.rbr_identity = nn.BatchNorm2d(c2)
            # else:
            self.rbr_identity = None

            self.rbr_dense = nn.Sequential(
                nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2),
            )

            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(c1, c2, 1, s, padding_11, groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2),
            )

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.act(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros(
                    (self.in_channels, input_dim, 3, 3), dtype=np.float32
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return (
            kernel.detach().cpu().numpy(),
            bias.detach().cpu().numpy(),
        )

    def fuse_conv_bn(self, conv, bn):

        std = (bn.running_var + bn.eps).sqrt()
        bias = bn.bias - bn.running_mean * bn.weight / std

        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        weights = conv.weight * t

        bn = nn.Identity()
        conv = nn.Conv2d(in_channels=conv.in_channels,
                         out_channels=conv.out_channels,
                         kernel_size=conv.kernel_size,
                         stride=conv.stride,
                         padding=conv.padding,
                         dilation=conv.dilation,
                         groups=conv.groups,
                         bias=True,
                         padding_mode=conv.padding_mode)

        conv.weight = torch.nn.Parameter(weights)
        conv.bias = torch.nn.Parameter(bias)
        return conv

    def fuse_repvgg_block(self):
        if self.deploy:
            return
        print(f"RepConv.fuse_repvgg_block")

        self.rbr_dense = self.fuse_conv_bn(self.rbr_dense[0], self.rbr_dense[1])

        self.rbr_1x1 = self.fuse_conv_bn(self.rbr_1x1[0], self.rbr_1x1[1])
        rbr_1x1_bias = self.rbr_1x1.bias
        weight_1x1_expanded = torch.nn.functional.pad(self.rbr_1x1.weight, [1, 1, 1, 1])

        # Fuse self.rbr_identity
        if (isinstance(self.rbr_identity, nn.BatchNorm2d) or isinstance(self.rbr_identity,
                                                                        nn.modules.batchnorm.SyncBatchNorm)):
            # print(f"fuse: rbr_identity == BatchNorm2d or SyncBatchNorm")
            identity_conv_1x1 = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=self.groups,
                bias=False)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.to(self.rbr_1x1.weight.data.device)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.squeeze().squeeze()
            # print(f" identity_conv_1x1.weight = {identity_conv_1x1.weight.shape}")
            identity_conv_1x1.weight.data.fill_(0.0)
            identity_conv_1x1.weight.data.fill_diagonal_(1.0)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.unsqueeze(2).unsqueeze(3)
            # print(f" identity_conv_1x1.weight = {identity_conv_1x1.weight.shape}")

            identity_conv_1x1 = self.fuse_conv_bn(identity_conv_1x1, self.rbr_identity)
            bias_identity_expanded = identity_conv_1x1.bias
            weight_identity_expanded = torch.nn.functional.pad(identity_conv_1x1.weight, [1, 1, 1, 1])
        else:
            # print(f"fuse: rbr_identity != BatchNorm2d, rbr_identity = {self.rbr_identity}")
            bias_identity_expanded = torch.nn.Parameter(torch.zeros_like(rbr_1x1_bias))
            weight_identity_expanded = torch.nn.Parameter(torch.zeros_like(weight_1x1_expanded))

            # print(f"self.rbr_1x1.weight = {self.rbr_1x1.weight.shape}, ")
        # print(f"weight_1x1_expanded = {weight_1x1_expanded.shape}, ")
        # print(f"self.rbr_dense.weight = {self.rbr_dense.weight.shape}, ")

        self.rbr_dense.weight = torch.nn.Parameter(
            self.rbr_dense.weight + weight_1x1_expanded + weight_identity_expanded)
        self.rbr_dense.bias = torch.nn.Parameter(self.rbr_dense.bias + rbr_1x1_bias + bias_identity_expanded)

        self.rbr_reparam = self.rbr_dense
        self.deploy = True

        if self.rbr_identity is not None:
            del self.rbr_identity
            self.rbr_identity = None

        if self.rbr_1x1 is not None:
            del self.rbr_1x1
            self.rbr_1x1 = None

        if self.rbr_dense is not None:
            del self.rbr_dense
            self.rbr_dense = None


class RepBlock(nn.Module):
    def __init__(self, c1, c2, n=0):
        super(RepBlock, self).__init__()
        c_ = int(c1 * 0.5)
        self.c_ = c_
        self.conv1 = Conv(c1, c2, 1, 1)
        self.conv2 = Conv(2 * c1, c2, 1, 1)
        self.rep1 = RepConv(c_, c_, 3, 1)
        self.rep2 = RepConv(c_, c_, 3, 1)
        self.rep3 = RepConv(c_, c_, 3, 1)

    def forward(self, x):
        x = self.conv1(x)
        x1, x2 = torch.split(x, (self.c_, self.c_), dim=1)
        x2 = self.rep1(x2)
        x3 = self.rep2(x2)
        x4 = self.rep3(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        return self.conv2(x)


class RepConv1x1(nn.Module):
    def __init__(self, c1, c2, k, s, p=None, g=1, act=True, deploy=False):
        super().__init__()
        assert c1 == c2, "Input and output have not same dimension"

        self.deploy = deploy
        self.in_channels = c1
        self.out_channels = c2
        self.groups = g

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(c1, c2, k, s, autopad(k, p), g),
            nn.BatchNorm2d(c2)
        )
        self.re_param = None
        self.identity = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        if self.re_param is not None:
            return self.act(self.re_param(x))
        else:
            return self.act(self.conv1x1(x) + self.identity(x))

    def fuse_conv_bn(self, conv, bn):

        std = (bn.running_var + bn.eps).sqrt()
        bias = bn.bias - bn.running_mean * bn.weight / std

        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        weights = conv.weight * t

        bn = nn.Identity()
        conv = nn.Conv2d(in_channels=conv.in_channels,
                         out_channels=conv.out_channels,
                         kernel_size=conv.kernel_size,
                         stride=conv.stride,
                         padding=conv.padding,
                         dilation=conv.dilation,
                         groups=conv.groups,
                         bias=True,
                         padding_mode=conv.padding_mode)

        conv.weight = torch.nn.Parameter(weights)
        conv.bias = torch.nn.Parameter(bias)
        return conv

    def fuse_repvgg_block(self):
        if self.deploy:
            return
        print(f"RepConv1x1.fuse_repvgg_block")

        self.conv1x1 = self.fuse_conv_bn(self.conv1x1[0], self.conv1x1[1])

        # Fuse self.rbr_identity
        identity_conv_1x1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=self.groups,
            bias=False)
        identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.to(self.conv1x1.weight.data.device)
        identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.squeeze().squeeze()

        identity_conv_1x1.weight.data.fill_(0.0)
        identity_conv_1x1.weight.data.fill_diagonal_(1.0)
        identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.unsqueeze(2).unsqueeze(3)

        identity_conv_1x1 = self.fuse_conv_bn(identity_conv_1x1, self.identity)

        self.conv1x1.weight = torch.nn.Parameter(
            self.conv1x1.weight + identity_conv_1x1.weight)
        self.conv1x1.bias = torch.nn.Parameter(self.conv1x1.bias + identity_conv_1x1.bias)

        self.re_param = self.conv1x1
        self.deploy = True

        if self.identity is not None:
            del self.rbr_identity
            self.identity = None

        if self.conv1x1 is not None:
            del self.conv1x1
            self.conv1x1 = None


# class RepBlock(nn.Module):
#     def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
#         super(RepBlock, self).__init__()
#         c_ = int(c2 * e)
#         self.cv1 = Conv(c1, c_, k=1, s=1)#RepConv(c1, c_, k=3, s=1, g=g)
#         self.cv2 = RepConv(c_, c2, k=3, s=1, g=g)
#         self.add = shortcut and c1 == c2
#
#     def forward(self, x):
#         return x + self.cv2(self.cv1(x)) if self.add else self.cv2(x) + self.cv1(x)

class C3_REP(C3):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(C3_REP, self).__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(RepBlock(c_, c_, shortcut, g, e=1) for _ in range(n)))


class C3_REP_CBAM(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(C3_REP_CBAM, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(RepBlock(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        self.cbam = CBAM(c_, c_)

    def forward(self, x):
        out1 = self.cv2(x)
        out2 = self.m(self.cv1(x))
        return self.cv3(torch.cat((self.cbam(self.m(self.cv1(x))), self.cv2(x)), dim=1))


#  RepGhost
def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.0).clamp_(0.0, 6.0).div_(6.0)
    else:
        return F.relu6(x + 3.0) / 6.0


class SqueezeExcite(nn.Module):
    def __init__(
            self,
            in_chs,
            se_ratio=0.25,
            reduced_base_chs=None,
            act_layer=nn.SiLU,
            gate_fn=hard_sigmoid,
            divisor=4,
            **_,
    ):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible(
            (reduced_base_chs or in_chs) * se_ratio, divisor,
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


class RepGhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, deploy=False):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()

        self.deploy = deploy
        self.in_channels = c1
        self.out_channels = c2
        self.groups = g

        self.conv1 = Conv(c1, c2, k, s, g=g)

        self.rbr_dense = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=5, stride=1, padding=2,
                      groups=self.out_channels),
            nn.BatchNorm2d(self.out_channels)
        )
        self.rbr_1x1 = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, 1, 1, groups=self.out_channels),
            nn.BatchNorm2d(self.out_channels)
        )
        self.rbr_identity = nn.BatchNorm2d(self.out_channels)
        self.act = nn.SiLU(inplace=True)

        self.rbr_reparam = None

    def forward(self, x):
        x1 = self.conv1(x)
        if self.rbr_reparam is not None:
            return self.act(self.rbr_reparam(x1))
        else:
            x2 = self.rbr_dense(x1)
            x3 = self.rbr_identity(x1)
            return self.act(x2 + x3)

    def fuse_conv_bn(self, conv, bn):

        std = (bn.running_var + bn.eps).sqrt()
        bias = bn.bias - bn.running_mean * bn.weight / std

        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        weights = conv.weight * t

        bn = nn.Identity()
        conv = nn.Conv2d(in_channels=conv.in_channels,
                         out_channels=conv.out_channels,
                         kernel_size=conv.kernel_size,
                         stride=conv.stride,
                         padding=conv.padding,
                         dilation=conv.dilation,
                         groups=conv.groups,
                         bias=True,
                         padding_mode=conv.padding_mode)

        conv.weight = torch.nn.Parameter(weights)
        conv.bias = torch.nn.Parameter(bias)
        return conv

    def fuse_repvgg_block(self):
        if self.deploy:
            return
        print(f"RepConv.fuse_repvgg_block")

        self.rbr_dense = self.fuse_conv_bn(self.rbr_dense[0], self.rbr_dense[1])

        self.rbr_1x1 = self.fuse_conv_bn(self.rbr_1x1[0], self.rbr_1x1[1])
        rbr_1x1_bias = self.rbr_1x1.bias
        weight_1x1_expanded = torch.nn.functional.pad(self.rbr_1x1.weight, [2, 2, 2, 2])

        # Fuse self.rbr_identity
        if (isinstance(self.rbr_identity, nn.BatchNorm2d) or isinstance(self.rbr_identity,
                                                                        nn.modules.batchnorm.SyncBatchNorm)):
            # print(f"fuse: rbr_identity == BatchNorm2d or SyncBatchNorm")
            identity_conv_1x1 = nn.Conv2d(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=self.out_channels,
                bias=False)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.to(self.rbr_1x1.weight.data.device)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.squeeze().squeeze()
            # print(f" identity_conv_1x1.weight = {identity_conv_1x1.weight.shape}")
            identity_conv_1x1.weight.data.fill_(0.0)
            identity_conv_1x1.weight.data.fill_diagonal_(1.0)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.unsqueeze(2).unsqueeze(3)
            # print(f" identity_conv_1x1.weight = {identity_conv_1x1.weight.shape}")

            identity_conv_1x1 = self.fuse_conv_bn(identity_conv_1x1, self.rbr_identity)
            bias_identity_expanded = identity_conv_1x1.bias
            weight_identity_expanded = torch.nn.functional.pad(identity_conv_1x1.weight, [2, 2, 2, 2])
        else:
            # print(f"fuse: rbr_identity != BatchNorm2d, rbr_identity = {self.rbr_identity}")
            bias_identity_expanded = torch.nn.Parameter(torch.zeros_like(rbr_1x1_bias))
            weight_identity_expanded = torch.nn.Parameter(torch.zeros_like(weight_1x1_expanded))

            # print(f"self.rbr_1x1.weight = {self.rbr_1x1.weight.shape}, ")
        # print(f"weight_1x1_expanded = {weight_1x1_expanded.shape}, ")
        # print(f"self.rbr_dense.weight = {self.rbr_dense.weight.shape}, ")

        self.rbr_dense.weight = torch.nn.Parameter(
            self.rbr_dense.weight + weight_identity_expanded)
        self.rbr_dense.bias = torch.nn.Parameter(self.rbr_dense.bias + bias_identity_expanded)

        self.rbr_reparam = self.rbr_dense
        self.deploy = True

        if self.rbr_identity is not None:
            del self.rbr_identity
            self.rbr_identity = None

        if self.rbr_1x1 is not None:
            del self.rbr_1x1
            self.rbr_1x1 = None

        if self.rbr_dense is not None:
            del self.rbr_dense
            self.rbr_dense = None


class RepGhostBottleneck(nn.Module):
    """RepGhost bottleneck w/ optional SE"""

    def __init__(
            self,
            in_chs,
            mid_chs,
            out_chs,
            dw_kernel_size=3,
            stride=1,
            se_ratio=0.0,
            width=0.58,
            shortcut=True,
    ):
        super(RepGhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.0

        self.stride = stride
        self.enable_shortcut = shortcut
        self.in_chs = in_chs
        self.out_chs = out_chs

        # Point-wise expansion
        self.ghost1 = RepGhostConv(
            in_chs,
            mid_chs,
            k=3,
            s=1
        )

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(
                mid_chs,
                mid_chs,
                dw_kernel_size,
                stride=stride,
                padding=(dw_kernel_size - 1) // 2,
                groups=mid_chs,
                bias=False,
            )
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = RepGhostConv(
            mid_chs,
            out_chs,
            k=3,
            s=1
        )

        # shortcut
        if in_chs == out_chs and self.stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_chs,
                    in_chs,
                    dw_kernel_size,
                    stride=stride,
                    padding=(dw_kernel_size - 1) // 2,
                    groups=in_chs,
                    bias=False,
                ),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(
                    in_chs, out_chs, 1, stride=1,
                    padding=0, bias=False,
                ),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        residual = x

        # 1st repghost bottleneck
        x1 = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x1)
            x = self.bn_dw(x)
        else:
            x = x1

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd repghost bottleneck
        x = self.ghost2(x)
        if not self.enable_shortcut and self.in_chs == self.out_chs and self.stride == 1:
            return x
        return x + self.shortcut(residual)


# class RepGhostBottleneck(nn.Module):
#     # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
#     def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
#         super().__init__()
#         c_ = c2 // 2
#         self.conv = nn.Sequential(
#             RepGhostConv(c1, c_, 1, 1),  # pw
#             DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
#             RepGhostConv(c_, c2, 1, 1, act=False))  # pw-linear
#         self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
#                                       Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
#
#     def forward(self, x):
#         return self.conv(x) + self.shortcut(x)

class C3RepGhost(C3):
    # C3 module with GhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepGhostBottleneck(c_, c_) for _ in range(n)))


class AAM(nn.Module):
    def __init__(self, c1, c2=256, b1=0.2, b2=0.5, b3=0.8):
        super().__init__()
        self.beta1 = b1
        self.beta2 = b2
        self.beta3 = b3

        self.up1 = nn.Upsample(scale_factor=1 / b1, mode="bilinear")
        self.up2 = nn.Upsample(scale_factor=1 / b2, mode="bilinear")
        self.up3 = nn.Upsample(scale_factor=1 / b3, mode="bilinear")

        self.cv1 = nn.Conv2d(c1, 256, 1, 1, bias=False)
        self.cv2 = nn.Conv2d(c1, 256, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c1, 256, 1, 1, bias=False)
        # self.cv4 = nn.Conv2d(c1, 256, 1, 1, bias=False)

        self.conv = nn.Sequential(
            nn.Conv2d(256 * 3, 256 * 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256 * 3, 256 * 3, 3, 1, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        h, w = x.shape[-2], x.shape[-1]
        x1, x2, x3 = F.adaptive_avg_pool2d(x, (int(h * self.beta1), int(w * self.beta1))), \
            F.adaptive_avg_pool2d(x, (int(h * self.beta2), int(w * self.beta2))), \
            F.adaptive_avg_pool2d(x, (int(h * self.beta3), int(w * self.beta3)))
        x1, x2, x3 = self.cv1(x1), self.cv2(x2), self.cv3(x3)

        x1, x2, x3 = self.up1(x1), self.up2(x2), self.up3(x3)
        x4 = torch.cat((x1, x2, x3), dim=1)
        x4 = self.conv(x4) * x4
        x1, x2, x3 = x4.split(256, dim=1)
        x = x1 + x2 + x3 + x
        return x


class FEM(nn.Module):
    def __init__(self, c1, c2=256):
        super().__init__()
        self.dia_cv1 = nn.Sequential(
            nn.Conv2d(c1, c1, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )

        self.dia_cv2 = nn.Sequential(
            nn.Conv2d(c1, c1, 3, 1, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )
        self.dia_cv3 = nn.Sequential(
            nn.Conv2d(c1, c1, 3, 1, padding=5, dilation=5, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1, x2, x3 = self.dia_cv1(x), self.dia_cv2(x), self.dia_cv3(x)
        out = (x1 + x2 + x3) / 3.
        return out


class ADD(nn.Module):
    def __init__(self, c1=512, c2=512):
        super().__init__()
        pass

    def forward(self, x):
        return x[0] + x[1]


#  动态卷积-ODConv
class ODConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = ODConv2d(c1, c2, k, s, autopad(k, p), groups=g, kernel_num=4)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class ODConv_Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = ODConv(c_, c2, 3, 1, g=g, act=False)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3_ODConv(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(ODConv_Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


#  可形变卷积
class DCNv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=1, dilation=1, groups=1, deformable_groups=1):
        super(DCNv2, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.dilation = (dilation, dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *self.kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))

        out_channels_offset_mask = (self.deformable_groups * 3 *
                                    self.kernel_size[0] * self.kernel_size[1])
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels,
            out_channels_offset_mask,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
        self.reset_parameters()

    def forward(self, x):
        offset_mask = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        x = torch.ops.torchvision.deform_conv2d(
            x,
            self.weight,
            offset,
            mask,
            self.bias,
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1],
            self.groups,
            self.deformable_groups,
            True
        )
        x = self.bn(x)
        x = self.act(x)
        return x

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        std = 1. / math.sqrt(n)
        self.weight.data.uniform_(-std, std)
        self.bias.data.zero_()
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()


from models.DeformConv2d import DeformConv2d


class Bottleneck_DCN(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 3, 1, 1)
        self.cv2 = DeformConv2d(c_, c_, 3, padding=1, modulation=True)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3_DCN(C3):
    # C3 module with DCNv2
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck_DCN(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


#   GSConv + VoVCSP
class GSConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        # shuffle
        b, n, h, w = x2.data.size()
        b_n = b * n // 2
        y = x2.reshape(b_n, 2, h * w)
        y = y.permute(1, 0, 2)
        y = y.reshape(2, -1, n // 2, h, w)

        return torch.cat((y[0], y[1]), 1)


class GSBottleneck(nn.Module):
    def __init__(self, c1, c2, k=3, s=1):
        super(GSBottleneck, self).__init__()
        c_ = c2 // 2
        # for lighting
        self.conv_lighting = nn.Sequential(
            GSConv(c1, c_, 1, 1),
            GSConv(c_, c2, 1, 1, act=False))
        # for receptive field
        self.conv = nn.Sequential(
            GSConv(c1, c_, 3, 1),
            GSConv(c_, c2, 3, 1, act=False))
        self.shortcut = Conv(c1, c2, 3, 1, act=False)

    def forward(self, x):
        return self.conv_lighting(x) + self.shortcut(x)


class VoVGSCSP(nn.Module):
    # VoV-GSCSP https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(GSBottleneck(c_, c_) for _ in range(n)))

    def forward(self, x):
        x1 = self.cv1(x)
        return self.cv2(torch.cat((self.m(x1), x1), dim=1))


# 空洞卷积
class DilateConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1):
        super().__init__()
        c_ = c1 // 2
        self.cv3x3 = Conv(c1, c_, 3, 1)
        self.cv1x1_1 = Conv(c1, c_, 1, 1)
        self.cv1x1_2 = Conv(c_, c_, 1, 1)
        self.cv1x1_3 = Conv(c1, c1, 1, 1)
        self.cv3x3_2 = nn.Sequential(
            nn.Conv2d(c_, c_, 3, 1, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU(inplace=True)
        )
        self.cv3x3_3 = nn.Sequential(
            nn.Conv2d(c_, c_, 3, 1, padding=8, dilation=8, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.cv3x3(x)
        x2 = self.cv1x1_1(x)
        x3_1 = self.cv3x3_2(x2)
        x3_2 = self.cv3x3_3(x2)
        x3 = self.cv1x1_2(x3_1 + x3_2)
        out = self.cv1x1_3(torch.cat((x1, x3), 1))

        return out

# 基于归一化的注意力模块：Normalization-based Attention Module
class NAMAtt(nn.Module):
    def __init__(self, channels, t=16):
        super().__init__()
        self.channels = channels
        self.bn2 = nn.BatchNorm2d(self.channels, affine=True)

    def forward(self, x):
        residual = x
        x = self.bn2(x)
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = torch.sigmoid(x) * residual  #
        return x


class ConvTri(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.att = NAMAtt(c2)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.att(self.conv(x))))


class BottleneckTri(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
        self.att = NAMAtt(c2)

    def forward(self, x):
        return self.att(x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x)))


class C3_Tri(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(BottleneckTri(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


#  ConvNeXt
class LayerNorm_s(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


# ConvNextBlock
class ConvNextBlock(nn.Module):

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm_s(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class CNeB(nn.Module):
    # CSP ConvNextBlock with 3 convolutions by iscyy/yoloair
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(ConvNextBlock(c_) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


#  ACmix
import torch.nn.functional as F
import time


def position(H, W, is_cuda=True):
    if is_cuda:
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)
    else:
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc


def stride(x, stride):
    b, c, h, w = x.shape
    return x[:, :, ::stride, ::stride]


def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)


def init_rate_0(tensor):
    if tensor is not None:
        tensor.data.fill_(0.)


class ACmix(nn.Module):
    def __init__(self, in_planes, kernel_att=7, head=4, kernel_conv=3, stride=1, dilation=1):
        super(ACmix, self).__init__()
        self.in_planes = in_planes
        self.out_planes = in_planes
        self.head = head
        self.kernel_att = kernel_att
        self.kernel_conv = kernel_conv
        self.stride = stride
        self.dilation = dilation
        self.rate1 = torch.nn.Parameter(torch.Tensor(1))
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))
        self.head_dim = self.out_planes // self.head

        self.conv1 = nn.Conv2d(in_planes, self.out_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, self.out_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, self.out_planes, kernel_size=1)
        self.conv_p = nn.Conv2d(2, self.head_dim, kernel_size=1)

        self.padding_att = (self.dilation * (self.kernel_att - 1) + 1) // 2
        self.pad_att = torch.nn.ReflectionPad2d(self.padding_att)
        self.unfold = nn.Unfold(kernel_size=self.kernel_att, padding=0, stride=self.stride)
        self.softmax = torch.nn.Softmax(dim=1)

        self.fc = nn.Conv2d(3 * self.head, self.kernel_conv * self.kernel_conv, kernel_size=1, bias=False)
        self.dep_conv = nn.Conv2d(self.kernel_conv * self.kernel_conv * self.head_dim, self.out_planes,
                                  kernel_size=self.kernel_conv, bias=True, groups=self.head_dim, padding=1,
                                  stride=stride)

        self.reset_parameters()

    def reset_parameters(self):
        init_rate_half(self.rate1)
        init_rate_half(self.rate2)
        kernel = torch.zeros(self.kernel_conv * self.kernel_conv, self.kernel_conv, self.kernel_conv)
        for i in range(self.kernel_conv * self.kernel_conv):
            kernel[i, i // self.kernel_conv, i % self.kernel_conv] = 1.
        kernel = kernel.squeeze(0).repeat(self.out_planes, 1, 1, 1)
        self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=True)
        self.dep_conv.bias = init_rate_0(self.dep_conv.bias)

    def forward(self, x):
        q, k, v = self.conv1(x), self.conv2(x), self.conv3(x)
        scaling = float(self.head_dim) ** -0.5
        b, c, h, w = q.shape
        h_out, w_out = h // self.stride, w // self.stride

        # ### att
        # ## positional encoding https://github.com/iscyy/yoloair
        mid = position(h, w, x.is_cuda)
        mid = mid.type_as(x)
        pe = self.conv_p(mid)

        q_att = q.view(b * self.head, self.head_dim, h, w) * scaling
        k_att = k.view(b * self.head, self.head_dim, h, w)
        v_att = v.view(b * self.head, self.head_dim, h, w)

        if self.stride > 1:
            q_att = stride(q_att, self.stride)
            q_pe = stride(pe, self.stride)
        else:
            q_pe = pe

        unfold_k = self.unfold(self.pad_att(k_att)).view(b * self.head, self.head_dim,
                                                         self.kernel_att * self.kernel_att, h_out,
                                                         w_out)  # b*head, head_dim, k_att^2, h_out, w_out
        unfold_rpe = self.unfold(self.pad_att(pe)).view(1, self.head_dim, self.kernel_att * self.kernel_att, h_out,
                                                        w_out)  # 1, head_dim, k_att^2, h_out, w_out

        att = (q_att.unsqueeze(2) * (unfold_k + q_pe.unsqueeze(2) - unfold_rpe)).sum(
            1)  # (b*head, head_dim, 1, h_out, w_out) * (b*head, head_dim, k_att^2, h_out, w_out) -> (b*head, k_att^2, h_out, w_out)
        att = self.softmax(att)

        out_att = self.unfold(self.pad_att(v_att)).view(b * self.head, self.head_dim, self.kernel_att * self.kernel_att,
                                                        h_out, w_out)
        out_att = (att.unsqueeze(1) * out_att).sum(2).view(b, self.out_planes, h_out, w_out)

        ## conv
        f_all = self.fc(torch.cat(
            [q.view(b, self.head, self.head_dim, h * w), k.view(b, self.head, self.head_dim, h * w),
             v.view(b, self.head, self.head_dim, h * w)], 1))
        f_conv = f_all.permute(0, 2, 1, 3).reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])

        out_conv = self.dep_conv(f_conv)

        return self.rate1 * out_att + self.rate2 * out_conv


#  ASFF
class ASFFV5(nn.Module):
    def __init__(self, level, multiplier=1, rfb=False, vis=False, act_cfg=True):
        """
        ASFF version for YoloV5 .
        different than YoloV3
        multiplier should be 1, 0.5
        which means, the channel of ASFF can be
        512, 256, 128 -> multiplier=1
        256, 128, 64 -> multiplier=0.5
        For even smaller, you need change code manually.
        """
        super(ASFFV5, self).__init__()
        self.level = level
        self.dim = [int(1024 * multiplier), int(512 * multiplier),
                    int(256 * multiplier)]
        # print(self.dim)

        self.inter_dim = self.dim[self.level]
        if level == 0:
            self.stride_level_1 = Conv(int(512 * multiplier), self.inter_dim, 3, 2)

            self.stride_level_2 = Conv(int(256 * multiplier), self.inter_dim, 3, 2)

            self.expand = Conv(self.inter_dim, int(
                1024 * multiplier), 3, 1)
        elif level == 1:
            self.compress_level_0 = Conv(
                int(1024 * multiplier), self.inter_dim, 1, 1)
            self.stride_level_2 = Conv(
                int(256 * multiplier), self.inter_dim, 3, 2)
            self.expand = Conv(self.inter_dim, int(512 * multiplier), 3, 1)
        elif level == 2:
            self.compress_level_0 = Conv(
                int(1024 * multiplier), self.inter_dim, 1, 1)
            self.compress_level_1 = Conv(
                int(512 * multiplier), self.inter_dim, 1, 1)
            self.expand = Conv(self.inter_dim, int(
                256 * multiplier), 3, 1)

        # when adding rfb, we use half number of channels to save memory
        compress_c = 8 if rfb else 16
        self.weight_level_0 = Conv(
            self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = Conv(
            self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(
            self.inter_dim, compress_c, 1, 1)

        self.weight_levels = Conv(
            compress_c * 3, 3, 1, 1)
        self.vis = vis

    def forward(self, x):  # l,m,s
        """
        # 128, 256, 512
        512, 256, 128
        from small -> large
        """
        x_level_0 = x[2]  # l
        x_level_1 = x[1]  # m
        x_level_2 = x[0]  # s
        # print('x_level_0: ', x_level_0.shape)
        # print('x_level_1: ', x_level_1.shape)
        # print('x_level_2: ', x_level_2.shape)
        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_downsampled_inter = F.max_pool2d(
                x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)
        elif self.level == 1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(
                level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
        elif self.level == 2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(
                level_0_compressed, scale_factor=4, mode='nearest')
            x_level_1_compressed = self.compress_level_1(x_level_1)
            level_1_resized = F.interpolate(
                x_level_1_compressed, scale_factor=2, mode='nearest')
            level_2_resized = x_level_2

        # print('level: {}, l1_resized: {}, l2_resized: {}'.format(self.level,
        #      level_1_resized.shape, level_2_resized.shape))
        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        # print('level_0_weight_v: ', level_0_weight_v.shape)
        # print('level_1_weight_v: ', level_1_weight_v.shape)
        # print('level_2_weight_v: ', level_2_weight_v.shape)

        levels_weight_v = torch.cat(
            (level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
                            level_1_resized * levels_weight[:, 1:2, :, :] + \
                            level_2_resized * levels_weight[:, 2:, :, :]

        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out


#  efficientNet
from torchvision import models


class Efficient1(nn.Module):
    def __init__(self, ignore) -> None:
        super().__init__()
        model = models.efficientnet_b2()
        modules = list(model.children())
        modules = modules[0][:4]
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class Efficient2(nn.Module):
    def __init__(self, ignore) -> None:
        super().__init__()
        model = models.efficientnet_b2()
        modules = list(model.children())
        modules = modules[0][4:6]
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class Efficient3(nn.Module):
    def __init__(self, ignore) -> None:
        super().__init__()
        model = models.efficientnet_b2()
        modules = list(model.children())
        modules = modules[0][6:]
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = heads
        self.dim = dim
        self.fc = nn.Linear(dim, dim)
        self.scale = torch.sqrt(torch.FloatTensor([dim // heads]))
        self.do = nn.Dropout(0.1)

    def forward(self, query, key, value):
        self.scale = self.scale.to(query.device)
        bsz = query.shape[0]
        Q = query.view(bsz, -1, self.n_heads, self.dim // self.n_heads).permute(0, 2, 1, 3)
        K = key.view(bsz, -1, self.n_heads, self.dim // self.n_heads).permute(0, 2, 1, 3)
        V = value.view(bsz, -1, self.n_heads, self.dim // self.n_heads).permute(0, 2, 1, 3)

        # 第 1 步：Q 乘以 K的转置，除以scale
        # [64,6,12,50] * [64,6,50,10] = [64,6,12,10]
        # attention：[64,6,12,10]
        attention = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale.type_as(V)

        # 第 2 步：计算上一步结果的 softmax，再经过 dropout，得到 attention。
        # 注意，这里是对最后一维做 softmax，也就是在输入序列的维度做 softmax
        # attention: [64,6,12,10]

        attention = self.do(torch.softmax(attention, dim=-1))

        # 第三步，attention结果与V相乘，得到多头注意力的结果
        # [64,6,12,10] * [64,6,10,50] = [64,6,12,50]
        # x: [64,6,12,50]
        x = torch.matmul(attention, V)

        # 因为 query 有 12 个词，所以把 12 放到前面，把 5 和 60 放到后面，方便下面拼接多组的结果
        # x: [64,6,12,50] 转置-> [64,12,6,50]
        x = x.permute(0, 2, 1, 3).contiguous()
        # 这里的矩阵转换就是：把多组注意力的结果拼接起来
        # 最终结果就是 [64,12,300]
        # x: [64,12,6,50] -> [64,12,300]
        x = x.view(bsz, -1, self.n_heads * (self.dim // self.n_heads))
        x = self.fc(x)
        return x


class TransformerLayer(nn.Module):
    def __init__(self, c, num_heads):
        super().__init__()

        self.ln1 = nn.LayerNorm(c)
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = MultiHeadAttention(dim=c, heads=num_heads)  # nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)#
        self.ln2 = nn.LayerNorm(c)
        self.fc1 = nn.Linear(c, 4 * c, bias=False)
        self.fc2 = nn.Linear(4 * c, c, bias=False)
        self.dropout = nn.Dropout(0.1)
        self.act = nn.ReLU(True)

    def forward(self, x):
        x_ = self.ln1(x)
        x = self.dropout(self.ma(self.q(x_), self.k(x_), self.v(x_))) + x
        x_ = self.ln2(x)
        x_ = self.fc2(self.dropout(self.act(self.fc1(x_))))
        x = x + self.dropout(x_)
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).unsqueeze(0).transpose(0, 3).squeeze(3)
        return self.tr(p + self.linear(p)).unsqueeze(3).transpose(0, 3).reshape(b, self.c2, w, h)


class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


# ----------------------------------#
#   yolov4
# ----------------------------------#

# 主干网络CSP
class BottleneckCSP_v4(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP_v4, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 1, 1)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.bn(torch.cat((y1, y2), dim=1)))


# ------------------ 专利用注意力机制 -------------#
class CAM(nn.Module):
    def __init__(self, in_channels):
        super(CAM, self).__init__()
        self.dw = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1,
                            groups=in_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.C_MLP = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=int(in_channels // 2), kernel_size=1, stride=1),
            nn.GELU(),
            nn.Conv2d(in_channels=int(in_channels // 2), out_channels=in_channels, kernel_size=1, stride=1),
        )

    def forward(self, x):
        res = x
        x = torch.sigmoid(self.C_MLP(self.pool(self.dw(x))))

        return x * res


class SAM(nn.Module):
    def __init__(self, in_channels, k=[5, 7, 13]):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, stride=1)
        self.pool = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=int(x // 2)) for x in k])
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1, stride=1)

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.conv2(torch.cat([x] + [m(x) for m in self.pool], dim=1))
        return res * torch.sigmoid(x)


class PAM(nn.Module):
    def __init__(self, in_channels):
        super(PAM, self).__init__()
        self.cam = CAM(in_channels=in_channels)
        self.sam = SAM(in_channels=in_channels)

    def forward(self, x):
        return x + self.cam(x) + self.sam(x)


class C3_MLP_PAM(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(C3_MLP_PAM, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.mlp = LightMLPBlock(c_, c_)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        self.pam = PAM(c2)

    def forward(self, x):
        return self.cv3(self.pam(torch.cat((self.m(self.cv1(x)), self.mlp(self.cv2(x))), dim=1)))


# ------------------- RFB ------------------#
class BasicRFB(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, scale=0.5):
        super().__init__()
        self.scale = scale
        self.out_channels = out_channels
        inter_channels = in_channels // 4

        self.branch0 = nn.Sequential(
            Conv(in_channels, inter_channels, k=1, s=1),
            Conv(inter_channels, inter_channels, k=3, s=1, act=False)
        )
        self.branch1 = nn.Sequential(
            Conv(in_channels, inter_channels, k=1, s=1),
            Conv(inter_channels, inter_channels, k=(3, 1), s=1, p=(1, 0)),
            Conv(inter_channels, inter_channels, k=3, s=1, p=3, dilation=3, act=False)
        )
        self.branch2 = nn.Sequential(
            Conv(in_channels, inter_channels, k=1, s=1),
            Conv(inter_channels, inter_channels, k=(1, 3), s=stride, p=(0, 1)),
            Conv(inter_channels, inter_channels, k=3, s=1, p=3, dilation=3, act=False)
        )
        self.branch3 = nn.Sequential(
            Conv(in_channels, inter_channels // 2, k=1, s=1),
            Conv(inter_channels // 2, (inter_channels // 4) * 3, k=(1, 3), s=1, p=(0, 1)),
            Conv((inter_channels // 4) * 3, inter_channels, k=(3, 1), s=stride, p=(1, 0)),
            Conv(inter_channels, inter_channels, k=3, s=1, p=5, dilation=5, act=False)
        )

        self.ConvLinear = Conv(4 * inter_channels, out_channels, k=1, s=1, act=False)
        self.shortcut = Conv(in_channels, out_channels, k=1, s=stride, act=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        out = torch.cat((x0, x1, x2, x3), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)

        return out


class CRFB(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(BasicRFB(c_, c_, 1, scale=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


from torch.autograd import Function


class Covpool(Function):
    @staticmethod
    def forward(ctx, input):
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        h = x.data.shape[2]
        w = x.data.shape[3]
        M = h * w
        x = x.reshape(batchSize, dim, M)
        I_hat = (-1. / M / M) * torch.ones(M, M, device=x.device) + (1. / M) * torch.eye(M, M, device=x.device)
        I_hat = I_hat.view(1, M, M).repeat(batchSize, 1, 1).type(x.dtype)
        y = x.bmm(I_hat).bmm(x.transpose(1, 2))
        ctx.save_for_backward(input, I_hat)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        input, I_hat = ctx.saved_tensors
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        h = x.data.shape[2]
        w = x.data.shape[3]
        M = h * w
        x = x.reshape(batchSize, dim, M)
        grad_input = grad_output + grad_output.transpose(1, 2)
        grad_input = grad_input.bmm(x).bmm(I_hat)
        grad_input = grad_input.reshape(batchSize, dim, h, w)
        return grad_input


class Sqrtm(Function):
    @staticmethod
    def forward(ctx, input, iterN):
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        dtype = x.dtype
        I3 = 3.0 * torch.eye(dim, dim, device=x.device).view(1, dim, dim).repeat(batchSize, 1, 1).type(dtype)
        normA = (1.0 / 3.0) * x.mul(I3).sum(dim=1).sum(dim=1)
        A = x.div(normA.view(batchSize, 1, 1).expand_as(x))
        Y = torch.zeros(batchSize, iterN, dim, dim, requires_grad=False, device=x.device)
        Z = torch.eye(dim, dim, device=x.device).view(1, dim, dim).repeat(batchSize, iterN, 1, 1)
        if iterN < 2:
            ZY = 0.5 * (I3 - A)
            Y[:, 0, :, :] = A.bmm(ZY)
        else:
            ZY = 0.5 * (I3 - A)
            Y[:, 0, :, :] = A.bmm(ZY)
            Z[:, 0, :, :] = ZY
            for i in range(1, iterN - 1):
                ZY = 0.5 * (I3 - Z[:, i - 1, :, :].bmm(Y[:, i - 1, :, :]))
                Y[:, i, :, :] = Y[:, i - 1, :, :].bmm(ZY)
                Z[:, i, :, :] = ZY.bmm(Z[:, i - 1, :, :])
            ZY = 0.5 * Y[:, iterN - 2, :, :].bmm(I3 - Z[:, iterN - 2, :, :].bmm(Y[:, iterN - 2, :, :]))
        y = ZY * torch.sqrt(normA).view(batchSize, 1, 1).expand_as(x)
        ctx.save_for_backward(input, A, ZY, normA, Y, Z)
        ctx.iterN = iterN
        return y

    @staticmethod
    def backward(ctx, grad_output):
        input, A, ZY, normA, Y, Z = ctx.saved_tensors
        iterN = ctx.iterN
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        dtype = x.dtype
        der_postCom = grad_output * torch.sqrt(normA).view(batchSize, 1, 1).expand_as(x)
        der_postComAux = (grad_output * ZY).sum(dim=1).sum(dim=1).div(2 * torch.sqrt(normA))
        I3 = 3.0 * torch.eye(dim, dim, device=x.device).view(1, dim, dim).repeat(batchSize, 1, 1).type(dtype)
        if iterN < 2:
            der_NSiter = 0.5 * (der_postCom.bmm(I3 - A) - A.bmm(der_sacleTrace))
        else:
            dldY = 0.5 * (der_postCom.bmm(I3 - Y[:, iterN - 2, :, :].bmm(Z[:, iterN - 2, :, :])) -
                          Z[:, iterN - 2, :, :].bmm(Y[:, iterN - 2, :, :]).bmm(der_postCom))
            dldZ = -0.5 * Y[:, iterN - 2, :, :].bmm(der_postCom).bmm(Y[:, iterN - 2, :, :])
            for i in range(iterN - 3, -1, -1):
                YZ = I3 - Y[:, i, :, :].bmm(Z[:, i, :, :])
                ZY = Z[:, i, :, :].bmm(Y[:, i, :, :])
                dldY_ = 0.5 * (dldY.bmm(YZ) -
                               Z[:, i, :, :].bmm(dldZ).bmm(Z[:, i, :, :]) -
                               ZY.bmm(dldY))
                dldZ_ = 0.5 * (YZ.bmm(dldZ) -
                               Y[:, i, :, :].bmm(dldY).bmm(Y[:, i, :, :]) -
                               dldZ.bmm(ZY))
                dldY = dldY_
                dldZ = dldZ_
            der_NSiter = 0.5 * (dldY.bmm(I3 - A) - dldZ - A.bmm(dldY))
        grad_input = der_NSiter.div(normA.view(batchSize, 1, 1).expand_as(x))
        grad_aux = der_NSiter.mul(x).sum(dim=1).sum(dim=1)
        for i in range(batchSize):
            grad_input[i, :, :] += (der_postComAux[i] \
                                    - grad_aux[i] / (normA[i] * normA[i])) \
                                   * torch.ones(dim, device=x.device).diag()
        return grad_input, None


def CovpoolLayer(var):
    return Covpool.apply(var)


def SqrtmLayer(var, iterN):
    return Sqrtm.apply(var, iterN)


class SOCA(nn.Module):
    # second-order Channel attention
    def __init__(self, channel, reduction=8):
        super(SOCA, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2)

        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, C, h, w = x.shape  # x: NxCxHxW
        N = int(h * w)
        min_h = min(h, w)
        h1 = 1000
        w1 = 1000
        if h < h1 and w < w1:
            x_sub = x
        elif h < h1 and w > w1:
            W = (w - w1) // 2
            x_sub = x[:, :, :, W:(W + w1)]
        elif w < w1 and h > h1:
            H = (h - h1) // 2
            x_sub = x[:, :, H:H + h1, :]
        else:
            H = (h - h1) // 2
            W = (w - w1) // 2
            x_sub = x[:, :, H:(H + h1), W:(W + w1)]
        cov_mat = CovpoolLayer(x_sub)  # Global Covariance pooling layer
        cov_mat_sqrt = SqrtmLayer(cov_mat,
                                  5)  # Matrix square root layer( including pre-norm,Newton-Schulz iter. and post-com. with 5 iteration)
        cov_mat_sum = torch.mean(cov_mat_sqrt, 1)
        cov_mat_sum = cov_mat_sum.view(batch_size, C, 1, 1)
        cov_mat_sum = cov_mat_sum.to(x.dtype)
        y_cov = self.conv_du(cov_mat_sum)
        return y_cov * x


class C3_BiFormer_Block(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(C3_BiFormer_Block, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.BiF = BiFormerBlock(c_)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.BiF(self.cv2(x))), dim=1))


class C3BiFormer(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(C3BiFormer, self).__init__()
        self.m = nn.Sequential(*(BiFormerBlock(c1) for _ in range(n)))
        self.conv = Conv(c1, c2, 1)

    def forward(self, x):
        return self.conv(self.m(x))


class C3BiFormer_1(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(C3BiFormer_1, self).__init__()
        self.m = nn.Sequential(*(BiFormerBlock(c2) for _ in range(n)))
        self.conv = Conv(c1, c2, 1)

    def forward(self, x):
        return self.m(self.conv(x))


class asign_index(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kernel, guide_feature):
        ctx.save_for_backward(kernel, guide_feature)
        guide_mask = torch.zeros_like(guide_feature).scatter_(1, guide_feature.argmax(dim=1, keepdim=True),
                                                              1).unsqueeze(2)  # B x 3 x 1 x 25 x 25
        return torch.sum(kernel * guide_mask, dim=1)

    @staticmethod
    def backward(ctx, grad_output):
        kernel, guide_feature = ctx.saved_tensors
        guide_mask = torch.zeros_like(guide_feature).scatter_(1, guide_feature.argmax(dim=1, keepdim=True),
                                                              1).unsqueeze(2)  # B x 3 x 1 x 25 x 25
        grad_kernel = grad_output.clone().unsqueeze(1) * guide_mask  # B x 3 x 256 x 25 x 25
        grad_guide = grad_output.clone().unsqueeze(1) * kernel  # B x 3 x 256 x 25 x 25
        grad_guide = grad_guide.sum(dim=2)  # B x 3 x 25 x 25
        softmax = F.softmax(guide_feature, 1)  # B x 3 x 25 x 25
        grad_guide = softmax * (grad_guide - (softmax * grad_guide).sum(dim=1, keepdim=True))  # B x 3 x 25 x 25
        return grad_kernel, grad_guide


def xcorr_slow(x, kernel, kwargs):
    """for loop to calculate cross correlation
    """
    batch = x.size()[0]
    out = []
    for i in range(batch):
        px = x[i]
        pk = kernel[i]
        px = px.view(1, px.size()[0], px.size()[1], px.size()[2])
        pk = pk.view(-1, px.size()[1], pk.size()[1], pk.size()[2])
        po = F.conv2d(px, pk, padding=1, **kwargs)
        out.append(po)
    out = torch.cat(out, 0)
    return out


def xcorr_fast(x, kernel, kwargs):
    """group conv2d to calculate cross correlation
    """
    batch = kernel.size()[0]
    pk = kernel.view(-1, x.size()[1], kernel.size()[2], kernel.size()[3])
    px = x.view(1, -1, x.size()[2], x.size()[3])
    po = F.conv2d(px, pk, padding=1, **kwargs, groups=batch)
    po = po.view(batch, -1, po.size()[2], po.size()[3])
    return po


class Corr(Function):
    @staticmethod
    def symbolic(g, x, kernel, groups):
        return g.op("Corr", x, kernel, groups_i=groups)

    @staticmethod
    def forward(self, x, kernel, groups, kwargs):
        """group conv2d to calculate cross correlation
        """
        batch = x.size(0)
        channel = x.size(1)
        x = x.view(1, -1, x.size(2), x.size(3))
        kernel = kernel.view(-1, channel // groups, kernel.size(2), kernel.size(3))
        out = F.conv2d(x, kernel, padding=1, **kwargs, groups=groups * batch)
        out = out.view(batch, -1, out.size(2), out.size(3))
        return out


class Correlation(nn.Module):
    use_slow = True

    def __init__(self, use_slow=None):
        super(Correlation, self).__init__()
        if use_slow is not None:
            self.use_slow = use_slow
        else:
            self.use_slow = Correlation.use_slow

    def extra_repr(self):
        if self.use_slow: return "xcorr_slow"
        return "xcorr_fast"

    def forward(self, x, kernel, **kwargs):
        if self.training:
            if self.use_slow:
                return xcorr_slow(x, kernel, kwargs)
            else:
                return xcorr_fast(x, kernel, kwargs)
        else:
            return Corr.apply(x, kernel, 1, kwargs)


class DRConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, region_num=8, **kwargs):
        super(DRConv2d, self).__init__()
        self.region_num = region_num

        self.conv_kernel = nn.Sequential(
            nn.AdaptiveAvgPool2d((kernel_size, kernel_size)),
            nn.Conv2d(in_channels, region_num * region_num, kernel_size=1),
            nn.Sigmoid(),
            nn.Conv2d(region_num * region_num, region_num * in_channels * out_channels, kernel_size=1,
                      groups=region_num)
        )
        self.conv_guide = nn.Conv2d(in_channels, region_num, kernel_size=kernel_size, padding=padding, **kwargs)

        self.corr = Correlation(use_slow=False)
        self.kwargs = kwargs
        self.asign_index = asign_index.apply

    def forward(self, input):
        kernel = self.conv_kernel(input)
        kernel = kernel.view(kernel.size(0), -1, kernel.size(2), kernel.size(3))  # B x (r*in*out) x W X H
        output = self.corr(input, kernel, **self.kwargs)  # B x (r*out) x W x H
        output = output.view(output.size(0), self.region_num, -1, output.size(2), output.size(3))  # B x r x out x W x H
        guide_feature = self.conv_guide(input)
        output = self.asign_index(output, guide_feature)
        return output


class DRConvBlock(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = DRConv2d(c_, c2, 3, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3_DRConv(C3):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(DRConvBlock(c_, c_, shortcut, g, 1.0) for _ in range(n)))


class DetectMultiBackend(nn.Module):
    # YOLOv5 MultiBackend class for python inference on various backends
    def __init__(self, weights='yolov5s.pt', device=torch.device('cpu'), dnn=False, data=None, fp16=False, fuse=True):
        # Usage:
        #   PyTorch:              weights = *.pt
        #   TorchScript:                    *.torchscript
        #   ONNX Runtime:                   *.onnx
        #   ONNX OpenCV DNN:                *.onnx with --dnn
        #   OpenVINO:                       *.xml
        #   CoreML:                         *.mlmodel
        #   TensorRT:                       *.engine
        #   TensorFlow SavedModel:          *_saved_model
        #   TensorFlow GraphDef:            *.pb
        #   TensorFlow Lite:                *.tflite
        #   TensorFlow Edge TPU:            *_edgetpu.tflite
        from models.experimental import attempt_download, attempt_load  # scoped to avoid circular import

        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs = self.model_type(w)  # get backend
        #
        w = attempt_download(w)  # download if not local (权重文件存在则返回权重文件名,不存在则下载权重文件)
        fp16 &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16
        stride, names = 32, [f'class{i}' for i in range(1000)]  # assign defaults
        if data:  # assign class names (optional)
            with open(data, errors='ignore') as f:
                names = yaml.safe_load(f)['names']

        if pt:  # PyTorch 基于pt model文件获取模型信息
            model = attempt_load(weights if isinstance(weights, list) else w, device=device, inplace=True, fuse=fuse)
            stride = max(int(model.stride.max()), 32)  # model stride
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            model.half() if fp16 else model.float()
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
        elif jit:  # TorchScript
            LOGGER.info(f'Loading {w} for TorchScript inference...')
            extra_files = {'config.txt': ''}  # model metadata
            model = torch.jit.load(w, _extra_files=extra_files)
            model.half() if fp16 else model.float()
            if extra_files['config.txt']:
                d = json.loads(extra_files['config.txt'])  # extra_files dict
                stride, names = int(d['stride']), d['names']
        elif dnn:  # ONNX OpenCV DNN
            LOGGER.info(f'Loading {w} for ONNX OpenCV DNN inference...')
            check_requirements(('opencv-python>=4.5.4',))
            net = cv2.dnn.readNetFromONNX(w)
        elif onnx:  # ONNX Runtime
            LOGGER.info(f'Loading {w} for ONNX Runtime inference...')
            cuda = torch.cuda.is_available()
            check_requirements(('onnx', 'onnxruntime-gpu' if cuda else 'onnxruntime'))
            import onnxruntime
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
            session = onnxruntime.InferenceSession(w, providers=providers)
            meta = session.get_modelmeta().custom_metadata_map  # metadata
            if 'stride' in meta:
                stride, names = int(meta['stride']), eval(meta['names'])
        elif xml:  # OpenVINO
            LOGGER.info(f'Loading {w} for OpenVINO inference...')
            check_requirements(('openvino',))  # requires openvino-dev: https://pypi.org/project/openvino-dev/
            from openvino.runtime import Core, Layout, get_batch
            ie = Core()
            if not Path(w).is_file():  # if not *.xml
                w = next(Path(w).glob('*.xml'))  # get *.xml file from *_openvino_model dir
            network = ie.read_model(model=w, weights=Path(w).with_suffix('.bin'))
            if network.get_parameters()[0].get_layout().empty:
                network.get_parameters()[0].set_layout(Layout("NCHW"))
            batch_dim = get_batch(network)
            if batch_dim.is_static:
                batch_size = batch_dim.get_length()
            executable_network = ie.compile_model(network, device_name="CPU")  # device_name="MYRIAD" for Intel NCS2
            output_layer = next(iter(executable_network.outputs))
            meta = Path(w).with_suffix('.yaml')
            if meta.exists():
                stride, names = self._load_metadata(meta)  # load metadata
        elif engine:  # TensorRT
            LOGGER.info(f'Loading {w} for TensorRT inference...')
            import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download
            check_version(trt.__version__, '7.0.0', hard=True)  # require tensorrt>=7.0.0
            Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
            logger = trt.Logger(trt.Logger.INFO)
            with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
                model = runtime.deserialize_cuda_engine(f.read())
            context = model.create_execution_context()
            bindings = OrderedDict()
            fp16 = False  # default updated below
            dynamic = False
            for index in range(model.num_bindings):
                name = model.get_binding_name(index)
                dtype = trt.nptype(model.get_binding_dtype(index))
                if model.binding_is_input(index):
                    if -1 in tuple(model.get_binding_shape(index)):  # dynamic
                        dynamic = True
                        context.set_binding_shape(index, tuple(model.get_profile_shape(0, index)[2]))
                    if dtype == np.float16:
                        fp16 = True
                shape = tuple(context.get_binding_shape(index))
                data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
                bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
            binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
            batch_size = bindings['images'].shape[0]  # if dynamic, this is instead max batch size
        elif coreml:  # CoreML
            LOGGER.info(f'Loading {w} for CoreML inference...')
            import coremltools as ct
            model = ct.models.MLModel(w)
        else:  # TensorFlow (SavedModel, GraphDef, Lite, Edge TPU)
            if saved_model:  # SavedModel
                LOGGER.info(f'Loading {w} for TensorFlow SavedModel inference...')
                import tensorflow as tf
                keras = False  # assume TF1 saved_model
                model = tf.keras.models.load_model(w) if keras else tf.saved_model.load(w)
            elif pb:  # GraphDef https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
                LOGGER.info(f'Loading {w} for TensorFlow GraphDef inference...')
                import tensorflow as tf

                def wrap_frozen_graph(gd, inputs, outputs):
                    x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped
                    ge = x.graph.as_graph_element
                    return x.prune(tf.nest.map_structure(ge, inputs), tf.nest.map_structure(ge, outputs))

                gd = tf.Graph().as_graph_def()  # graph_def
                with open(w, 'rb') as f:
                    gd.ParseFromString(f.read())
                frozen_func = wrap_frozen_graph(gd, inputs="x:0", outputs="Identity:0")
            elif tflite or edgetpu:  # https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python
                try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
                    from tflite_runtime.interpreter import Interpreter, load_delegate
                except ImportError:
                    import tensorflow as tf
                    Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate,
                if edgetpu:  # Edge TPU https://coral.ai/software/#edgetpu-runtime
                    LOGGER.info(f'Loading {w} for TensorFlow Lite Edge TPU inference...')
                    delegate = {
                        'Linux': 'libedgetpu.so.1',
                        'Darwin': 'libedgetpu.1.dylib',
                        'Windows': 'edgetpu.dll'}[platform.system()]
                    interpreter = Interpreter(model_path=w, experimental_delegates=[load_delegate(delegate)])
                else:  # Lite
                    LOGGER.info(f'Loading {w} for TensorFlow Lite inference...')
                    interpreter = Interpreter(model_path=w)  # load TFLite model
                interpreter.allocate_tensors()  # allocate
                input_details = interpreter.get_input_details()  # inputs
                output_details = interpreter.get_output_details()  # outputs
            elif tfjs:
                raise Exception('ERROR: YOLOv5 TF.js inference is not supported')
            else:
                raise Exception(f'ERROR: {w} is not a supported format')
        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False, val=False):
        # YOLOv5 MultiBackend inference
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16

        if self.pt:  # PyTorch
            y = self.model(im, augment=augment, visualize=visualize)[0]
        elif self.jit:  # TorchScript
            y = self.model(im)[0]
        elif self.dnn:  # ONNX OpenCV DNN
            im = im.cpu().numpy()  # torch to numpy
            self.net.setInput(im)
            y = self.net.forward()
        elif self.onnx:  # ONNX Runtime
            im = im.cpu().numpy()  # torch to numpy
            y = self.session.run([self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name: im})[0]
        elif self.xml:  # OpenVINO
            im = im.cpu().numpy()  # FP32
            y = self.executable_network([im])[self.output_layer]
        elif self.engine:  # TensorRT
            if self.dynamic and im.shape != self.bindings['images'].shape:
                i_in, i_out = (self.model.get_binding_index(x) for x in ('images', 'output'))
                self.context.set_binding_shape(i_in, im.shape)  # reshape if dynamic
                self.bindings['images'] = self.bindings['images']._replace(shape=im.shape)
                self.bindings['output'].data.resize_(tuple(self.context.get_binding_shape(i_out)))
            s = self.bindings['images'].shape
            assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
            self.binding_addrs['images'] = int(im.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            y = self.bindings['output'].data
        elif self.coreml:  # CoreML
            im = im.permute(0, 2, 3, 1).cpu().numpy()  # torch BCHW to numpy BHWC shape(1,320,192,3)
            im = Image.fromarray((im[0] * 255).astype('uint8'))
            # im = im.resize((192, 320), Image.ANTIALIAS)
            y = self.model.predict({'image': im})  # coordinates are xywh normalized
            if 'confidence' in y:
                box = xywh2xyxy(y['coordinates'] * [[w, h, w, h]])  # xyxy pixels
                conf, cls = y['confidence'].max(1), y['confidence'].argmax(1).astype(np.float)
                y = np.concatenate((box, conf.reshape(-1, 1), cls.reshape(-1, 1)), 1)
            else:
                k = 'var_' + str(sorted(int(k.replace('var_', '')) for k in y)[-1])  # output key
                y = y[k]  # output
        else:  # TensorFlow (SavedModel, GraphDef, Lite, Edge TPU)
            im = im.permute(0, 2, 3, 1).cpu().numpy()  # torch BCHW to numpy BHWC shape(1,320,192,3)
            if self.saved_model:  # SavedModel
                y = (self.model(im, training=False) if self.keras else self.model(im)).numpy()
            elif self.pb:  # GraphDef
                y = self.frozen_func(x=self.tf.constant(im)).numpy()
            else:  # Lite or Edge TPU
                input, output = self.input_details[0], self.output_details[0]
                int8 = input['dtype'] == np.uint8  # is TFLite quantized uint8 model
                if int8:
                    scale, zero_point = input['quantization']
                    im = (im / scale + zero_point).astype(np.uint8)  # de-scale
                self.interpreter.set_tensor(input['index'], im)
                self.interpreter.invoke()
                y = self.interpreter.get_tensor(output['index'])
                if int8:
                    scale, zero_point = output['quantization']
                    y = (y.astype(np.float32) - zero_point) * scale  # re-scale
            y[..., :4] *= [w, h, w, h]  # xywh normalized to pixels

        if isinstance(y, np.ndarray):
            y = torch.tensor(y, device=self.device)
        return (y, []) if val else y

    def warmup(self, imgsz=(1, 3, 640, 640)):
        # Warmup model by running inference once
        warmup_types = self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb
        if any(warmup_types) and self.device.type != 'cpu':
            im = torch.zeros(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
            for _ in range(2 if self.jit else 1):  #
                self.forward(im)  # warmup

    @staticmethod
    def model_type(p='path/to/model.pt'):
        # Return model type from model path, i.e. path='path/to/model.onnx' -> type=onnx
        from export import export_formats
        suffixes = list(export_formats().Suffix) + ['.xml']  # export suffixes
        check_suffix(p, suffixes)  # checks
        p = Path(p).name  # eliminate trailing separators
        pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, xml2 = (s in p for s in suffixes)
        xml |= xml2  # *_openvino_model or *.xml
        tflite &= not edgetpu  # *.tflite
        return pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs

    @staticmethod
    def _load_metadata(f='path/to/meta.yaml'):
        # Load metadata from meta.yaml if it exists
        with open(f, errors='ignore') as f:
            d = yaml.safe_load(f)
        return d['stride'], d['names']  # assign stride, names


class AutoShape(nn.Module):
    # YOLOv5 input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    agnostic = False  # NMS class-agnostic
    multi_label = False  # NMS multiple labels per box
    classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
    max_det = 1000  # maximum number of detections per image
    amp = False  # Automatic Mixed Precision (AMP) inference

    def __init__(self, model, verbose=True):
        super().__init__()
        if verbose:
            LOGGER.info('Adding AutoShape... ')
        copy_attr(self, model, include=('yaml', 'nc', 'hyp', 'names', 'stride', 'abc'), exclude=())  # copy attributes
        self.dmb = isinstance(model, DetectMultiBackend)  # DetectMultiBackend() instance
        self.pt = not self.dmb or model.pt  # PyTorch model
        self.model = model.eval()
        if self.pt:
            m = self.model.model.model[-1] if self.dmb else self.model.model[-1]  # Detect()
            m.inplace = False  # Detect.inplace=False for safe multithread inference

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        if self.pt:
            m = self.model.model.model[-1] if self.dmb else self.model.model[-1]  # Detect()
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self

    @torch.no_grad()
    def forward(self, imgs, size=640, augment=False, profile=False):
        # Inference from various sources. For height=640, width=1280, RGB images example inputs are:
        #   file:       imgs = 'data/images/zidane.jpg'  # str or PosixPath
        #   URI:             = 'https://ultralytics.com/images/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg') or ImageGrab.grab()  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        t = [time_sync()]
        p = next(self.model.parameters()) if self.pt else torch.zeros(1, device=self.model.device)  # for device, type
        autocast = self.amp and (p.device.type != 'cpu')  # Automatic Mixed Precision (AMP) inference
        if isinstance(imgs, torch.Tensor):  # torch
            with amp.autocast(autocast):
                return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        n, imgs = (len(imgs), list(imgs)) if isinstance(imgs, (list, tuple)) else (1, [imgs])  # number, list of images
        shape0, shape1, files = [], [], []  # image and inference shapes, filenames
        for i, im in enumerate(imgs):
            f = f'image{i}'  # filename
            if isinstance(im, (str, Path)):  # filename or uri
                im, f = Image.open(requests.get(im, stream=True).raw if str(im).startswith('http') else im), im
                im = np.asarray(exif_transpose(im))
            elif isinstance(im, Image.Image):  # PIL Image
                im, f = np.asarray(exif_transpose(im)), getattr(im, 'filename', f) or f
            files.append(Path(f).with_suffix('.jpg').name)
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[..., :3] if im.ndim == 3 else np.tile(im[..., None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
        shape1 = [make_divisible(x, self.stride) if self.pt else size for x in np.array(shape1).max(0)]  # inf shape
        x = [letterbox(im, shape1, auto=False)[0] for im in imgs]  # pad
        x = np.ascontiguousarray(np.array(x).transpose((0, 3, 1, 2)))  # stack and BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255  # uint8 to fp16/32
        t.append(time_sync())

        with amp.autocast(autocast):
            # Inference
            y = self.model(x, augment, profile)  # forward
            t.append(time_sync())

            # Post-process
            y = non_max_suppression(y if self.dmb else y[0],
                                    self.conf,
                                    self.iou,
                                    self.classes,
                                    self.agnostic,
                                    self.multi_label,
                                    max_det=self.max_det)  # NMS
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])

            t.append(time_sync())
            return Detections(imgs, y, files, t, self.names, x.shape)


class Detections:
    # YOLOv5 detections class for inference results
    def __init__(self, imgs, pred, files, times=(0, 0, 0, 0), names=None, shape=None):
        super().__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*(im.shape[i] for i in [1, 0, 1, 0]), 1, 1], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.times = times  # profiling times
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple((times[i + 1] - times[i]) * 1000 / self.n for i in range(3))  # timestamps (ms)
        self.s = shape  # inference BCHW shape

    def display(self, pprint=False, show=False, save=False, crop=False, render=False, labels=True, save_dir=Path('')):
        crops = []
        for i, (im, pred) in enumerate(zip(self.imgs, self.pred)):
            s = f'image {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '  # string
            if pred.shape[0]:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                if show or save or render or crop:
                    annotator = Annotator(im, example=str(self.names))
                    for *box, conf, cls in reversed(pred):  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        if crop:
                            file = save_dir / 'crops' / self.names[int(cls)] / self.files[i] if save else None
                            crops.append({
                                'box': box,
                                'conf': conf,
                                'cls': cls,
                                'label': label,
                                'im': save_one_box(box, im, file=file, save=save)})
                        else:  # all others
                            annotator.box_label(box, label if labels else '', color=colors(cls))
                    im = annotator.im
            else:
                s += '(no detections)'

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
            if pprint:
                print(s.rstrip(', '))
            if show:
                im.show(self.files[i])  # show
            if save:
                f = self.files[i]
                im.save(save_dir / f)  # save
                if i == self.n - 1:
                    LOGGER.info(f"Saved {self.n} image{'s' * (self.n > 1)} to {colorstr('bold', save_dir)}")
            if render:
                self.imgs[i] = np.asarray(im)
        if crop:
            if save:
                LOGGER.info(f'Saved results to {save_dir}\n')
            return crops

    def print(self):
        self.display(pprint=True)  # print results
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' % self.t)

    def show(self, labels=True):
        self.display(show=True, labels=labels)  # show results

    def save(self, labels=True, save_dir='runs/detect/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/detect/exp', mkdir=True)  # increment save_dir
        self.display(save=True, labels=labels, save_dir=save_dir)  # save results

    def crop(self, save=True, save_dir='runs/detect/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/detect/exp', mkdir=True) if save else None
        return self.display(crop=True, save=save, save_dir=save_dir)  # crop results

    def render(self, labels=True):
        self.display(render=True, labels=labels)  # render results
        return self.imgs

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        r = range(self.n)  # iterable
        x = [Detections([self.imgs[i]], [self.pred[i]], [self.files[i]], self.times, self.names, self.s) for i in r]
        # for d in x:
        #    for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
        #        setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def __len__(self):
        return self.n  # override len(results)

    def __str__(self):
        self.print()  # override print(results)
        return ''


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)  # to x(b,c2,1,1)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)


if __name__ == "__main__":
    model = C3_CBAM(32, 32, n=1, shortcut=True, g=1, e=0.5)
    input = torch.randn((5, 32, 8, 8))
    output = model(input)
    print()
