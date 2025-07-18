# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python path/to/models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import contextlib
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

sys.path.append(str(ROOT / "prune_"))
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import (fuse_conv_and_bn, initialize_weights, model_info, profile, scale_img, select_device,
                               time_sync)

from utils.assigner_utils import make_anchors, dist2bbox
from models.BiFormer import BiLevelRoutingAttention
from models.TSCODE_Detect import TSCODE_Detect
from models.GFPN import CSPStage
from models.FasterNet import *
from models.Conv2Former import Conv2Formers

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor →每个锚框的预测结果个数：类别数+5（边框4+置信度）
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors  →特征图每个像素点对应的锚框数
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output

        a = [None] * len(x)  # changed !!!

        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv

            a[i] = torch.sigmoid(x[i])  # changed !!!  用于转换为onnx模型

            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85) → batch_size、每个特征图点对应的预测类别数
            #   将第i个输出的shape变为(bs, num_anchor, 5+num_class, w, h)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()  # shape：(bs, self.na,  ny, nx,self.no)

            if not self.training:  # inference
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    #  制作第i预测层的网格
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                if self.inplace:
                    #   这里没有减0.5，因为_make_grid的时候已经减去了0.5
                    y[..., 0:2] = (y[..., 0:2] * 2 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.no))

        hf_mode = True  # hf_mode 下onnx模型为多个检测头输出[[1,3*(5+classes),p1,p1],[1,3*(5+classes),p2,p2],[1,3*(5+classes),p3,p3]]; 非hf_mode 下 onnx模型输出归并为一个,[1,候选框总数, 5+classes]
        if self.export:
            print(f"^*^ ^*^ Exporting....  ^*^ ^*^")
        if hf_mode:
            return x if self.training else a if self.export else (torch.cat(z, 1), x)  # changed !!!
        else:
            return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid(y, x, indexing='ij')
        else:
            yv, xv = torch.meshgrid(y, x)
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


class Detect_simota(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            # test = F.sigmoid(x[i].reshape((-1, )))
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)

            if not self.training:  # inference
                x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    #   制作第i预测层的网格
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                if self.inplace:
                    #   这里没有减0.5，因为_make_grid的时候已经减去了0.5
                    y[..., 0:2] = (y[..., 0:2] * 2 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        # self.stride = self.stride.to(d)
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid(y, x, indexing='ij')
        else:
            yv, xv = torch.meshgrid(y, x)
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


class Decoupled_Detect(nn.Module):
    stride = None
    onnx_dynamic = False
    export = False

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):
        super(Decoupled_Detect, self).__init__()
        self.nc = nc  # 类别数量
        self.no = nc + 5  # 每个anchor输出的特征数量
        self.nl = len(anchors)  # 输出检测层的数量
        self.na = len(anchors[0]) // 2  # anchor的数量
        self.grid = [torch.zeros(1)] * self.nl  # 初始化anchor网格
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))
        self.m = nn.ModuleList(LightHead(x, nc, anchors) for x in ch)
        # self.m = nn.ModuleList(DecoupledHead(x, nc, anchors) for x in ch) 解耦头
        self.inplace = inplace

    def forward(self, x):
        z = []  # 推理时的输出
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 + self.grid[i]) * self.stride[i]
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                else:
                    self.grid[i], self.anchor_grid[i] = self.grid[i].to(self.anchors.device), self.anchor_grid[i].to(
                        self.anchors.device)
                    xy, wh, conf = y.split((2, 2, self.nc + 1), 4)
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]
                    y = torch.cat((xy, wh, conf), dim=4)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # anchor的形状
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        if check_version(torch.__version__, '1.10.0'):
            yv, xv = torch.meshgrid(y, x, indexing='ij')
        else:
            yv, xv = torch.meshgrid(y, x)
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


class MixDecoupled_Detect(nn.Module):
    stride = None
    onnx_dynamic = False
    export = False

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):
        super(MixDecoupled_Detect, self).__init__()
        self.nc = nc  # 类别数量
        self.no = nc + 5  # 每个anchor输出的特征数量
        self.nl = len(anchors)  # 输出检测层的数量
        self.na = len(anchors[0]) // 2  # anchor的数量
        self.grid = [torch.zeros(1)] * self.nl  # 初始化anchor网格
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))
        self.conv_cls = nn.ModuleList(nn.Conv2d(x, self.na * (self.nc + 1), kernel_size=1, stride=1) for x in ch[0::2])
        self.conv_reg = nn.ModuleList(nn.Conv2d(x, self.na * 4, kernel_size=1, stride=1) for x in ch[1::2])

        self.inplace = inplace

    def forward(self, input):
        x = input[:3]
        z = []  # 推理时的输出
        for i in range(self.nl):
            x[i] = torch.cat([self.conv_reg[i](input[i * 2]), self.conv_cls[i](input[i * 2 + 1])], dim=1)
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)

            if not self.training:  # inference
                x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    #   制作第i预测层的网格
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                if self.inplace:
                    #   这里没有减0.5，因为_make_grid的时候已经减去了0.5
                    y[..., 0:2] = (y[..., 0:2] * 2 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # anchor的形状
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        if check_version(torch.__version__, '1.10.0'):
            yv, xv = torch.meshgrid(y, x, indexing='ij')
        else:
            yv, xv = torch.meshgrid(y, x)
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


class ASFF_Detect(nn.Module):  # add ASFFV5 layer and Rfb
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), multiplier=0.25, rfb=False, inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.l0_fusion = ASFFV5(level=0, multiplier=multiplier, rfb=rfb)
        self.l1_fusion = ASFFV5(level=1, multiplier=multiplier, rfb=rfb)
        self.l2_fusion = ASFFV5(level=2, multiplier=multiplier, rfb=rfb)
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        result = []

        result.append(self.l2_fusion(x))
        result.append(self.l1_fusion(x))
        result.append(self.l0_fusion(x))
        x = result
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()  # https://github.com/iscyy/yoloair
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid(y, x, indexing='ij')
        else:
            yv, xv = torch.meshgrid(y, x)
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        # print(anchor_grid)
        return grid, anchor_grid


class DFL(nn.Module):
    # Integral module of Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


# class Detect_T(nn.Module):
#     stride = None  # strides computed during build
#     onnx_dynamic = False  # ONNX export parameter
#     export = False  # export mode
#     shape = None
#
#     def __init__(self, nc=80, anchors=(), ch=(), reg_max=16, inplace=True):  # detection layer
#         super().__init__()
#         self.nc = nc  # number of classes
#         self.reg_max = reg_max
#         self.no = nc + 4 * reg_max  # number of outputs per anchor
#         self.nl = len(anchors)  # number of detection layers
#         self.na = 1  # number of anchors
#         self.grid = [torch.zeros(1)] * self.nl  # init grid
#         self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
#         self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
#         c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], self.nc)
#         self.cv2 = nn.ModuleList(nn.Sequential(Conv(x, c2, k=3), Conv(c2, c2, k=3), nn.Conv2d(c2, 4 * self.reg_max, kernel_size=1)) for x in ch)
#         self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, k=3), Conv(c3, c3, k=3), nn.Conv2d(c3, self.nc, kernel_size=1)) for x in ch)
#         self.inplace = inplace  # use inplace ops (e.g. slice assignment)
#         self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
#
#     def forward(self, x):
#         shape = x[0].shape  # BCHW
#         for i in range(self.nl):
#             x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), dim=1)  # conv
#         if self.training:
#             return x
#         elif self.shape != shape:  # inference
#             self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5, self.na))
#             self.shape = shape
#             self.anchors = self.anchors.view((2, -1))
#             self.strides = self.strides.view((1, -1))
#         box, cls = torch.cat([xi.view((shape[0], self.no, -1)) for xi in x], 2).split((self.reg_max * 4, self.nc), 1)
#         # ltrb -> xywh(原图）
#         dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
#         y = torch.cat((dbox, cls.sigmoid()), 1)
#         return y if self.export else(y, x)
#
#     def bias_init(self):
#         # Initialize Detect() biases, WARNING: requires stride availability
#         m = self  # self.model[-1]  # Detect() module
#         # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
#         # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
#         for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
#             a[-1].bias.data[:] = 1.0  # box
#             b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

class Detect_T(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter
    export = False  # export mode
    shape = None

    def __init__(self, nc=80, anchors=(), ch=(), reg_max=16, inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.reg_max = reg_max
        self.na = len(anchors[0]) // 2  # number of anchors
        self.no = nc + 4 * reg_max  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        # self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], self.nc)
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, k=3), Conv(c2, c2, k=3), nn.Conv2d(c2, 4 * self.reg_max, kernel_size=1)) for x in
            ch)
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, k=3), Conv(c3, c3, k=3), nn.Conv2d(c3, self.nc, kernel_size=1)) for x in ch)
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
        self.proj = torch.arange(self.reg_max, dtype=torch.float)

    def forward(self, x):
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), dim=1)  # conv
        if self.training:
            return x
        elif self.shape != shape:  # inference
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape
            # self.anchors = self.anchors.view((2, -1))
            # self.strides = self.strides.view((1, -1))
        box, cls = torch.cat([xi.view((shape[0], self.no, -1)) for xi in x], 2).split((self.reg_max * 4, self.nc), 1)
        # ltrb -> xywh(输入大小）
        box = box.permute((0, 2, 1))
        box = box.view(box.shape[0], box.shape[1], 4, box.shape[2] // 4).softmax(3).matmul(
            self.proj.type(box.dtype).to(box.device))
        box = box.permute(0, 2, 1)
        dbox = dist2bbox(box, self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        # Initialize Detect() biases, WARNING: requires stride availability
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)


class Model(nn.Module):
    # YOLOv5 model
    def __init__(self, cfg='yolov5l.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super().__init__()
        if isinstance(cfg, dict):  # 判断cfg是不是字典类型
            self.yaml = cfg  # model dict 将模型字典赋给self.yaml
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict self.yaml是字典形式的配置信息

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        #   判断输入的类别数目和yaml中的类别数目是否相同
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value

        #   对anchors进行四舍五入，防止输入的小数而报错
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        #   给种类编号 0~nc-1
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 640  # 2x min stride
            m.inplace = self.inplace
            #   self.forward是模型的输出，是又个对象的列表
            #   m.stride是每个输出对应的下采样的倍数
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            #   根据yolov5Detect（）的步幅顺序检查锚框顺序，必要时进行纠正
            check_anchor_order(m)  # must be in pixel-space (not grid-space)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()  # only run once
        if isinstance(m, Detect_simota):
            s = 640  # 2x min stride
            m.inplace = self.inplace
            #   self.forward是模型的输出，是又个对象的列表
            #   m.stride是每个输出对应的下采样的倍数
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            #   根据yolov5Detect（）的步幅顺序检查锚框顺序，必要时进行纠正
            check_anchor_order(m)  # must be in pixel-space (not grid-space)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()  # only run once
        if isinstance(m, (Decoupled_Detect, ASFF_Detect, TSCODE_Detect)):
            s = 640  # 2x min stride
            m.inplace = self.inplace
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            try:
                self._initialize_biases()  # only run once
                LOGGER.info('initialize_biases done')
            except:
                LOGGER.info('decoupled no biase ')
        if isinstance(m, Detect_T):
            s = 640
            m.inplace = self.inplace
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])
            self.stride = m.stride
            m.bias_init()
        if isinstance(m, MixDecoupled_Detect):
            s = 640  # 2x min stride
            m.inplace = self.inplace
            #   self.forward是模型的输出，是又个对象的列表
            #   m.stride是每个输出对应的下采样的倍数
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            #   根据yolov5Detect（）的步幅顺序检查锚框顺序，必要时进行纠正
            check_anchor_order(m)  # must be in pixel-space (not grid-space)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride

        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []  # outputs
        #   m是model中的模块，如conv， C3
        for m in self.model:
            #   在head层中m.f != -1时有用
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _profile_one_layer(self, m, x, dt):
        c = isinstance(m, (Detect, Detect_simota, TSCODE_Detect, Decoupled_Detect, ASFF_Detect,
                           Detect_T))  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        if isinstance(m, TSCODE_Detect):
            for mi, s in zip(m.m_conf, m.stride):  # from
                b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
                b.data += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
                mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

            for mi, s in zip(m.m_cls, m.stride):  # from
                b = mi[-1].bias.view(m.na, -1)  # conv.bias(255) to (3,85)
                b.data += math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())  # cls
                mi[-1].bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        else:
            for mi, s in zip(m.m, m.stride):  # from
                b = mi.bias.view(m.na, -1).detach()  # conv.bias(255) to (3,85)
                b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
                b[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
                mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            LOGGER.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             LOGGER.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, RepConv):
                m.fuse_repvgg_block()
            elif isinstance(m, RepGhostConv):
                m.fuse_repvgg_block()
            elif isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Detect_simota, Decoupled_Detect, Detect_T, MixDecoupled_Detect, TSCODE_Detect, ASFF_Detect)):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


def parse_model(d, ch):  # model_dict, input_channels(3)
    # 日志记载
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")

    #   读取配置dict中的参数
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings 将模块类型转化为值
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):  # contextlib.suppress实现选择性忽略特定异常
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings 将每个模块参数args[j]转换为值

        #   模块的数量 在深度上缩放
        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain

        if m in (
                Conv, DeConv, ODConv2d, ODConv, DilateConv, ConvTri, C3_Tri, GSConv, VoVGSCSP, RepConv, GhostConv,
                gnconv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x, C3_CBAM, CA,
                C3BiFormer_1, C3HB, C3STR, EVCBlock, C3_REP, C3_REP_CBAM, RepGhostConv, C3RepGhost, C3_ODConv, C3_MLP,
                C3_MLP_PCBAM, CNeB, AAM, FEM, BottleneckCSP_v4, C3_MLP, C3_PCBAM, C3_MLP_PAM, CRFB, C3_DenseGhost_MLP,
                GhostPoolConv, GhostConvP, C3_Ghost_MLP, CSPStage, C3_BiFormer_Block, C3BiFormer, BasicStage,
                PatchEmbed, PatchMerging, C3_DRConv, RepBlock, Conv2Formers, C3_DCN):
            #   ch用于保存之前所有模块的输出channel
            #   args[0]是默认的输出通道
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                #   如果不是最终输出，保证输出的通道数是8的倍数，通过gw参数调整通道数
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3, C3TR, C3Ghost, C3x, C3_CBAM, C3HB, C3STR, C3_REP, C3_REP_CBAM, C3RepGhost,
                     C3_MLP, C3_MLP_PCBAM, C3_ODConv, VoVGSCSP, CNeB, BottleneckCSP_v4,
                     C3_MLP, C3_PCBAM, C3_MLP_PAM, CRFB, C3_DenseGhost_MLP, C3_Ghost_MLP, CSPStage, C3_BiFormer_Block,
                     C3BiFormer, C3BiFormer_1, C3_DRConv, Conv2Formers, C3_DCN]:
                args.insert(2, n)  # number of repeats
                n = 1
            if m in [BasicStage]:
                args.pop(1)
        elif m is RepGhostBottleneck:
            c1 = ch[f]
            args[0] = make_divisible(args[0] * args[-1], 4)
            args[1] = make_divisible(args[1] * args[-1], 4)
            c2 = args[1]
            args = [c1, *args]
        elif m is SOCA:
            c1, c2 = ch[f], args[0]
            if c2 != no:
                c2 = make_divisible(c2 * gw, 8)
            args = [c1, *args[1:]]
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is ADD:
            c2 = ch[f[0]]
        elif m is Concat_BiFPN:
            c2 = max([ch[x] for x in f])
        elif m is RFACAConv:
            c2 = ch[f]
            args = [c2, *args]
        elif m is BiFormerBlock:
            c2 = ch[f]
            args = [c2, *args]
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Detect_simota:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is ASFF_Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif (m is Detect_T) or (m is MixDecoupled_Detect):
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is ACmix:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)
            args = [c1, *args[1:]]
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is SPD_conv:
            c2 = 4 * ch[f]
        elif m is Efficient1 or m is Efficient2 or m is Efficient3:
            c2 = args[0]
        elif m is CoordAttn:
            inp = args[0]
            inp = make_divisible(inp * gw, 8) if inp != no else inp
            args = [inp]
        elif m is BiLevelRoutingAttention:
            c2 = ch[f]
            args = [c2, *args]
        elif m in [Decoupled_Detect, TSCODE_Detect]:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        #   拿args里的参数去构建module m_，模块的循环次数用n控制，整体受到宽度缩放系数影响，C3模块受到深度缩放影响
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        #   保存需要用的层的输出（如concat层需要concat某些层，这些层的结果就需要存起来）
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        #   把构建的模块保存到layers中
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)  # 把该层的输出通道保存到ch列表里
    return nn.Sequential(*layers), sorted(save)


# -- 剪枝相关 --
from prune_.prune_model import *


def parse_pruned_model(maskbndict, d, ch):
    LOGGER.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    # gd, gw = 0.33, 0.50
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)
    ch = [3]
    fromlayer = []  # last module bn layer name
    from_to_map = {}
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out

    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain

        named_m_base = "model.{}".format(i)
        if m in [Conv]:
            named_m_bn = named_m_base + '.bn'
            bnc = int(maskbndict[named_m_bn].sum())
            c1, c2 = ch[f], bnc
            args = [c1, c2, *args[1:]]
            layertmp = named_m_bn
            if i != 0:
                from_to_map[layertmp] = fromlayer[f]
            fromlayer.append(named_m_bn)

        elif m in [Focus]:
            named_m_bn = named_m_base + ".conv.bn"
            bnc = int(maskbndict[named_m_bn].sum())
            c1, c2 = ch[f], bnc

            args = [c1, c2, *args[1:]]
            layertmp = named_m_bn
            # from_to_map[layertmp] = fromlayer[f]
            fromlayer.append(named_m_bn)

        elif m in [C3Pruned]:
            named_m_cv1_bn = named_m_base + ".cv1.bn"
            named_m_cv2_bn = named_m_base + ".cv2.bn"
            named_m_cv3_bn = named_m_base + ".cv3.bn"
            from_to_map[named_m_cv1_bn] = fromlayer[f]
            from_to_map[named_m_cv2_bn] = fromlayer[f]

            fromlayer.append(named_m_cv3_bn)

            cv1in = ch[f]
            cv1out = int(maskbndict[named_m_cv1_bn].sum())
            cv2out = int(maskbndict[named_m_cv2_bn].sum())
            cv3out = int(maskbndict[named_m_cv3_bn].sum())
            args = [cv1in, cv1out, cv2out, cv3out, n, args[-1]]
            bottle_args = []
            chin = [cv1out]

            c3fromlayer = [named_m_cv1_bn]
            for p in range(n):
                named_m_bottle_cv1_bn = named_m_base + ".m.{}.cv1.bn".format(p)
                named_m_bottle_cv2_bn = named_m_base + ".m.{}.cv2.bn".format(p)
                bottle_cv1in = chin[-1]
                bottle_cv1out = int(maskbndict[named_m_bottle_cv1_bn].sum())
                bottle_cv2out = int(maskbndict[named_m_bottle_cv2_bn].sum())
                chin.append(bottle_cv2out)
                bottle_args.append([bottle_cv1in, bottle_cv1out, bottle_cv2out])
                from_to_map[named_m_bottle_cv1_bn] = c3fromlayer[p]
                from_to_map[named_m_bottle_cv2_bn] = named_m_bottle_cv1_bn
                c3fromlayer.append(named_m_bottle_cv2_bn)
            args.insert(4, bottle_args)
            c2 = cv3out
            n = 1
            from_to_map[named_m_cv3_bn] = [c3fromlayer[-1], named_m_cv2_bn]

        elif m in [SPPPruned]:
            named_m_cv1_bn = named_m_base + ".cv1.bn"
            named_m_cv2_bn = named_m_base + ".cv2.bn"
            cv1in = ch[f]
            from_to_map[named_m_cv1_bn] = fromlayer[f]
            from_to_map[named_m_cv2_bn] = [named_m_cv1_bn] * 4
            fromlayer.append(named_m_cv2_bn)
            cv1out = int(maskbndict[named_m_cv1_bn].sum())
            cv2out = int(maskbndict[named_m_cv2_bn].sum())
            args = [cv1in, cv1out, cv2out, *args[1:]]
            c2 = cv2out

        elif m in [SPPFPruned]:
            named_m_cv1_bn = named_m_base + ".cv1.bn"
            named_m_cv2_bn = named_m_base + ".cv2.bn"
            cv1in = ch[f]
            from_to_map[named_m_cv1_bn] = fromlayer[f]
            from_to_map[named_m_cv2_bn] = [named_m_cv1_bn] * 4
            fromlayer.append(named_m_cv2_bn)
            cv1out = int(maskbndict[named_m_cv1_bn].sum())
            cv2out = int(maskbndict[named_m_cv2_bn].sum())
            args = [cv1in, cv1out, cv2out, *args[1:]]
            c2 = cv2out

        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
            inputtmp = [fromlayer[x] for x in f]
            fromlayer.append(inputtmp)
        elif m is Detect:
            print("detect input : ", fromlayer, len(fromlayer), f)
            from_to_map[named_m_base + ".m.0"] = fromlayer[f[0]]
            from_to_map[named_m_base + ".m.1"] = fromlayer[f[1]]
            from_to_map[named_m_base + ".m.2"] = fromlayer[f[2]]
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]
            fromtmp = fromlayer[-1]
            fromlayer.append(fromtmp)

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        print('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)

    return nn.Sequential(*layers), sorted(save), from_to_map


class ModelPruned(nn.Module):
    def __init__(self, maskbndict, cfg='yolov5s.yaml', ch=3, nc=None,
                 anchors=None):  # model, input channels, number of classes
        super(ModelPruned, self).__init__()
        self.maskbndict = maskbndict
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.safe_load(f)  # model dict

        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save, self.from_to_map = parse_pruned_model(self.maskbndict, deepcopy(self.yaml),
                                                                     ch=[ch])  # model, savelist

        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)
        # logger.info([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # logger.info('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info('')

    def forward(self, x, augment=False, profile=False):
        if augment:
            return self.forward_augment(x)  # augmented inference, None
        else:
            return self.forward_once(x, profile)  # single-scale inference, train

    def forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self.forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        return torch.cat(y, 1), None  # augmented inference, train

    def forward_once(self, x, profile=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile:
                o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                t = time_sync()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_sync() - t) * 100)
                if m == self.model[0]:
                    LOGGER.info(f"{'time (ms)':>10s} {'GFLOPS':>10s} {'params':>10s}  {'module'}")
                LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        if profile:
            LOGGER.info('%.1fms total' % sum(dt))
        return x

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            LOGGER.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             logger.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def nms(self, mode=True):  # add or remove NMS module
        pass

    def autoshape(self):  # add autoShape module
        pass

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    parser.add_argument('--line-profile', action='store_true', help='profile model speed layer by layer')
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(vars(opt))
    device = select_device(opt.device)

    # Create model
    im = torch.rand(opt.batch_size, 3, 640, 640).to(device)
    model = Model(opt.cfg).to(device)

    # Options
    if opt.line_profile:  # profile layer by layer
        _ = model(im, profile=True)

    elif opt.profile:  # profile forward-backward
        results = profile(input=im, ops=[model], n=3)

    elif opt.test:  # test all models
        for cfg in Path(ROOT / 'models').rglob('yolo*.yaml'):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f'Error in {cfg}: {e}')

    else:  # report fused model summary
        model.fuse()
