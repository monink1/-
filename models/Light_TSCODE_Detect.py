from einops import rearrange
import torch.nn as nn
import torch
from models.common import Conv
import math
from utils.general import check_version

class TSCODE_Detect(nn.Module):
    # YOLOv5 Detect head for detection models
    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m_sce = nn.ModuleList(SCE(ch[id:id + 2]) for id in range(1, len(ch) - 1))
        self.m_dpe = nn.ModuleList(DPE(ch[id - 1:id + 2], ch[id]) for id in range(1, len(ch) - 1))

        self.m_cls = nn.ModuleList(nn.Sequential(Conv(sum(ch[id:id + 2]), ch[id], 3), nn.Conv2d(ch[id], self.na * self.nc * 4, 1)) for id in range(1, len(ch) - 1))  # cls conv
        self.m_reg_conf = nn.ModuleList(Conv(ch[id], ch[id], 3) for id in range(1, len(ch) - 1))  # reg_conf stem conv
        self.m_reg = nn.ModuleList(nn.Conv2d(ch[id], self.na * 4, 1) for id in range(1, len(ch) - 1))  # reg conv
        self.m_conf = nn.ModuleList(nn.Conv2d(ch[id], self.na * 1, 1) for id in range(1, len(ch) - 1))  # conf conv
        self.ph, self.pw = 2, 2

        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

    def forward(self, x_):
        x, z = [], []  # inference output
        for i, idx in enumerate(range(1, self.nl + 1)):
            bs, _, ny, nx = x_[idx].shape

            x_sce, x_dpe = self.m_sce[i](x_[idx:idx + 2]), self.m_dpe[i](x_[idx - 1:idx + 2])
            x_cls = rearrange(self.m_cls[i](x_sce), 'bs (nl ph pw nc) h w -> bs nl nc (h ph) (w pw)', nl=self.nl, ph=self.ph, pw=self.pw, nc=self.nc)
            x_cls = x_cls.permute(0, 1, 3, 4, 2).contiguous()

            x_reg_conf = self.m_reg_conf[i](x_dpe)
            x_reg = self.m_reg[i](x_reg_conf).view(bs, self.na, 4, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            x_conf = self.m_conf[i](x_reg_conf).view(bs, self.na, 1, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            x.append(torch.cat([x_reg, x_conf, x_cls], dim=4))

            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, '1.10.0')):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing='ij') if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid

### Task-Specific Context Decoupling for Object Detection

class SCE(nn.Module):
    def __init__(self, c1):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.down = Conv(c1[0], c1[0])

    def forward(self, x):
        x_p1, x_p2 = x
        x = torch.concat([self.down(self.pool(x_p1)), x_p2], dim=1)
        return x


class DPE(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.adjust_channel_forp1 = Conv(c1[0], c2, k=1)


        self.up_forp2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
        )
        self.up_forp3 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv(c1[2], c2, k=1)
        )
        self.down = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x_p2 = x[1]
        x_p1 = self.adjust_channel_forp1(x[0]) + self.up_forp2(x_p2)
        x_p1 = self.down(x_p1)

        x_p3 = self.up_forp3(x[2])

        return x_p1 + x_p2 + x_p3

