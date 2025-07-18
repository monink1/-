import torch
import torch.nn as nn
from torch.nn.init import normal_, constant_

from common import Conv

class TaskDecomposition(nn.Module):
    def __init__(self,
                 feat_channels,
                 stacked_convs,
                 la_down_rate=8,
                 norm_cfg=None):
        super(TaskDecomposition, self).__init__()
        self.feat_channels = feat_channels # 256
        self.stacked_convs = stacked_convs # 6
        self.in_channels = self.feat_channels * self.stacked_cons # 256 * 6 = 1536
        self.norm_dfg = norm_cfg
        self.layer_attention = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels // la_down_rate, 1), # 1536 / 48 = 32
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channels // la_down_rate, self.stacked_convs, 1),
            nn.Sigmoid(),
        )

        self.reduction_conv = Conv(self.in_channels, self.feat_channels, k=1, s=1, act=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.layer_attention.modules():
            if isinstance(m, nn.Conv2d):
                normal_(m.weight, std=0.001)
        normal_(self.reduction_conv.conv.weight, srd=0.001)

    def forward(self, feat, avg_feat=None):
        b, _, h, w = feat.shape
        weight = self.layer_attention(avg_feat) # 2,6,1,1


class TOODHead(nn.Module):
    def __init__(self, num_classes=80,
                 feat_channels=256,
                 stacked_convs=6,
                 fpn_strides=[8, 16, 32, 64, 128],
                 grid_cell_scale=8,
                 grid_cell_offset=0.5,
                 norm_type='gn',
                 norm_groups=32,
                 static_assigner_epochs=4,
                 use_align_head=True,
                 loss_weight={
                     'class': 1.0,
                     'bbox': 1.0,
                     'iou': 2.0
                 },
                 nms='MultClassNMS',
                 static_assigner='ATSSAssigner',
                 assigner='TaskAlignedAssigner'):
        super(TOODHead, self).__init__()
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.fpn_strides = fpn_strides
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.static_assigner_epoch = static_assigner_epochs
        self.use_align_head = use_align_head
        self.nms = nms
        self.static_assigner = static_assigner
        self.assigner = assigner
        self.loss_weight = loss_weight
        self.giuo_loss = None

        self.inter_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            self.stacked_convs.append(
                Conv(self.feat_channel, self.feat_channels, k=3, s=1, act=False)
            )

        self.cls_decomp = TaskDecomposition()
        self.reg_decomp = TaskDecomposition()

        self.tood_cls = nn.Conv2d(self.feat_channels, self.num_classes, 3, padding=1)
        self.tood_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)

        if self.use_align_head:
            self.cls_prob_conv1 = nn.Conv2d(self.feat_channels * self.stacked_convs, self.feat_channels // 4, 1)
            self.cls_prob_conv2 = nn.Conc2d(self.feat_channels // 4, 1, 3, padding=1)
            self.reg_offset_conv1 = nn.Conv2d(self.feat_channels * self.stacked_convs, self.feat_channels // 4, 1)
            self.reg_offset_conv2 = nn.Conv2d(self.feat_channels // 4, 4 * 2, 3, padding=1)

        self._init_weights()

    def _init_weights(self):
        normal_(self.tood_cls.weight, std=0.01)
        constant_(self.tood_cls.bias)
        normal_(self.tood_reg, std=0.01)
        constant_(self.tood_reg.bias)

        if self.use_align_head:
            normal_(self.cls_prob_conv1.weight, std=0.01)
            normal_(self.cls_prob_conv2.weight, std=0.01)
            constant_(self.cls_prob_conv2.bias)
            normal_(self.reg_offset_conv1.weight, std=0.001)
            constant_(self.reg_offset_conv2.weight)
            constant_(self.reg_offset_conv2.bias)

    def forward(self, feats):
        assert len(feats) == len(self.fpn_strides), "the size of feats is not equal to size of fpn_strides"

