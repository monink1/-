# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou, wasserstein_loss, box_iou
from utils.torch_utils import de_parallel
from utils.assigner_utils import *
from utils.general import xywh2xyxy

def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, NWD_ratio=0, autobalance=False):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters
        self.NWD_ratio = NWD_ratio

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g, 0.5), FocalLoss(BCEobj, g, 0.5)

        #   获取到模型的Detect模块
        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device

    def __call__(self, p, targets):  # predictions, targets
        #   p为模型的预测输出[(bs, 3, 80, 80, 85),(bs, 3, 40, 40, 85),(bs, 3, 20, 20, 85)]
        #   targets为标签信息 target是shape为（目标数，6）的tensor，第一列表示标签所属的图片标签0~63
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        # 每一个都是append的 有feature map个 每个都是当前这个feature map中3个anchor筛选出的所有的target(3个grid_cell进行预测)
        # tcls: 表示这个target所属的class index
        # tbox: xywh 其中xy为这个target对当前grid_cell左上角的偏移量
        # indices: b: 表示这个target属于的image index
        #          a: 表示这个target使用的anchor index
        #          gj: 经过筛选后确定某个target在某个网格中进行预测(计算损失)  gj表示这个网格的左上角y坐标
        #          gi: 表示这个网格的左上角x坐标
        # anch: 表示这个target所使用anchor的尺度（相对于这个feature map）  注意可能一个target会使用大小不同anchor进行计算
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)

                if self.NWD_ratio != 0:
                    nwd_ratio = 0.5  # 平衡nwd和原始iou的权重
                    nwd = wasserstein_loss(pbox, tbox[i]).squeeze()
                    lbox += (1 - nwd_ratio) * (1.0 - nwd).mean() + nwd_ratio * (1.0 - iou).mean()  # iou loss

                    # Objectness
                    iou = (iou.detach() * nwd_ratio + nwd.detach() * (1 - nwd_ratio)).clamp(0, 1).type(tobj.dtype)
                else:
                    lbox += (1.0 - iou).mean()  # iou loss

                    # Objectness
                    iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h) target是shape为（目标数，6）的tensor，第一列表示标签所属的图片标签0~63
        #   获取anchor的数量 和目标的数量
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        #   ai的shape为（3, nt） [1, 3] -> [3, 1] -> [3, 63]=[na, nt]
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        #   原本targets.shape=[nt,6],[image,class,x,y,w,h]，repeat根据每一层的anchor数量将targets增加一维,shape[na=3,nt,6]
        #   ai[:, :, None] 2维变３维torch.Size([3, nt, 1])
        #   torch.cat两个3维的tensort在第2维上concat,targets的torch.Size([3, nt, 7])
        #   第一维增加layer的索引，并且将原本targets[image,class,x,y,w,h]在最后上增加anchor的索引[image,class,x,y,w,h,anchor indices],
        #   也就说把每个gt框分配给了每一层输出的每一个anchor [63, 6] [3, 63] -> [3, 63, 6] [3, 63, 1] -> [3, 63, 7]  7: [image_index+class+xywh+anchor_index]
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        g = 0.5  # bias 中心偏移，用来衡量target离哪个格子更近
        off = torch.tensor(
            [#   以自身+周围左上右下4个网格 = 5个网格，用来计算offsets
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets

        for i in range(self.nl):
            #   anchors：3x2
            #   shape:bs, 3, 80, 80, 85
            anchors, shape = self.anchors[i], p[i].shape
            #   读出网络第i个输出的特征图大小
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            #   将标签匹配到特征图大小
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                #   yolov5抛弃了Maxiou匹配规则，计算标签box和当前层anchors的宽高比，wb/wa,hb/ha
                #   如果宽高比大于设定的阈值说明该box在该层没有合适的anchor，在该预测层之间将这些box当背景
                #   对b中保存下来的gt进行扩充
                #   1）保存box中心点坐标Xc距离网格左边的距离小于0.5且坐标大于1的box
                #   2）保存box中心点坐标Yc距离网格上边距离小于0.5且坐标大于1的box
                #   3）保存box中心点坐标Xc距离网格右边距离小于0.5且坐标大于1的box
                #   4）保存box中心点坐标Yc距离网格下上边距离小于0.5且坐标大于1的box
                #   补充:为什么会取距离四边小于0.5的点,是因为等于0.5时,我们认为该box正好落到该网格中,
                #   但是小于0.5时,可能是因为在网络不断降采样时,对特征图尺度进行取整导致box中心产生了偏差,
                #   所以作者将小于0.5的box减去偏执1（off矩阵）,使得box中心移动到相邻的特征图网格中,
                #   从而对正样本进行扩充,保证了偏差导致的box错位以及扩充了正样本的个数
                #   anchors[:, None]将[3, 2]扩充为了[3, 1, 2]
                #   r的形状[3, nt, 2] 表示标签和anchor的比值
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                #   j的形状[3, nt]
                # 筛选条件  GT与anchor的宽比或高比超过一定的阈值 就当作负样本
                # torch.max(r, 1. / r)=[3, 63, 2] 筛选出宽比w1/w2 w2/w1 高比h1/h2 h2/h1中最大的那个
                # .max(2)返回宽比 高比两者中较大的一个值和它的索引  [0]返回较大的一个值
                # j: [3, 63]  False: 当前anchor是当前gt的负样本  True: 当前anchor是当前gt的正样本
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                # t: [3, 63, 7] -> [126, 7]  [num_Positive_sample, image_index+class+xywh+anchor_index]
                t = t[j]  # filter

                # Offsets
                #   获得box的中心点坐标
                gxy = t[:, 2:4]  # grid xy
                #   并转换为以特征图右下角为原点的坐标
                gxi = gain[[2, 3]] - gxy  # inverse
                # 筛选中心坐标 距离当前grid_cell的左、上方偏移小于g=0.5 且 中心坐标必须大于1(坐标不能在边上 此时就没有4个格子了)
                # j: [126] bool 如果是True表示当前target中心点所在的格子的左边格子也对该target进行回归(后续进行计算损失)
                # k: [126] bool 如果是True表示当前target中心点所在的格子的上边格子也对该target进行回归(后续进行计算损失)
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                # 筛选中心坐标 距离当前grid_cell的右、下方偏移小于g=0.5 且 中心坐标必须大于1(坐标不能在边上 此时就没有4个格子了)
                # l: [126] bool 如果是True表示当前target中心点所在的格子的右边格子也对该target进行回归(后续进行计算损失)
                # m: [126] bool 如果是True表示当前target中心点所在的格子的下边格子也对该target进行回归(后续进行计算损失)
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                # j: [5, 126]  torch.ones_like(j): 当前格子, 不需要筛选全是True  j, k, l, m: 左上右下格子的筛选结果
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                # 得到筛选后所有格子的正样本 格子数<=3*126 都不在边上等号成立
                # t: [126, 7] -> 复制5份target[5, 126, 7]  分别对应当前格子和左上右下格子5个格子
                # j: [5, 126] + t: [5, 126, 7] => t: [378, 7] 理论上是小于等于3倍的126 当且仅当没有边界的格子等号成立
                t = t.repeat((5, 1, 1))[j]
                # torch.zeros_like(gxy)[None]: [1, 126, 2]   off[:, None]: [5, 1, 2]  => [5, 126, 2]
                # j筛选后: [378, 2]  得到所有筛选后的网格的中心相对于这个要预测的真实框所在网格边界（左右上下边框）的偏移量
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define 将t沿1轴分为4份，7/4除不尽，分为2，2，2，1
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            # 预测真实框的网格所在的左上角坐标(有左上右下的网格)
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch




#-----------------------#
#   TOOD loss
#-----------------------#

class ComputeTLoss:
    def __init__(self, model):
        device = next(model.parameters()).device
        h = model.hyp # 超参数
        m = de_parallel(model).model[-1] # Detect_T
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.na = m.na
        self.nc = m.nc
        self.nl = m.nl
        self.no = m.no # self.na * (num_classes + 4 * reg_max)
        self.hyp = h
        self.stride = m.stride
        self.device = device
        self.reg_max = m.reg_max
        self.use_dfl = m.reg_max > 1
        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        # self.anchors = m.anchors

    def preprocess(self, targets, batch_size, scale_tensor):
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, targets):
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        # targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_bboxes /= stride_tensor
        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        # bbox loss
        if fg_mask.sum():
            loss[0], loss[1] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask)

        loss[2] *= 0.5
        loss[0] *= 7.5
        loss[1] *= 1.5

        return loss.sum() * batch_size, loss.detach()


#-----------------------#
#   distill loss
#-----------------------#
def compute_distillation_output_loss(p, t_p, model, d_weight=1):
    t_ft = torch.cuda.FloatTensor if t_p[0].is_cuda else torch.Tensor
    t_lcls, t_lbox, t_lobj = t_ft([0]), t_ft([0]), t_ft([0])
    h = model.hyp  # hyperparameters
    red = 'mean'  # Loss reduction (sum or mean)
    if red != "mean":
        raise NotImplementedError("reduction must be mean in distillation mode!")

    DboxLoss = nn.MSELoss(reduction="none")
    DclsLoss = nn.MSELoss(reduction="none")
    DobjLoss = nn.MSELoss(reduction="none")
    # per output
    for i, pi in enumerate(p):  # layer index, layer predictions
        t_pi = t_p[i]
        t_obj_scale = t_pi[..., 4].sigmoid()

        # BBox
        b_obj_scale = t_obj_scale.unsqueeze(-1).repeat(1, 1, 1, 1, 4)
        t_lbox += torch.mean(DboxLoss(pi[..., :4], t_pi[..., :4]) * b_obj_scale)

        # Class
        if model.nc > 1:  # cls loss (only if multiple classes)
            c_obj_scale = t_obj_scale.unsqueeze(-1).repeat(1, 1, 1, 1, model.nc)
            # t_lcls += torch.mean(c_obj_scale * (pi[..., 5:] - t_pi[..., 5:]) ** 2)
            t_lcls += torch.mean(DclsLoss(pi[..., 5:], t_pi[..., 5:]) * c_obj_scale)

        # t_lobj += torch.mean(t_obj_scale * (pi[..., 4] - t_pi[..., 4]) ** 2)
        t_lobj += torch.mean(DobjLoss(pi[..., 4], t_pi[..., 4]) * t_obj_scale)
    t_lbox *= h['box']
    t_lobj *= h['obj']
    t_lcls *= h['cls']
    # bs = p[0].shape[0]  # batch size
    loss = (t_lobj + t_lbox + t_lcls) * d_weight
    return loss
