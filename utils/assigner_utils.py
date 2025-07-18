import torch
import torch.nn as nn
from utils.metrics import bbox_iou
import torch.nn.functional as F
from utils.general import check_version

def generate_anchors_for_grid_cell(feats, fpn_strides, anchors, grid_cell_size=5.0, grid_cell_offset=0.5, dtype='float32'):
    '''
    anchors: tensor shape 3x3x2
    '''
    assert len(feats) == len(fpn_strides)
    anchors = []
    num_anchors = 3
    anchor_points = []
    num_anchors_list = []
    stride_tensor = []
    for anchor, feat, stride in zip(anchors, feats, fpn_strides):
        _, _, h, w = feat.shape
        new_shape = 1, num_anchors, h, w, 2
        cell_half_size = grid_cell_size * stride * 0.5
        # 在原图中特征点中心点的坐标
        shift_x = (torch.arange(end=w) + grid_cell_offset) * stride
        shift_y = (torch.arange(end=h) + grid_cell_offset) * stride
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
        anchor_point = torch.stack([shift_x, shift_y], dim=2).astype(dtype).expand(new_shape)
        anchor_m = anchor.expand(new_shape)
        anchor = torch.cat([anchor_point[..., :] - anchor_m[..., :] * 0.5, anchor_point[..., :] + anchor_m[..., :] * 0.5], dim=-1)


def select_candidate_in_gts(xy_centers, gt_bboxes, eps=1e-9):
    '''
    在目标框内部选择候选框
    xy_centers: (h*w, 4)
    gt_bboxes: (b, max_num_obj, 4)

    return:
    (b, max_num_obj, h*w)
    '''
    n_anchors = xy_centers.shape[0]
    bs, n_boxes, _ = gt_bboxes.shape
    # lt, rb: (b*max_num_obj, 1, 4)
    lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)
    # xy_centers: (h*w, 4) -> (1, h*w, 4) bbox_delta: (b*max_num_obj, h*w, 4) -> (b, max_num_obj, h*w, 4)
    bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1)
    return bbox_deltas.amin(3).gt_(eps)

def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
    '''
    如果一个anchor box被分配给了多个gts，选择iou值最大的

    mask_pos: (b, n_max_boxes, h*w)
    overlaps: (b, n_max_boxes, h*w)

    return:
    target_gt_idx: (b, h*w)
    fg_mask: (b, h*w)
    mask_pos: (b, n_max_boxes, h*w)
    '''
    # (b, n_max_boxes, h*w) -> (b, h*w)
    fg_mask = mask_pos.sum(-2)
    if fg_mask.max() > 1: # 一个anchor匹配了多个gt_box
        # (b, h*w) -> (b, 1, h*w) -> (b, n_max_boxes, h*w)
        mask_multi_gts = (fg_mask.unsqueeze(1) > 1).repeat([1, n_max_boxes, 1])
        max_overlaps_idx = overlaps.argmax(1) # (b, h*w)
        is_max_overlaps = F.one_hot(max_overlaps_idx, n_max_boxes) # (b, h*w, n_max_box)
        is_max_overlaps = is_max_overlaps.permute(0, 2, 1).to(overlaps.dtype) # (b, n_max_boxes, h*w)
        mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos)
        fg_mask = mask_pos.sum(-2)

    target_gt_idx = mask_pos.argmax(-2)# (b, h*w)
    return target_gt_idx, fg_mask, mask_pos

class TaskAlignedAssigner(nn.Module):
    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        Args:
            pd_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            pd_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            anc_points (Tensor): shape(num_total_anchors, 2)
            gt_labels (Tensor): shape(bs, n_max_boxes, 1)
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
            mask_gt (Tensor): shape(bs, n_max_boxes, 1)
        Returns:
            target_labels (Tensor): shape(bs, num_total_anchors)
            target_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            target_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            fg_mask (Tensor): shape(bs, num_total_anchors)
        """
        self.bs = pd_scores.size(0)
        self.n_max_boxes = gt_bboxes.size(1)
        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            return (torch.full_like(pd_scores[..., 0], self.bg_idx).to(device), torch.zeros_like(pd_bboxes).to(device),
                    torch.zeros_like(pd_scores).to(device), torch.zeros_like(pd_scores[..., 0]).to(device),
                    torch.zeros_like(pd_scores[..., 0]).to(device))
        # (bs, n_max_bbox, h*w)
        mask_pos, align_metric, overlaps = self.get_pos_mask(pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt)
        # target_gt_ix: (bs, h*w) fg_mask: (bs, h*w) mask_pos: (b, n_max_boxes, h*w)
        target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)
        # (bs, h*w) (bs, h*w) (bs, h*w, 80)
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # normalize
        align_metric *= mask_pos # (bs, max_num_obj, h*w)
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True) # (bs, max_num_obj, h*w)
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True) # (bs, max_num_obj, h*w)
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics +self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        '''
        gt_labels: (bs, max_num_obj, 1)
        gt_bboxes: (bs, max_num_obj, 4)
        target_gt_idx: (bs, h*w)
        fg_mask: (bs, h*w)
        '''
        # assigned target labels (b, 1)
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes # (b, h*w)
        target_labels = gt_labels.long().flatten()[target_gt_idx] # (b, h*w)

        # assigned target boxes, (b, max_num_obj, 4) -> (b, h*w)
        target_bboxes = gt_bboxes.view(-1, 4)[target_gt_idx]

        # assigned target scores
        target_labels.clamp(0)
        target_scores = F.one_hot(target_labels, self.num_classes) # (b, h*w, 80)
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes) # (b, h*w, 80)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)
        return target_labels, target_bboxes, target_scores

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        # align_metri, over_laps: (bs, max_num_obj, h*w)
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes)
        mask_in_gts = select_candidate_in_gts(anc_points, gt_bboxes) #(bs, nmax_num_obj, h*w) 其值为0或1
        # align_metric * mask_in_gts 把在gt_bboxes中的anc_points置为1
        # mask_topk: (bs, max_num_obj, h*w)
        mask_topk = self.select_topk_candidates(align_metric * mask_in_gts, topk_mask=mask_gt.repeat([1, 1, self.topk]).bool())
        mask_pos = mask_topk * mask_in_gts * mask_gt
        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes):
        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long) # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).repeat(1, self.n_max_boxes) # b, max_num_obj
        ind[1] = gt_labels.long().squeeze(-1) # b, max_num_obj
        # b, max_num_obj, h*w
        bbox_scores = pd_scores[ind[0], :, ind[1]]
        # gt_bboxes:(bs, max_num_obj, 4) -> (bs, max_num_0bj, 1, 4) pd_bboxes:(bs, h*w, 4) -> (bs, 1, h*w, 4)
        # overlaps: (bs, max_num_obj, h*w) 每个gt_bbox和每个pd_bbox之间的IoU
        over_laps = bbox_iou(gt_bboxes.unsqueeze(2), pd_bboxes.unsqueeze(1), xywh=False, CIoU=True).squeeze(3).clamp(0)
        align_metric = bbox_scores.pow(self.alpha) * over_laps.pow(self.beta)
        return align_metric, over_laps

    def select_topk_candidates(self, metrics, largest=True, topk_mask=None):
        '''
        metrics: (b, max_num_obj, h*w)
        topk_mask: (b, max_num_obj, topk)
        '''
        num_anchors = metrics.shape[-1] #h*w
        # (b, max_num_obj, topk)
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=largest)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True) > self.eps).tile([1, 1, self.topk])
        topk_idxs = torch.where(topk_mask, topk_idxs, 0)
        # is_in_topk: (b, max_num_obj, topk) -> (b, max_num_obj, topk, h*w) -> (b, max_num_obj, h*w)
        is_in_topk = F.one_hot(topk_idxs, num_anchors).sum(-2)
        # 过滤掉无效的bbox
        is_in_topk = torch.where(is_in_topk > 1, 0, is_in_topk)
        return is_in_topk.to(metrics.dtype)


# def make_anchors(feats, strides, grid_cell_offset=0.5, num_anchors=3):
#     anchor_points, stride_tensor = [], []
#     assert feats is not None
#     dtype, device = feats[0].dtype, feats[0].device
#     for i, stride in enumerate(strides):
#         _, _, h, w = feats[i].shape
#         sx = torch.arange(end=w, dtype=dtype, device=device) + grid_cell_offset
#         sy = torch.arange(end=h, dtype=dtype, device=device) + grid_cell_offset
#         sy, sx = torch.meshgrid(sy, sx, indexing='ij') if check_version(torch.__version__, '1.10.0') else torch.meshgrid(sy, sx)
#         anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2).repeat(num_anchors, 1, 1)) #num_anchor, h*w, 2
#         stride_tensor.append(torch.full((num_anchors, h * w, 1), stride, dtype=dtype, device=device)) # num_anchor, h*w, 1
#     # (3, N, 2) (3, N, 1)  N=hw+hw+hw
#     return torch.cat(anchor_points, dim=1), torch.cat(stride_tensor, dim=1)

def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing='ij') if check_version(torch.__version__, '1.10.0') else torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)

def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = torch.split(distance, 2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox

def bbox2dist(anchor_points, bbox, reg_max):
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = torch.split(bbox, 2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp(0, reg_max - 0.01)  # dist (lt, rb)

class BboxLoss(nn.Module):

    def __init__(self, reg_max, use_dfl=False):
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        # IoU loss
        weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        # Return sum of left and right DFL losses
        # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl +
                F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr).mean(-1, keepdim=True)
