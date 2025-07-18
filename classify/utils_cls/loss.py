import torch
import torch.nn as nn
import torch.nn.functional as F


class Focal_loss(nn.Module):
    def __init__(self, alpha=None, gamma=None, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(Focal_loss, self).__init__()
        '''
        alpha: 列表，每个对象是类别的权重
        gamma：常数
        '''
        self.alpha = alpha
        self.gamma = gamma
        self.device = device

    def forward(self, pred, target):
        #   将target转换为one_hot向量
        target_one_hot = torch.zeros(pred.shape[0], pred.shape[1]).scatter_(1, target.data.cpu().view((-1, 1)), 1).to(
            self.device)
        log_pred = -F.log_softmax(pred, dim=1)
        pred_gamma = (1.0 - F.softmax(pred, dim=1)) ** self.gamma

        if self.alpha is not None:
            return ((self.alpha * pred_gamma * (log_pred * target_one_hot)).sum()) / pred.shape[0]
        return ((log_pred * target_one_hot).sum()) / pred.shape[0]
