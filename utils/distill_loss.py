import torch.nn as nn
import torch
import torch.nn.functional as F

class FeatureLoss(nn.Module):
    '''
    蒸馏损失的计算
    student_channels（int）：学生模型的特征图的通道数
    teacher_channels（int）：老师模型的特征图通道数
    temp（float，optional）：温度系数，默认0.5
    name（str）：该层的损失名称
    alpha_fgd（float，optional）：fg_loss的权重，默认0.001
    beta_fgd（float，optional）：bg_loss的权重，默认0.0005
    gamma_fgd（float，optional）：mask_loss的权重，默认0。001
    lambda_fgd（float，optional）：relation_loss的权重，默认0.000005
    '''

    def __init__(self,
                 student_channels,
                 teacher_channels,
                 temp=0.5,
                 alpha_fgd=0.001,
                 beta_fgd=0.0005,
                 gamma_fgd=0.001,
                 lambda_fgd=0.000005,
                 ):
        super(FeatureLoss, self).__init__()
        self.temp = temp
        self.alpha_fgd = alpha_fgd
        self.beta_fgd = beta_fgd
        self.gamma_fgd = gamma_fgd
        self.lambda_fgd = lambda_fgd

        # 如果学生和老师模型通道不同，进行矫正
        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None

        self.conv_mask_s = nn.Conv2d(teacher_channels, 1, kernel_size=1, bias=False)
        self.conv_mask_t = nn.Conv2d(teacher_channels, 1, kernel_size=1, bias=False)
        self.channel_add_conv_s = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels // 2, kernel_size=1, bias=False),
            nn.LayerNorm([teacher_channels // 2, 1, 1]),
            nn.ReLU(inplace=True),  # yapf: disable
            nn.Conv2d(teacher_channels // 2, teacher_channels, kernel_size=1, bias=False)
        )
        self.channel_add_conv_t = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels // 2, kernel_size=1, bias=False),
            nn.LayerNorm([teacher_channels // 2, 1, 1]),
            nn.ReLU(inplace=True),  # yapf: disable
            nn.Conv2d(teacher_channels // 2, teacher_channels, kernel_size=1, bias=False)
        )

        self.reset_parameters(self)

    def forward(self, pred_s, pred_t, gt_box):
        '''
        preds_S(Tensor): Bs*C*H*W, student's feature map
        preds_T(Tensor): Bs*C*H*W, teacher's feature map
        gt_box(tuple): Bs*[nt*4], pixel decimal: (x, y, w, h)
        '''

        gt_bboxes = []
        batch_ind = gt_box[:, 0].unique(sorted=True)
        for ind in batch_ind:
            gt_bboxes.append(gt_box[gt_box[:, 0] == ind, 2:])

        assert pred_s.shape[-2:] == pred_t.shape[-2:], 'the output dim of teacher and student differ'

        if self.align is not None:
            pred_s = self.align(pred_s)

        preds_s = pred_s[batch_ind.long(), ...]
        preds_t = pred_t[batch_ind.long(), ...]


        N, C, H, W = preds_s.shape

        S_attention_t, C_attention_t = self.get_attention(preds_t, self.temp)
        S_attention_s, C_attention_s = self.get_attention(preds_s, self.temp)

        Mask_fg = torch.zeros_like(S_attention_s)
        Mask_bg = torch.ones_like(S_attention_t)
        wmin, wmax, hmin, hmax = [], [], [], []
        for i in range(N):
            new_boxxes = torch.ones_like(gt_bboxes[i])

            # 将gt_bboxes转换到特征图大小
            new_boxxes[:, 0] = (gt_bboxes[i][:, 0] - gt_bboxes[i][:, 2] / 2) * W
            new_boxxes[:, 2] = (gt_bboxes[i][:, 2] + gt_bboxes[i][:, 2] / 2) * W
            new_boxxes[:, 1] = (gt_bboxes[i][:, 1] - gt_bboxes[i][:, 3] / 2) * H
            new_boxxes[:, 3] = (gt_bboxes[i][:, 3] + gt_bboxes[i][:, 3] / 2) * H

            # 变成整数
            wmin.append(torch.floor(new_boxxes[:, 0]).int()) # 向下取整
            wmax.append(torch.ceil(new_boxxes[:, 2]).int())  # 向上取整
            hmin.append(torch.floor(new_boxxes[:, 1]).int())
            hmax.append(torch.ceil(new_boxxes[:, 3]).int())

            area = 1.0 / (hmax[i].view(1, -1) + 1 - hmin[i].view(1, -1)) / (
                        wmax[i].view(1, -1) + 1 - wmin[i].view(1, -1))

            for j in range(len(gt_bboxes[i])):
                Mask_fg[i][hmin[i][j]:hmax[i][j] + 1, wmin[i][j]:wmax[i][j] + 1] = \
                    torch.maximum(Mask_fg[i][hmin[i][j]:hmax[i][j] + 1, wmin[i][j]:wmax[i][j] + 1], area[0][j])

            Mask_bg[i] = torch.where(Mask_fg[i] > 0, 0, 1)
            if torch.sum(Mask_bg[i]):
                Mask_bg[i] /= torch.sum(Mask_bg[i])

        fg_loss, bg_loss = self.get_fea_loss(preds_s, preds_t, Mask_fg, Mask_bg,
                                             C_attention_s, C_attention_t, S_attention_s, S_attention_t)

        mask_loss = self.get_mask_loss(C_attention_s, C_attention_t, S_attention_s, S_attention_t)
        rela_loss = self.get_rela_loss(preds_s, preds_t)

        loss = self.alpha_fgd * fg_loss + self.beta_fgd * bg_loss \
               + self.gamma_fgd * mask_loss + self.lambda_fgd * rela_loss

        return loss

    def get_mask_loss(self, C_s, C_t, S_s, S_t):
        mask_loss = torch.sum(torch.abs((C_s - C_t))) / len(C_s) + torch.sum(torch.abs((S_s - S_t))) / len(S_s)

        return mask_loss

    def spatial_pool(self, x, in_type):
        batch, channel, width, height = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        if in_type == 0:
            context_mask = self.conv_mask_s(x)
        else:
            context_mask = self.conv_mask_t(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = F.softmax(context_mask, dim=2)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(-1)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context

    def get_rela_loss(self, preds_s, preds_t):
        loss_mse = nn.MSELoss(reduction='sum')

        context_s = self.spatial_pool(preds_s, 0)
        context_t = self.spatial_pool(preds_t, 1)

        out_s = preds_s
        out_t = preds_t

        channel_add_s = self.channel_add_conv_s(context_s)
        out_s = out_s + channel_add_s

        channel_add_t = self.channel_add_conv_t(context_t)
        out_t = out_t + channel_add_t

        rela_loss = loss_mse(out_s, out_t) / len(out_s)

        return rela_loss

    def get_fea_loss(self, preds_s, preds_t, Mask_fg, Mask_bg, C_s, C_t, S_s, S_t):
        loss_mse = nn.MSELoss(reduction='sum')

        Mask_fg = Mask_fg.unsqueeze(dim=1)
        Mask_bg = Mask_bg.unsqueeze(dim=1)

        C_t = C_t.unsqueeze(dim=-1)
        C_t = C_t.unsqueeze(dim=-1)

        S_t = S_t.unsqueeze(dim=1)

        fea_t = torch.mul(preds_t, torch.sqrt(S_t))
        fea_t = torch.mul(fea_t, torch.sqrt(C_t))
        fg_fea_t = torch.mul(fea_t, torch.sqrt(Mask_fg))
        bg_fea_t = torch.mul(fea_t, torch.sqrt(Mask_bg))

        fea_s = torch.mul(preds_s, torch.sqrt(S_t))
        fea_s = torch.mul(fea_s, torch.sqrt(C_t))
        fg_fea_s = torch.mul(fea_s, torch.sqrt(Mask_fg))
        bg_fea_s = torch.mul(fea_s, torch.sqrt(Mask_bg))

        fg_loss = loss_mse(fg_fea_s, fg_fea_t) / len(Mask_fg)
        bg_loss = loss_mse(bg_fea_s, bg_fea_t) / len(Mask_bg)

        return fg_loss, bg_loss

    def get_attention(self, preds, temp):
        '''
        preds:B*C*H*W
        '''

        N, C, H, W = preds.shape

        # B*H*W
        value = torch.abs(preds)
        fea_map = value.mean(dim=1, keepdim=True)
        s_attention = (H * W * F.softmax((fea_map / temp).view(N, -1), dim=1)).view(N, H, W)

        # B*C
        channel_map = value.mean(dim=2, keepdim=False).mean(dim=2, keepdim=False)
        c_attention = C * F.softmax(channel_map / temp, dim=1)

        return s_attention, c_attention

    def last_zero_init(self, m):
        if isinstance(m, nn.Sequential):
            nn.init.constant_(m[-1], val=0)
        else:
            nn.init.constant_(m, val=0)

    def reset_parameters(self, model):
        for m in model.modules():
            t = type(m)
            if t is nn.Conv2d:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.03
            elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                m.inplace = True
            elif t is nn.Linear:
                nn.init.uniform_(m.weight, a=-0.1, b=0.1)
                m.bias.data.fill_(0.01)


