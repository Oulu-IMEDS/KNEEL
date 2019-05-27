from torch import nn
import torch
from torch.nn import functional as F


class HGHMLoss(nn.Module):
    def __init__(self, hm_loss='mse', w=0.5):
        super(HGHMLoss, self).__init__()
        assert hm_loss in ['mse', 'bce']

        if hm_loss == 'mse':
            self.hm_loss = nn.MSELoss()
        else:
            self.hm_loss = nn.BCEWithLogitsLoss()

        self.weights = torch.FloatTensor([w, 1-w])

    def forward(self, preds, gt_hm):
        pred_hm_0, pred_hm_1 = preds
        return self.hm_loss(pred_hm_0, gt_hm).mul(self.w[0]) + self.hm_loss(pred_hm_1, gt_hm).mul(self.w[1])


class ElasticLoss(nn.Module):
    def __init__(self, w):
        super(ElasticLoss, self).__init__()
        self.weights = torch.FloatTensor([1, 1 - w])

    def forward(self, pred_kp, gt_kp):
        l2 = F.mse_loss(pred_kp, gt_kp).mul(self.weights[0])
        l1 = F.l1_loss(pred_kp, gt_kp).mul(self.weights[1])

        return l2 + l1
