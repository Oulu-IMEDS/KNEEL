from torch import nn
import torch
from torch.nn import functional as F


class ElasticLoss(nn.Module):
    def __init__(self, w=0.5):
        super(ElasticLoss, self).__init__()
        self.weights = torch.FloatTensor([w, 1 - w])

    def forward(self, preds, gt):
        loss = 0

        if not isinstance(preds, tuple):
            preds = (preds, )

        for i in range(len(preds)):
            l2 = F.mse_loss(preds[i].squeeze(), gt.squeeze()).mul(self.weights[0])
            l1 = F.l1_loss(preds[i].squeeze(), gt.squeeze()).mul(self.weights[1])
            loss += l1 + l2

        return loss


class HGHMLoss(nn.Module):
    def __init__(self, hm_loss='mse', w=0.5):
        super(HGHMLoss, self).__init__()
        assert hm_loss in ['mse', 'elastic']

        if hm_loss == 'mse':
            self.hm_loss = nn.MSELoss()
        else:
            self.hm_loss = ElasticLoss()

        self.weights = torch.FloatTensor([w, 1-w])

    def forward(self, preds, gt_hm):
        pred_hm_0, pred_hm_1 = preds
        return self.hm_loss(pred_hm_0, gt_hm).mul(self.weights[0]) + self.hm_loss(pred_hm_1, gt_hm).mul(self.weights[1])

