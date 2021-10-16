import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import LOSSES, build_loss
from mmdet.models.losses.utils import weighted_loss


@weighted_loss
def dice_loss(input, target, eps=1e-3, numerator_eps=0):
    input = input.reshape(input.size()[0], -1)
    target = target.reshape(target.size()[0], -1).float()

    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + eps
    c = torch.sum(target * target, 1) + eps
    d = (2 * a + numerator_eps) / (b + c)
    return 1 - d


@LOSSES.register_module()
class DiceLoss(nn.Module):

    def __init__(self,
                 eps=1e-3,
                 numerator_eps=0.0,
                 use_sigmoid=True,
                 reduction='mean',
                 loss_weight=1.0):
        super(DiceLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.use_sigmoid = use_sigmoid
        self.numerator_eps = numerator_eps

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if weight is not None and not torch.any(weight > 0):
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        pred = pred.sigmoid()
        loss = self.loss_weight * dice_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            numerator_eps=self.numerator_eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss
