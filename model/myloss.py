import torch
import torch.nn.functional as F


def bce_loss(pred, gt, pos_weight=None):
    if pos_weight is not None and not torch.is_tensor(pos_weight):
        pos_weight = torch.tensor([pos_weight], device=pred.device, dtype=pred.dtype) if pos_weight > 0 else None
    return F.binary_cross_entropy_with_logits(pred, gt, pos_weight=pos_weight)


def dice_loss(pred, gt, eps=1e-6):
    prob = torch.sigmoid(pred)
    dims = (0, 2, 3)
    intersection = (prob * gt).sum(dims)
    union = prob.sum(dims) + gt.sum(dims)
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def total_loss(pred, gt, pos_weight=None, eps=1e-6):
    bce = bce_loss(pred, gt, pos_weight=pos_weight)
    dice = dice_loss(pred, gt, eps=eps)
    return bce + dice, bce, dice
