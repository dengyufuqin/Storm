import torch
import torch.nn.functional as F
from torch import nn

def dice_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    inputs = inputs.sigmoid()
    inputs_flat = inputs.flatten(1)
    targets_flat = targets.flatten(1).float()
    intersection = (inputs_flat * targets_flat).sum(1)
    numerator = 2 * intersection + 1.0
    denominator = inputs_flat.sum(1) + targets_flat.sum(1) + 1.0
    loss = 1.0 - (numerator / denominator)
    return loss.mean()

def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.1,   
    gamma: float = 2.0,
) -> torch.Tensor:
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * loss
    return loss.mean()

def iou_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    prob = inputs.sigmoid()
    pred = prob > 0.5
    gt   = targets > 0.5
    pred_flat = pred.flatten(1)
    gt_flat   = gt.flatten(1)
    intersection = (pred_flat & gt_flat).sum(1).float()
    union        = (pred_flat | gt_flat).sum(1).float()
    iou = (intersection + eps) / (union + eps)
    loss = 1.0 - iou
    return loss.mean()

class TverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.eps   = eps

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        prob_flat = inputs.sigmoid().flatten(1)
        tgt_flat  = targets.flatten(1).float()
        TP = (prob_flat * tgt_flat).sum(1)
        FN = ((1 - prob_flat) * tgt_flat).sum(1)
        FP = (prob_flat * (1 - tgt_flat)).sum(1)
        ti = (TP + self.eps) / (TP + self.alpha * FN + self.beta * FP + self.eps)
        return (1.0 - ti).mean()

class FbetaLoss(nn.Module):
    def __init__(self, beta: float = 0.5, eps: float = 1e-6):
        super().__init__()
        self.beta2 = beta ** 2
        self.eps   = eps

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        prob = inputs.sigmoid().flatten(1)
        tgt  = targets.flatten(1).float()
        TP = (prob * tgt).sum(1)
        FP = (prob * (1 - tgt)).sum(1)
        FN = ((1 - prob) * tgt).sum(1)
        num = (1 + self.beta2) * TP + self.eps
        den = (1 + self.beta2) * TP + self.beta2 * FN + FP + self.eps
        fbeta = num / den
        return (1 - fbeta).mean()

class CombinedLoss(nn.Module):
    def __init__(
        self,
        weight_fbeta:    float = 1.0,
        weight_tversky:  float = 1.0,
        weight_focal:    float = 0.2,
        weight_dice:     float = 0.0,
        weight_iou:      float = 0.0,
        focal_alpha:     float = 0.1,
        focal_gamma:     float = 2.0,
        tversky_alpha:   float = 0.3,
        tversky_beta:    float = 0.7,
        fbeta_beta:      float = 0.5,
    ):
        super().__init__()
        self.w_fbeta   = weight_fbeta
        self.w_tversky = weight_tversky
        self.w_focal   = weight_focal
        self.w_dice    = weight_dice
        self.w_iou     = weight_iou

        self.fbeta   = FbetaLoss(beta=fbeta_beta)
        self.tversky = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        self.focal   = lambda x, y: sigmoid_focal_loss(x, y, alpha=focal_alpha, gamma=focal_gamma)
        self.dice    = dice_loss
        self.iou     = iou_loss

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        if self.w_fbeta   != 0: loss += self.w_fbeta   * self.fbeta(logits, targets)
        if self.w_tversky != 0: loss += self.w_tversky * self.tversky(logits, targets)
        if self.w_focal   != 0: loss += self.w_focal   * self.focal(logits, targets)
        if self.w_dice    != 0: loss += self.w_dice    * self.dice(logits, targets)
        if self.w_iou     != 0: loss += self.w_iou     * self.iou(logits, targets)
        return loss