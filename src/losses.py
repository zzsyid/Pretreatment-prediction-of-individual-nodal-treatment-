import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_alignment_loss(real_token, hallucinated_token):
    target = real_token.detach()
    cos_sim = F.cosine_similarity(hallucinated_token, target, dim=-1)
    return (1.0 - cos_sim).mean()

class BinaryFocalLoss(nn.Module):

    def __init__(self, pos_weight=None, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        if pos_weight is not None:
            self.register_buffer('pos_weight', pos_weight.clone().detach().float())
        else:
            self.pos_weight = None

    def forward(self, logits, targets):
        targets = targets.float()
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none', pos_weight=self.pos_weight)
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_weight = (1.0 - p_t).pow(self.gamma)
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            focal_weight = focal_weight * alpha_t
        loss = focal_weight * bce_loss
        if self.reduction == 'sum':
            return loss.sum()
        if self.reduction == 'none':
            return loss
        return loss.mean()

class AsymmetricBinaryLoss(nn.Module):

    def __init__(self, gamma_pos=1.0, gamma_neg=4.0, clip=0.05, eps=1e-08, reduction='mean'):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits, targets):
        targets = targets.float()
        probs = torch.sigmoid(logits)
        xs_pos = probs
        xs_neg = 1.0 - probs
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)
        loss_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        loss_neg = (1.0 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = loss_pos + loss_neg
        if self.gamma_pos > 0 or self.gamma_neg > 0:
            pt = xs_pos * targets + xs_neg * (1.0 - targets)
            gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            one_sided_weight = (1.0 - pt).pow(gamma)
            loss = loss * one_sided_weight
        loss = -loss
        if self.reduction == 'sum':
            return loss.sum()
        if self.reduction == 'none':
            return loss
        return loss.mean()

def build_classification_loss(loss_name, pos_weight, focal_gamma=2.0, focal_alpha=None, asl_gamma_pos=1.0, asl_gamma_neg=4.0, asl_clip=0.05):
    if loss_name == 'focal':
        return BinaryFocalLoss(pos_weight=pos_weight, gamma=focal_gamma, alpha=focal_alpha)
    if loss_name == 'asl':
        return AsymmetricBinaryLoss(gamma_pos=asl_gamma_pos, gamma_neg=asl_gamma_neg, clip=asl_clip)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
