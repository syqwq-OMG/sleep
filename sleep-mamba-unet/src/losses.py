from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_bce_with_logits(logits, targets, mask, pos_weight=None):
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_weight)
    loss = loss * mask
    return loss.sum() / mask.sum().clamp_min(1.0)


def masked_focal_bce_with_logits(logits, targets, mask, gamma=2.0, alpha=0.25):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    prob = torch.sigmoid(logits)
    pt = prob * targets + (1 - prob) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * (1 - pt).clamp_min(1e-6).pow(gamma) * bce
    loss = loss * mask
    return loss.sum() / mask.sum().clamp_min(1.0)


def boundary_consistency_loss(logits_onset, logits_wakeup, logits_sleep, mask):
    p_sleep = torch.sigmoid(logits_sleep)
    diff = torch.abs(p_sleep[:, 1:] - p_sleep[:, :-1])
    p_event = torch.maximum(torch.sigmoid(logits_onset[:, 1:]), torch.sigmoid(logits_wakeup[:, 1:]))
    m = mask[:, 1:]
    return (F.relu(diff - p_event) * m).sum() / m.sum().clamp_min(1.0)


def pairwise_rank_loss(logits_event, targets, mask, margin=0.2):
    pos = logits_event[targets > 0.5]
    neg = logits_event[(targets < 0.01) & (mask > 0)]
    if pos.numel() == 0 or neg.numel() == 0:
        return logits_event.sum() * 0
    pos = pos[:512]
    neg = neg[:512]
    return F.relu(margin - pos[:, None] + neg[None, :]).mean()


def compute_loss(outputs, targets, config):
    loss_cfg = config.get("loss", {})
    mask_event = targets["mask_event"]
    mask_sleep = targets["mask_sleep"]
    mask_valid = targets.get("mask_valid", torch.ones_like(mask_event))
    event_loss_name = str(loss_cfg.get("event_loss", "bce")).lower()
    if event_loss_name == "focal":
        gamma = float(loss_cfg.get("focal_gamma", 2.0))
        alpha = float(loss_cfg.get("focal_alpha", 0.25))
        event_loss = masked_focal_bce_with_logits(outputs["onset"], targets["y_onset"], mask_event, gamma, alpha)
        event_loss = event_loss + masked_focal_bce_with_logits(outputs["wakeup"], targets["y_wakeup"], mask_event, gamma, alpha)
    else:
        event_loss = masked_bce_with_logits(outputs["onset"], targets["y_onset"], mask_event)
        event_loss = event_loss + masked_bce_with_logits(outputs["wakeup"], targets["y_wakeup"], mask_event)
    sleep_loss = masked_bce_with_logits(outputs["sleep"], targets["y_sleep"], mask_sleep)
    boundary = boundary_consistency_loss(outputs["onset"], outputs["wakeup"], outputs["sleep"], mask_valid)
    rank_weight = float(loss_cfg.get("rank_weight", 0.0))
    rank = outputs["onset"].sum() * 0
    if rank_weight > 0:
        rank = pairwise_rank_loss(outputs["onset"], targets["y_onset"], mask_event)
        rank = rank + pairwise_rank_loss(outputs["wakeup"], targets["y_wakeup"], mask_event)
    return (
        float(loss_cfg.get("event_weight", 1.0)) * event_loss
        + float(loss_cfg.get("sleep_weight", 0.5)) * sleep_loss
        + float(loss_cfg.get("boundary_weight", 0.2)) * boundary
        + rank_weight * rank
    )
