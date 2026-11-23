import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.nn.functional as F

def _gather_training_batch(logits, labels):
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return (logits, labels)
    world_size = dist.get_world_size()
    count = torch.tensor([logits.numel()], device=logits.device, dtype=torch.long)
    counts = [torch.zeros_like(count) for _ in range(world_size)]
    dist.all_gather(counts, count)
    sizes = [int(value.item()) for value in counts]
    width = max(sizes)
    if width == 0:
        return (logits, labels)
    padded_logits = F.pad(logits, (0, width - logits.numel())).contiguous()
    padded_labels = F.pad(labels, (0, width - labels.numel())).contiguous()
    gathered_logits = dist_nn.all_gather(padded_logits)
    gathered_labels = [torch.empty_like(padded_labels) for _ in range(world_size)]
    dist.all_gather(gathered_labels, padded_labels)
    return (torch.cat([value[:size] for value, size in zip(gathered_logits, sizes)]), torch.cat([value[:size] for value, size in zip(gathered_labels, sizes)]))

def supervised_risk_ranking_loss(logits, labels, margin=0.1, distributed=True):
    if margin < 0:
        raise ValueError('Ranking margin must be nonnegative')
    logits = logits.reshape(-1)
    labels = labels.detach().to(device=logits.device).reshape(-1)
    if logits.numel() != labels.numel():
        raise ValueError('Logits and labels must have equal sample counts')
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError('Only original binary labels 1=resistant, 0=sensitive are accepted')
    if distributed:
        logits, labels = _gather_training_batch(logits, labels)
    positive = logits[labels == 1]
    negative = logits[labels == 0]
    if positive.numel() == 0 or negative.numel() == 0:
        return logits.sum() * 0.0
    difference = positive[:, None] - negative[None, :]
    return F.softplus(float(margin) - difference).mean()
