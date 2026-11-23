import torch
import torch.nn as nn
import torch.nn.functional as F

class MRIResponseSpatialNorm(nn.BatchNorm3d):

    def forward(self, x):
        return F.instance_norm(x, running_mean=None, running_var=None, weight=self.weight, bias=self.bias, use_input_stats=True, momentum=0.0, eps=self.eps)

def convert_response_norm(module):
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm3d):
            replacement = MRIResponseSpatialNorm(child.num_features, eps=child.eps, momentum=child.momentum, affine=child.affine, track_running_stats=True)
            replacement.load_state_dict(child.state_dict(), strict=True)
            replacement.train(child.training)
            setattr(module, name, replacement)
            count += 1
        else:
            count += convert_response_norm(child)
    return count

def student_mri_style(x, probability=0.8):
    size = (x.shape[0], x.shape[1], 1, 1, 1)
    active = torch.rand(size, device=x.device) < probability
    gain = torch.empty(size, device=x.device, dtype=x.dtype).uniform_(0.75, 1.25)
    shift = torch.empty(size, device=x.device, dtype=x.dtype).uniform_(-0.3, 0.3)
    gamma = torch.empty(size, device=x.device, dtype=x.dtype).uniform_(0.85, 1.15)
    nonzero = x != 0
    altered = x.sign() * x.abs().clamp_min(1e-08).pow(gamma)
    altered = (gain * altered + shift) * nonzero.to(x.dtype)
    return torch.where(active, altered, x)
