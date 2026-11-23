import numpy as np
import torch
import torch.nn.functional as F
from monai.data import MetaTensor
from monai.transforms import MapTransform, RandomizableTransform, SpatialCrop
FAMILIES = [('MRI_mask', ('T1', 'T2', 'T1CE', 'MRI_mask'), (5.0, 7.5)), ('CT_mask', ('CT', 'PET', 'CT_mask'), (1.0, 3.5))]

def spacing_from_tensor(value):
    if not isinstance(value, MetaTensor):
        raise TypeError('Native physical transforms require the saved image affine')
    spacing = torch.linalg.vector_norm(value.affine[:3, :3], dim=0).detach().cpu().numpy()
    if not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError('Invalid native spacing')
    return spacing

def native_slice_axis(value):
    spacing = spacing_from_tensor(value)
    directions = value.affine[:3, :3].detach().cpu().numpy() / spacing[None, :]
    tied = np.flatnonzero(np.isclose(spacing, spacing.max(), rtol=0.001, atol=1e-05))
    return int(tied[np.argmax(np.abs(directions[2, tied]))])

class PhysicalMaskBoundingBoxd(MapTransform):

    def __init__(self, halo_mm=5.0):
        super().__init__(keys=['MRI_mask', 'CT_mask'], allow_missing_keys=True)
        self.halo_mm = float(halo_mm)

    def __call__(self, data):
        result = dict(data)
        for mask_key, keys, _ in FAMILIES:
            if mask_key not in result:
                if any((key in result for key in keys if key != mask_key)):
                    raise ValueError('Observed family without native ROI mask')
                continue
            mask = result[mask_key]
            if mask.ndim != 4 or mask.shape[0] != 1:
                raise ValueError('Expected channel-first native mask')
            occupied = torch.nonzero(mask[0].as_tensor() > 0)
            if not occupied.numel():
                raise ValueError('Empty mask is not a valid observed nodal ROI')
            spacing = spacing_from_tensor(mask)
            pad = np.ceil(self.halo_mm / spacing).astype(int)
            low = occupied.min(dim=0).values.cpu().numpy()
            high = occupied.max(dim=0).values.cpu().numpy() + 1
            size = np.asarray(mask.shape[1:])
            start, end = (np.maximum(low - pad, 0), np.minimum(high + pad, size))
            crop = SpatialCrop(roi_start=start.tolist(), roi_end=end.tolist())
            old_count = int((mask > 0).sum())
            for key in keys:
                if key not in result:
                    continue
                image = result[key]
                if image.shape != mask.shape or not torch.allclose(image.affine, mask.affine, atol=0.0001, rtol=0):
                    raise ValueError('Native within-family image/mask grid mismatch: ' + key)
                result[key] = crop(image)
            if int((result[mask_key] > 0).sum()) != old_count:
                raise RuntimeError('Physical crop discarded mask foreground')
        return result

class NativeSliceResolutiond(RandomizableTransform, MapTransform):

    def __init__(self, probability=0.5):
        MapTransform.__init__(self, keys=['T1', 'T2', 'T1CE', 'CT', 'PET'], allow_missing_keys=True)
        RandomizableTransform.__init__(self, prob=1.0)
        self.family_probability = float(probability)
        self.targets = {}

    def randomize(self, data=None):
        self.targets = {mask: float(self.R.uniform(*limits)) if self.R.rand() < self.family_probability else None for mask, _, limits in FAMILIES}

    def __call__(self, data):
        self.randomize()
        result = dict(data)
        for mask_key, keys, _ in FAMILIES:
            target = self.targets[mask_key]
            if target is None or mask_key not in result:
                continue
            mask = result[mask_key]
            spacing = spacing_from_tensor(mask)
            axis = native_slice_axis(mask)
            if target <= spacing[axis]:
                continue
            shape = list(mask.shape[1:])
            pooled_shape = list(shape)
            pooled_shape[axis] = min(shape[axis], max(2, int(round(shape[axis] * spacing[axis] / target))))
            if pooled_shape == shape:
                continue
            for key in keys:
                if key == mask_key or key not in result:
                    continue
                image = result[key]
                pooled = F.adaptive_avg_pool3d(image.as_tensor().float().unsqueeze(0), pooled_shape)
                restored = F.interpolate(pooled, size=shape, mode='trilinear', align_corners=True)[0]
                result[key] = MetaTensor(restored, affine=image.affine.clone(), meta=dict(image.meta))
        return result
