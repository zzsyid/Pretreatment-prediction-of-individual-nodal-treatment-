import json
import torch
import numpy as np
from monai.data import CacheDataset, DataLoader
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Resized, ScaleIntensityRangePercentilesd, ToTensord, MapTransform, Lambda, RandAffined, RandGaussianNoised
from torch.utils.data.distributed import DistributedSampler
from spatial import PhysicalMaskBoundingBoxd, NativeSliceResolutiond

class CleanMissingKeysd(MapTransform):

    def __call__(self, data):
        d = dict(data)
        keys_to_check = ['T2', 'T1', 'T1CE', 'CT', 'PET', 'MRI_mask', 'CT_mask']
        for key in keys_to_check:
            if key in d:
                if d[key] is None or (isinstance(d[key], str) and d[key].strip() == ''):
                    del d[key]
        return d

class PadMissingModalityd(MapTransform):

    def __init__(self, keys, spatial_size=(64, 64, 64)):
        super().__init__(keys)
        self.spatial_size = spatial_size

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key not in d or d[key] is None or str(d[key]).strip() == '':
                d[key] = torch.zeros((1, *self.spatial_size), dtype=torch.float32)
                d[f'{key}_present'] = torch.tensor([0.0], dtype=torch.float32)
            else:
                d[f'{key}_present'] = torch.tensor([1.0], dtype=torch.float32)
        return d

class ProcessMetadata(MapTransform):

    def __call__(self, data):
        d = dict(data)
        mri_vol = d.get('mri_physical_volume', 0.0)
        d['mri_physical_volume'] = torch.tensor([np.log1p(mri_vol)], dtype=torch.float32)
        ct_vol = d.get('ct_physical_volume', 0.0)
        d['ct_physical_volume'] = torch.tensor([np.log1p(ct_vol)], dtype=torch.float32)
        d['region'] = torch.tensor([d.get('region', 0)], dtype=torch.float32)
        d['label'] = torch.tensor([d.get('label')], dtype=torch.float32)
        return d

def get_node_dataloader(json_path, batch_size=16, spatial_size=(64, 64, 64), is_train=True, use_ddp=False, stage=2, extra_json_paths=None, extra_stage=2, extra_repeat=1, loader_workers=8, cache_workers=2, cache_rate=0.8, persistent_workers=True, disable_random_aug=None, balanced_sampler=True, physical_preprocessing=True):

    def _filter_by_stage(items, filter_stage):
        if filter_stage != 1:
            return items
        modalities = ['T1', 'T2', 'T1CE', 'CT', 'PET']
        return [item for item in items if all((item.get(m) is not None for m in modalities))]
    with open(json_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    if stage == 1:
        filtered_data_list = []
        modalities = ['T1', 'T2', 'T1CE', 'CT', 'PET']
        for item in data_list:
            if all((item.get(m) is not None for m in modalities)):
                filtered_data_list.append(item)
        data_list = filtered_data_list
        print(f'Stage 1: 过滤后剩余 {len(data_list)} 个模态齐全的样本用于预训练。')
    if extra_json_paths:
        if isinstance(extra_json_paths, str):
            extra_json_paths = [extra_json_paths]
        repeat = max(int(extra_repeat), 1)
        extra_data_list = []
        for extra_json_path in extra_json_paths:
            with open(extra_json_path, 'r', encoding='utf-8') as f:
                loaded_extra = json.load(f)
            loaded_extra = _filter_by_stage(loaded_extra, extra_stage)
            extra_data_list.extend(loaded_extra * repeat)
            print(f'Extra data: {extra_json_path} added {len(loaded_extra)} samples, repeat={repeat}')
        data_list = data_list + extra_data_list
        print(f'Combined dataset size: {len(data_list)}')
    image_keys = ['T2', 'T1', 'T1CE', 'CT', 'PET']
    mask_keys = ['MRI_mask', 'CT_mask']
    all_keys = image_keys + mask_keys
    transform_list = [CleanMissingKeysd(keys=['T2', 'T1', 'T1CE', 'CT', 'PET', 'MRI_mask', 'CT_mask']), LoadImaged(keys=all_keys, allow_missing_keys=True), EnsureChannelFirstd(keys=all_keys, allow_missing_keys=True), *([PhysicalMaskBoundingBoxd(halo_mm=5.0)] if physical_preprocessing else []), *([NativeSliceResolutiond(probability=0.5)] if physical_preprocessing and (is_train and (not bool(disable_random_aug))) else []), Lambda(lambda d: {k: v for k, v in d.items() if not (isinstance(v, torch.Tensor) and v.numel() == 0)}), Resized(keys=image_keys, spatial_size=spatial_size, mode='trilinear', align_corners=True, allow_missing_keys=True), Resized(keys=mask_keys, spatial_size=spatial_size, mode='nearest', allow_missing_keys=True)]
    if not bool(disable_random_aug):
        transform_list.extend([RandAffined(keys=image_keys + mask_keys, mode=('bilinear', 'bilinear', 'bilinear', 'bilinear', 'bilinear', 'nearest', 'nearest'), prob=0.8, rotate_range=(np.pi / 30, np.pi / 30, np.pi / 30), translate_range=(3, 3, 3), allow_missing_keys=True), RandGaussianNoised(keys=image_keys, prob=0.1, mean=0.0, std=0.05, allow_missing_keys=True)])
    transform_list.extend([ScaleIntensityRangePercentilesd(keys=['CT', 'PET'], lower=0.5, upper=99.9, b_min=0.0, b_max=1.0, clip=True, allow_missing_keys=True), PadMissingModalityd(keys=image_keys, spatial_size=spatial_size), PadMissingModalityd(keys=mask_keys, spatial_size=spatial_size), ProcessMetadata(keys=['mri_physical_volume']), ToTensord(keys=all_keys, allow_missing_keys=True)])
    transforms = Compose(transform_list)
    dataset = CacheDataset(data=data_list, transform=transforms, cache_rate=cache_rate, num_workers=cache_workers)
    sampler = None
    if is_train:
        labels = [item.get('label', 0) for item in data_list]
        neg_count = sum((1 for l in labels if l == 0))
        pos_count = sum((1 for l in labels if l == 1))
        print(f'�� 训练集分布 - 负样本(0): {neg_count}, 正样本(1): {pos_count}')
        weight_neg = 1.0 / neg_count if neg_count > 0 else 0
        weight_pos = 1.0 / pos_count if pos_count > 0 else 0
        sample_weights = [weight_neg if l == 0 else weight_pos for l in labels]
        if use_ddp:
            print('�� 启用 DistributedSampler 进行多卡均匀采样')
            sampler = DistributedSampler(dataset)
        elif balanced_sampler:
            print('�� 启用 WeightedRandomSampler 进行单卡平衡采样')
            sampler = torch.utils.data.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        else:
            print('启用随机打乱进行单卡均匀采样')
    elif use_ddp:
        sampler = DistributedSampler(dataset, shuffle=False)
    shuffle = bool(is_train and sampler is None)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=loader_workers, sampler=sampler, persistent_workers=persistent_workers if loader_workers > 0 else False, pin_memory=True)
    return (loader, sampler)
