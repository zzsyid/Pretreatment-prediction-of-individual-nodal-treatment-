#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Generate entirely synthetic NIfTI data for execution checks')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    import nibabel as nib
    import numpy as np
    target = Path(args.output).resolve()
    target.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3407)
    axis = np.indices((20, 20, 20)).astype(float)
    mask = (((axis - 10.) ** 2).sum(axis=0) < 20).astype(np.uint8)
    affine = np.diag([1., 1., 2., 1.])
    datasets = {'train': [], 'val': [], 'test': []}
    for i in range(20):
        folder = target / ('synthetic_subject_' + str(i).zfill(2))
        folder.mkdir(exist_ok=True)
        row = {'id': folder.name + '_node_A', 'patient_id': folder.name, 'node_id': 0,
               'region': 1, 'label': i % 2, 'mri_physical_volume': float(mask.sum() * 2),
               'ct_physical_volume': float(mask.sum() * 2)}
        for key in ['T1', 'T2', 'T1CE', 'CT', 'PET', 'MRI_mask', 'CT_mask']:
            value = mask if key.endswith('mask') else rng.random((20, 20, 20), dtype=np.float32)
            path = folder / (key + '.nii.gz')
            nib.save(nib.Nifti1Image(value, affine), str(path))
            row[key] = str(path.relative_to(target))
        split = 'train' if i < 14 else 'val' if i < 16 else 'test'
        datasets[split].append(row)
    for split, subset in datasets.items():
        (target / (split + '.json')).write_text(json.dumps(subset, indent=2) + '\n')
    print('Synthetic execution-check data:', target)


if __name__ == '__main__':
    main()
