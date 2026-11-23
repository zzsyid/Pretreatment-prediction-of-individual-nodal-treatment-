import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score
import torch

from common import RUN, bootstrap, metrics, rows, select_threshold, sha, validate, write
from data import get_node_dataloader
from objectives import apply_input_mode, disable_inplace, extract_batch, mask_images_by_present, make_student_present_dict
from teacher import MultiModalLymphNodeNet as Teacher
from student import MultiModalLymphNodeNet as Student


@torch.no_grad()
def collect(manifest, checkpoint, kind, workers=2, device=None):
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    cls = Teacher if kind == 'teacher' else Student
    model = disable_inplace(cls(num_regions=15, feature_dim=128)).to(device)
    state = torch.load(str(checkpoint), map_location=device)
    if 'model' in state and isinstance(state['model'], dict):
        state = state['model']
    model.load_state_dict({k[7:] if k.startswith('module.') else k: v for k, v in state.items()}, strict=True)
    del state
    model.eval()
    model.enable_aux_outputs = kind == 'student'
    if kind == 'student':
        model.pet_effect_mode, model.pet_effect_context_mode, model.pet_effect_scale = 'single_gate', 'full', 1.
    loader, _ = get_node_dataloader(str(manifest), batch_size=2, is_train=False, use_ddp=False, stage=2,
        loader_workers=workers, cache_workers=1, cache_rate=0., persistent_workers=False,
        disable_random_aug=True, balanced_sampler=False, physical_preprocessing=True)
    records = []
    for batch in loader:
        mri, ct, mp, cp, volume, region, label = extract_batch(batch, device)
        if kind == 'student':
            mri, mp, ct, cp = apply_input_mode(mri, mp, ct, cp, 'limited')
            mp, cp = make_student_present_dict(mp, cp, hide_prob_t1c=1., hide_prob_pet=1.)
            mri, ct = mask_images_by_present(mri, ct, mp, cp)
            assert torch.count_nonzero(mri[:, 1]).item() == torch.count_nonzero(ct[:, 1]).item() == 0
        logits, _, _ = model(mri, mp, ct, cp, volume, region)
        probabilities = logits.sigmoid().view(-1).cpu().numpy()
        for i, probability in enumerate(probabilities):
            record = {'id': str(batch['id'][i]), 'patient_id': str(batch['patient_id'][i]),
                      'label': int(label[i].item()), 'prob': float(probability)}
            if kind == 'student':
                record.update(base_logit=float(model.latest_aux['base_logits'][i].item()),
                              effect_logit=float(model.latest_aux['pet_effect_logit'][i].item()))
            records.append(record)
    return pd.DataFrame(records)


def scaled(frame, scale):
    return expit(frame.base_logit.to_numpy(dtype=float) + scale * frame.effect_logit.to_numpy(dtype=float))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    _, digest = validate()
    teacher = RUN / 'checkpoints/teacher_adapt/best.pth'
    student = RUN / 'checkpoints/student_final/best.pth'
    if not all((path.parent / 'training_complete.json').is_file() for path in [teacher, student]):
        raise ValueError('Training must finish before selection and holdout evaluation')
    val = collect(RUN / 'manifests/val.json', student, 'student', args.workers)
    scales = np.arange(0., 1. + .025, .05)
    aucs = np.array([roc_auc_score(val.label, scaled(val, value)) for value in scales])
    tied = np.flatnonzero(np.isclose(aucs, aucs.max(), rtol=0., atol=1e-12))
    scale = float(scales[tied[np.argmin(np.abs(scales[tied] - 1.))]])
    student_threshold = select_threshold(val.label.to_numpy(), scaled(val, scale))
    teacher_val = collect(RUN / 'manifests/val.json', teacher, 'teacher', args.workers)
    teacher_threshold = select_threshold(teacher_val.label.to_numpy(), teacher_val.prob.to_numpy())
    selection = {'protocol_sha256': digest, 'student_scale': scale, 'student_threshold': student_threshold,
        'teacher_threshold': teacher_threshold, 'student_checkpoint_sha256': sha(student),
        'teacher_checkpoint_sha256': sha(teacher), 'validation_auc': float(aucs.max()),
        'test_used_for_selection': False}
    locked = RUN / 'selection.json'
    if locked.exists() and json.loads(locked.read_text()) != selection:
        raise ValueError('Existing validation selection changed')
    write(locked, selection)
    pd.DataFrame({'scale': scales, 'validation_auc': aucs}).to_csv(RUN / 'metrics/validation_scale_curve.csv', index=False)
    val['prob'] = scaled(val, scale)
    val.to_csv(RUN / 'predictions/student_validation.csv', index=False)
    teacher_val.to_csv(RUN / 'predictions/teacher_validation.csv', index=False)
    test = collect(RUN / 'manifests/test.json', student, 'student', args.workers)
    test['prob'] = scaled(test, scale)
    test['threshold'] = student_threshold
    test['pred_threshold'] = (test.prob >= student_threshold).astype(int)
    teacher_test = collect(RUN / 'manifests/test.json', teacher, 'teacher', args.workers)
    teacher_test['threshold'] = teacher_threshold
    teacher_test['pred_threshold'] = (teacher_test.prob >= teacher_threshold).astype(int)
    assert list(test.id) == list(teacher_test.id)
    test.to_csv(RUN / 'predictions/student_test.csv', index=False)
    teacher_test.to_csv(RUN / 'predictions/teacher_test.csv', index=False)
    performance, differences = bootstrap(test.label.to_numpy(), test.patient_id.to_numpy(),
        {'student': test.prob.to_numpy(), 'teacher': teacher_test.prob.to_numpy()},
        {'student': student_threshold, 'teacher': teacher_threshold}, 'student')
    pd.DataFrame(performance).to_csv(RUN / 'metrics/test_metrics.csv', index=False)
    pd.DataFrame(differences).to_csv(RUN / 'metrics/paired_auc_difference.csv', index=False)
    write(RUN / 'summary.json', {'state': 'complete', 'selection': selection, 'performance': performance,
          'paired_auc_differences': differences})


if __name__ == '__main__':
    main()
