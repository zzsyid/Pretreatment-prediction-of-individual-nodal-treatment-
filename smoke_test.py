#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description='Small synthetic checks; no clinical data or model training')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
    import numpy as np
    import torch
    from common import bootstrap
    from objectives import disable_inplace
    from ranking import supervised_risk_ranking_loss
    from teacher import MultiModalLymphNodeNet as Teacher
    from student import MultiModalLymphNodeNet as Student
    torch.set_num_threads(2)
    torch.manual_seed(3407)
    device = torch.device(args.device)
    mri, ct = torch.randn(2, 3, 16, 16, 16, device=device), torch.randn(2, 2, 16, 16, 16, device=device)
    mp = {name: torch.ones(2, 1, device=device) for name in ['t1', 't1c', 't2']}
    cp = {name: torch.ones(2, 1, device=device) for name in ['ct', 'pet']}
    volume, regions = torch.ones(2, 1, device=device), torch.ones(2, 1, dtype=torch.long, device=device)
    teacher = disable_inplace(Teacher(num_regions=15)).to(device).eval()
    with torch.no_grad():
        output = teacher(mri, mp, ct, cp, volume, regions)[0]
    assert output.shape == (2, 1) and torch.isfinite(output).all()
    del teacher
    student = disable_inplace(Student(num_regions=15)).to(device).eval()
    student.enable_aux_outputs = True
    student.pet_effect_mode = 'single_gate'
    mri[:, 1] = 0; ct[:, 1] = 0; mp['t1c'].zero_(); cp['pet'].zero_()
    with torch.no_grad():
        output = student(mri, mp, ct, cp, volume, regions)[0]
    assert output.shape == (2, 1) and torch.isfinite(output).all()
    assert torch.allclose(output, student.latest_aux['base_logits'] + student.latest_aux['pet_effect_logit'], atol=1e-6)
    risk = torch.tensor([2., -2.], requires_grad=True)
    ranking = supervised_risk_ranking_loss(risk, torch.tensor([1., 0.]), distributed=False)
    ranking.backward()
    assert risk.grad[0] < 0 and risk.grad[1] > 0
    y, p = np.tile([0, 1], 8), np.tile([.1, .9], 8)
    performance, difference = bootstrap(y, np.repeat(np.arange(8).astype(str), 2),
        {'reference': p, 'identical': p}, {'reference': .5, 'identical': .5}, 'reference', iterations=30)
    assert all(row['auc'] == 1 for row in performance)
    assert difference[0]['paired_ci95_low'] == difference[0]['paired_ci95_high'] == 0
    print('PASS: teacher/student forward, hidden privileged inputs, residual logits, ranking gradient, paired patient bootstrap')


if __name__ == '__main__':
    main()
