#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description='Classic T1/T2/CT baselines on a fixed patient-disjoint split')
    for split in ['train', 'val', 'test']:
        parser.add_argument('--' + split + '-json', required=True)
    parser.add_argument('--data-root')
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--models', nargs='+', choices=['resnet18', 'densenet121', 'radiomics'],
                        default=['resnet18', 'densenet121', 'radiomics'])
    args = parser.parse_args()
    os.environ.update(ANON_RUN_DIR=str(Path(args.output).resolve()), ANON_SEED=str(args.seed),
                      CUDA_VISIBLE_DEVICES=args.gpu, PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg')
    sys.path.insert(0, str(ROOT / 'src'))
    from common import RUN, write
    from io_utils import setup
    setup(args, baseline=True)
    commands = []
    for name in args.models:
        if name == 'radiomics':
            base = [sys.executable, '-B', str(ROOT / 'src/baseline_radiomics.py')]
            if not (RUN / 'features/feature_names.json').exists():
                subprocess.run(base + ['--mode', 'preflight'], check=True)
        else:
            base = [sys.executable, '-B', str(ROOT / 'src/baseline_cnn.py'), '--arch', name]
        key = 'radiomics_LR' if name == 'radiomics' else name
        marker = RUN / 'checkpoints' / key / 'training_complete.json'
        if not marker.exists():
            with (RUN / 'logs' / (key + '.log')).open('a') as stream:
                subprocess.run(base + ['--mode', 'train'], check=True, stdout=stream, stderr=subprocess.STDOUT)
        commands.append((key, base))
    write(RUN / 'all_training_complete.json', {'models': [key for key, _ in commands],
                                             'all_choices_frozen_before_classic_test': True})
    for key, base in commands:
        if not (RUN / 'metrics' / (key + '_test.json')).exists():
            subprocess.run(base + ['--mode', 'test'], check=True)
    write(RUN / 'status.json', {'state': 'complete', 'models': args.models})
    print('Baseline outputs:', RUN, flush=True)


if __name__ == '__main__':
    main()
