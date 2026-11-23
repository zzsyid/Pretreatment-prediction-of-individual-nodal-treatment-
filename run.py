#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description='Train the final V9 teacher/student method from scratch')
    for split in ['train', 'val', 'test']:
        parser.add_argument('--' + split + '-json', required=True)
    parser.add_argument('--data-root')
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--memory-fraction', type=float, default=.35)
    parser.add_argument('--max-epochs', type=int, help='Optional cap for a quick execution check')
    parser.add_argument('--batch-size', type=int, help='Optional override; normal batches are8 for teachers and12 for students')
    parser.add_argument('--train-only', action='store_true')
    args = parser.parse_args()
    os.environ.update(ANON_RUN_DIR=str(Path(args.output).resolve()), ANON_SEED=str(args.seed),
        ANON_MEMORY_FRACTION=str(args.memory_fraction), CUDA_VISIBLE_DEVICES=args.gpu,
        PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg')
    sys.path.insert(0, str(ROOT / 'src'))
    from common import RUN, rows, sha, write
    from io_utils import setup
    setup(args)
    train = rows('train')
    positive = sum(row['label'] for row in train)
    if positive == 0 or positive == len(train):
        raise ValueError('Training requires both labels')
    recipes = json.loads((ROOT / 'recipes.json').read_text())
    completed = []
    replacements = {'train': str(RUN / 'manifests/train.json'), 'val': str(RUN / 'manifests/val.json'),
        'test': str(RUN / 'manifests/test.json'), 'checkpoints': str(RUN / 'checkpoints'),
        'positive_weight': str((len(train) - positive) / positive), 'seed': str(args.seed), 'workers': str(args.workers)}
    for name, recipe in recipes.items():
        arguments = [item.format(**replacements) for item in recipe['args']]
        if args.max_epochs is not None:
            arguments[arguments.index('--epochs') + 1] = str(min(args.max_epochs, int(arguments[arguments.index('--epochs') + 1])))
        if args.batch_size is not None:
            arguments[arguments.index('--batch-size') + 1] = str(args.batch_size)
        marker = RUN / 'checkpoints' / name / 'training_complete.json'
        if not marker.exists():
            command = [sys.executable, '-B', str(ROOT / 'src/stage.py'), '--source',
                       'adaptation' if recipe['physical_preprocessing'] else 'warmup', '--stage-key', name] + arguments
            print('Training', name, flush=True)
            write(RUN / 'status.json', {'state': 'training', 'phase': name, 'completed': completed})
            with (RUN / 'logs' / (name + '.log')).open('a') as stream:
                subprocess.run(command, env=dict(os.environ, ANON_PROTOCOL_SHA256=sha(RUN / 'protocol.json')),
                               cwd=str(RUN), stdout=stream, stderr=subprocess.STDOUT, check=True)
        completed.append(name)
    if not args.train_only:
        print('Selecting validation settings and evaluating the holdout', flush=True)
        subprocess.run([sys.executable, '-B', str(ROOT / 'src/evaluation.py'), '--workers', str(args.workers)],
                       env=os.environ.copy(), check=True)
    write(RUN / 'status.json', {'state': 'training_complete' if args.train_only else 'complete', 'completed': completed})
    print('Outputs:', RUN, flush=True)


if __name__ == '__main__':
    main()
