#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description='Predict with the final student using T1/T2/CT only')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--data-root')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--selection', required=True, help='Validation-derived selection.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--device', default=None, help='cpu or cuda; defaults to the available device')
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / 'src'))
    from common import sha
    from evaluation import collect, scaled
    from io_utils import normalise_manifest
    selection = json.loads(Path(args.selection).read_text())
    if sha(args.checkpoint) != selection['student_checkpoint_sha256']:
        raise ValueError('Checkpoint differs from the frozen validation selection')
    subset = normalise_manifest(args.manifest, args.data_root, labelled=False, baseline=True)
    # Privileged images are excluded even if the supplied manifest includes them.
    for row in subset:
        row['T1CE'] = row['PET'] = None
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / 'manifest.json'
        path.write_text(json.dumps(subset))
        frame = collect(path, args.checkpoint, 'student', args.workers, args.device)
    frame['prob'] = scaled(frame, selection['student_scale'])
    frame['threshold'] = selection['student_threshold']
    frame['pred_threshold'] = (frame.prob >= selection['student_threshold']).astype(int)
    frame = frame.drop(columns=['label', 'base_logit', 'effect_logit'])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(args.output)


if __name__ == '__main__':
    main()
