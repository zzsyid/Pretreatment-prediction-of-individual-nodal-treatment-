#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description='Paired patient-cluster bootstrap comparison of fixed predictions')
    parser.add_argument('--predictions', nargs='+', required=True, help='Pairs such as method=student_test.csv')
    parser.add_argument('--reference', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--iterations', type=int, default=2000)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / 'src'))
    import numpy as np
    import pandas as pd
    from common import bootstrap
    scores, thresholds, reference = {}, {}, None
    for entry in args.predictions:
        name, path = entry.split('=', 1)
        frame = pd.read_csv(path, dtype={'id': str, 'patient_id': str}, float_precision='round_trip')
        if not {'id', 'patient_id', 'label', 'prob', 'threshold'} <= set(frame):
            raise ValueError('Predictions require id,patient_id,label,prob,threshold columns')
        if frame.id.duplicated().any() or frame.threshold.nunique() != 1:
            raise ValueError('IDs must be unique; a fixed validation-derived threshold is required')
        if reference is None:
            reference = frame
        else:
            if set(reference.id) != set(frame.id):
                raise ValueError('Evaluated node IDs differ between methods')
            frame = frame.set_index('id').loc[reference.id].reset_index()
            if not np.array_equal(frame.label.values, reference.label.values) or list(frame.patient_id) != list(reference.patient_id):
                raise ValueError('Labels or patient mapping differ between methods')
        scores[name] = frame.prob.to_numpy(dtype=float)
        thresholds[name] = float(frame.threshold.iloc[0])
    if args.reference not in scores:
        raise ValueError('Reference is absent from supplied predictions')
    result, delta = bootstrap(reference.label.to_numpy(dtype=int), reference.patient_id.to_numpy(),
                               scores, thresholds, args.reference, args.iterations)
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result).to_csv(target / 'performance.csv', index=False)
    pd.DataFrame(delta).to_csv(target / 'paired_auc_differences.csv', index=False)
    print(target)


if __name__ == '__main__':
    main()
