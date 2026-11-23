import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, precision_recall_curve, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
RUN = Path(os.environ.get('ANON_RUN_DIR', str(ROOT / 'runs' / 'example'))).resolve()
SEED = int(os.environ.get('ANON_SEED', '3407'))


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def rows(split):
    return json.loads((RUN / 'manifests' / (split + '.json')).read_text())


def validate():
    protocol = json.loads((RUN / 'protocol.json').read_text())
    assert protocol['state'] == 'sealed'
    for split, digest in protocol['manifest_sha256'].items():
        assert sha(RUN / 'manifests' / (split + '.json')) == digest
    return protocol, sha(RUN / 'protocol.json')


def select_threshold(y, probability):
    precision, recall, thresholds = precision_recall_curve(y, probability)
    score = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.argmax(score))]) if len(thresholds) else .5


def metrics(y, probability, threshold):
    y, probability = np.asarray(y, dtype=int), np.asarray(probability, dtype=float)
    assert np.isfinite(probability).all() and set(y) <= {0, 1}
    tn, fp, fn, tp = confusion_matrix(y, probability >= threshold, labels=[0, 1]).ravel()
    return {'auc': float(roc_auc_score(y, probability)), 'accuracy': float((tn + tp) / len(y)),
            'sensitivity': float(tp / max(tp + fn, 1)), 'specificity': float(tn / max(tn + fp, 1)),
            'precision': float(tp / max(tp + fp, 1)), 'npv': float(tn / max(tn + fn, 1)),
            'f1': float(2 * tp / max(2 * tp + fp + fn, 1)),
            'auprc': float(average_precision_score(y, probability)),
            'brier': float(brier_score_loss(y, probability)), 'threshold': float(threshold),
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}


def align(frame, subset):
    frame = frame.copy()
    frame['id'] = frame.id.astype(str)
    assert not frame.id.duplicated().any()
    assert set(frame.id) == {str(r['id']) for r in subset}
    frame = frame.set_index('id').loc[[str(r['id']) for r in subset]].reset_index()
    assert np.array_equal(frame.label.to_numpy(dtype=int), [int(r['label']) for r in subset])
    assert list(frame.patient_id.astype(str)) == [str(r['patient_id']) for r in subset]
    return frame


def bootstrap(y, patients, scores, thresholds, reference, iterations=2000):
    patients = np.asarray(patients, dtype=str)
    y = np.asarray(y, dtype=int)
    names = sorted(set(patients))
    groups = [np.flatnonzero(patients == name) for name in names]
    rng = np.random.default_rng(SEED)
    estimates = {name: [] for name in scores}
    for _ in range(iterations):
        ix = np.concatenate([groups[int(i)] for i in rng.integers(0, len(names), size=len(names))])
        if np.unique(y[ix]).size < 2:
            continue
        for name, probability in scores.items():
            estimates[name].append(float(roc_auc_score(y[ix], probability[ix])))
    if len(estimates[reference]) < 20:
        raise ValueError('Too few valid patient bootstrap resamples')
    performance, differences = [], []
    reference_auc = float(roc_auc_score(y, scores[reference]))
    for name, probability in scores.items():
        lo, hi = np.percentile(estimates[name], [2.5, 97.5])
        performance.append({'method': name, 'nodes': len(y), 'patients': len(names),
            'resistant_nodes': int(y.sum()), 'auc_ci95_low': float(lo), 'auc_ci95_high': float(hi),
            'valid_patient_bootstraps': len(estimates[name]), **metrics(y, probability, thresholds[name])})
        if name != reference:
            delta = np.array(estimates[reference]) - np.array(estimates[name])
            dlo, dhi = np.percentile(delta, [2.5, 97.5])
            differences.append({'reference': reference, 'comparator': name,
                'auc_difference': reference_auc - float(roc_auc_score(y, probability)),
                'paired_ci95_low': float(dlo), 'paired_ci95_high': float(dhi),
                'valid_patient_bootstraps': len(delta), 'multiplicity_adjusted': False})
    return performance, differences
