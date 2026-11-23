import json
from pathlib import Path

from common import RUN, ROOT, rows, sha, write

IMAGE_KEYS = ('T1', 'T2', 'T1CE', 'CT', 'PET', 'MRI_mask', 'CT_mask')


def normalise_manifest(path, data_root=None, require_teacher=False, labelled=True, baseline=False):
    path = Path(path).resolve()
    root = Path(data_root).resolve() if data_root else path.parent
    result = json.loads(path.read_text())
    if not isinstance(result, list) or not result:
        raise ValueError('A manifest must be a nonempty JSON list')
    seen = set()
    for row in result:
        row['id'], row['patient_id'] = str(row['id']), str(row['patient_id'])
        if row['id'] in seen:
            raise ValueError('Duplicate node ID')
        seen.add(row['id'])
        if labelled and row.get('label') not in [0, 1]:
            raise ValueError('Use label0=sensitive and label1=resistant')
        row.setdefault('label', 0)
        row.setdefault('region', 0)
        row.setdefault('mri_physical_volume', 0.)
        row.setdefault('ct_physical_volume', 0.)
        for key in IMAGE_KEYS:
            if baseline and key in ['PET', 'T1CE']:
                row.pop(key, None)
                continue
            value = row.get(key)
            if value:
                image = Path(value)
                image = image if image.is_absolute() else root / image
                if not image.is_file():
                    raise FileNotFoundError(str(image))
                row[key] = str(image.resolve())
            else:
                row[key] = None
        required = IMAGE_KEYS if require_teacher else ('T1', 'T2', 'CT', 'MRI_mask', 'CT_mask')
        if any(not row.get(key) for key in required):
            raise ValueError('Required image or mask missing: ' + ','.join(required))
    return result


def setup(args, baseline=False):
    RUN.mkdir(parents=True, exist_ok=True)
    for name in ['manifests', 'checkpoints', 'logs', 'predictions', 'metrics', 'audit', 'features']:
        (RUN / name).mkdir(exist_ok=True)
    cohorts = {key: normalise_manifest(getattr(args, key + '_json'), args.data_root,
               require_teacher=not baseline, baseline=baseline) for key in ['train', 'val', 'test']}
    for subset in cohorts.values():
        if {r['label'] for r in subset} != {0, 1}:
            raise ValueError('Each split must contain both labels for AUC-based evaluation')
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        x, y = cohorts[a], cohorts[b]
        if ({r['patient_id'] for r in x} & {r['patient_id'] for r in y} or
            {r['id'] for r in x} & {r['id'] for r in y} or
            {r[k] for r in x for k in IMAGE_KEYS if r.get(k)} & {r[k] for r in y for k in IMAGE_KEYS if r.get(k)}):
            raise ValueError('Train/validation/test patients, nodes and image paths must be disjoint')
    previous = RUN / 'protocol.json'
    if previous.exists():
        old = json.loads(previous.read_text())
        if old['seed'] != args.seed or old['kind'] != ('baselines' if baseline else 'method'):
            raise ValueError('Existing output has incompatible settings; use a new output folder')
        if (old.get('max_epochs_override') != getattr(args, 'max_epochs', None) or
            old.get('batch_size_override') != getattr(args, 'batch_size', None)):
            raise ValueError('Resume must use the same epoch and batch-size settings')
        if any(rows(key) != value for key, value in cohorts.items()):
            raise ValueError('Existing output is bound to different manifests')
        return old
    report = {'splits': {}}
    for key, subset in cohorts.items():
        path = RUN / 'manifests' / (key + '.json')
        write(path, subset)
        report['splits'][key] = {'path': str(path), 'sha256': sha(path), 'nodes': len(subset),
                                'patients': len({r['patient_id'] for r in subset})}
    write(RUN / 'split_report.json', report)
    protocol = {'state': 'sealed', 'kind': 'baselines' if baseline else 'method', 'seed': args.seed,
        'workers': args.workers, 'manifest_sha256': {key: sha(RUN / 'manifests' / (key + '.json')) for key in cohorts},
        'max_epochs_override': getattr(args, 'max_epochs', None), 'batch_size_override': getattr(args, 'batch_size', None),
        'recipes_sha256': sha(ROOT / 'recipes.json'), 'patience': 5,
        'label_encoding': '0=treatment-sensitive;1=treatment-resistant',
        'selection': 'Validation AUC for weights/regularisation and validation F1 for thresholds',
        'test_selection': False}
    write(previous, protocol)
    return protocol
