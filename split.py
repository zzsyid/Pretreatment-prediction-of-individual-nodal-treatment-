#!/usr/bin/env python3
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Patient-disjoint7:1:2 split, stratified by any resistant node')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--exclude-duplicates', action='store_true')
    args = parser.parse_args()
    from sklearn.model_selection import train_test_split
    source = json.loads(Path(args.manifest).read_text())
    counts = Counter(str(row['id']) for row in source)
    repeated = {node for node, count in counts.items() if count > 1}
    if repeated and not args.exclude_duplicates:
        raise ValueError('Repeated node IDs; review annotations or explicitly exclude all copies')
    source = [row for row in source if str(row['id']) not in repeated]
    patients = defaultdict(list)
    for row in source:
        if row.get('label') not in [0, 1]:
            raise ValueError('Labels must be0=sensitive or1=resistant')
        patients[str(row['patient_id'])].append(row)
    ids = sorted(patients, key=lambda value: (0, int(value)) if value.isdigit() else (1, value))
    strata = {key: int(any(row['label'] == 1 for row in subset)) for key, subset in patients.items()}
    ntest, nval = round(len(ids) * .2), round(len(ids) * .1)
    if nval < 2 or ntest < 2:
        raise ValueError('Too few patients for stratified AUC validation and testing')
    trainval, testing = train_test_split(ids, test_size=ntest, random_state=args.seed,
                                        stratify=[strata[key] for key in ids])
    training, validation = train_test_split(trainval, test_size=nval, random_state=args.seed,
                                            stratify=[strata[key] for key in trainval])
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    if any((target / (name + '.json')).exists() for name in ['train', 'val', 'test']):
        raise ValueError('Output split files already exist')
    report = {'seed': args.seed, 'split_unit': 'patient', 'excluded_duplicate_ID_count': len(repeated), 'splits': {}}
    for name, selected in [('train', training), ('val', validation), ('test', testing)]:
        selected = set(selected)
        subset = sorted([row for row in source if str(row['patient_id']) in selected], key=lambda row: str(row['id']))
        (target / (name + '.json')).write_text(json.dumps(subset, indent=2) + '\n')
        report['splits'][name] = {'patients': len(selected), 'nodes': len(subset), 'resistant_nodes': sum(row['label'] for row in subset)}
    (target / 'split_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
