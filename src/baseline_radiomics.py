import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import logging
import os
from pathlib import Path
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import radiomics
from radiomics import featureextractor
import SimpleITK as sitk
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from common import RUN, SEED, metrics, now, rows, select_threshold, sha, validate, write
SETTINGS = {'normalize': True, 'normalizeScale': 100.0, 'binWidth': 25.0, 'resampledPixelSpacing': [1.0, 1.0, 1.0], 'interpolator': 'sitkBSpline', 'label': 1, 'geometryTolerance': 0.0001, 'correctMask': False, 'minimumROIDimensions': 2, 'padDistance': 5, 'additionalInfo': False, 'distances': [1], 'wavelet': 'coif1', 'level': 1}
CLASSES = ['firstorder', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm']
CS = [0.001, 0.01, 0.1, 1.0, 10.0]
EXTRACTORS = None

def initialise():
    global EXTRACTORS
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    logging.getLogger('radiomics').setLevel(logging.ERROR)
    EXTRACTORS = {}
    for modality in ['T1', 'T2', 'CT']:
        extractor = featureextractor.RadiomicsFeatureExtractor(**SETTINGS)
        extractor.disableAllImageTypes()
        extractor.enableImageTypeByName('Original')
        extractor.enableImageTypeByName('Wavelet')
        extractor.disableAllFeatures()
        for name in CLASSES + ([] if modality == 'T2' else ['shape']):
            extractor.enableFeatureClassByName(name)
        EXTRACTORS[modality] = extractor

def extract(row):
    if EXTRACTORS is None:
        initialise()
    result, errors = ({}, [])
    for modality in ['T1', 'T2', 'CT']:
        try:
            image = sitk.ReadImage(row[modality], sitk.sitkFloat32)
            mask = sitk.ReadImage(row['CT_mask' if modality == 'CT' else 'MRI_mask'])
            mask = sitk.Cast(mask > 0, sitk.sitkUInt8)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                features = EXTRACTORS[modality].execute(image, mask)
            for name, value in features.items():
                if not name.startswith('diagnostics_'):
                    scalar = float(np.asarray(value))
                    result[modality + '__' + name] = scalar if np.isfinite(scalar) else np.nan
        except Exception as exc:
            errors.append({'modality': modality, 'error': type(exc).__name__ + ': ' + str(exc)})
    return (result, errors)

def signature(row, digest):
    files = {key: {'path': row[key], 'bytes': Path(row[key]).stat().st_size, 'mtime_ns': Path(row[key]).stat().st_mtime_ns} for key in ['T1', 'T2', 'CT', 'MRI_mask', 'CT_mask']}
    return hashlib.sha256(json.dumps({'files': files, 'protocol': digest}, sort_keys=True).encode()).hexdigest()

def cache_case(payload):
    row, digest, names = payload
    assert set(row) == {'id', 'T1', 'T2', 'CT', 'MRI_mask', 'CT_mask'}
    key = signature(row, digest)
    path = RUN / 'features' / (row['id'] + '.npz')
    if path.exists():
        with np.load(str(path), allow_pickle=False) as old:
            assert str(old['signature']) == key
            assert len(old['values']) == len(names)
            return (row['id'], json.loads(str(old['errors'])), int(np.isnan(old['values']).sum()))
    values, errors = extract(row)
    unexpected = set(values) - set(names)
    assert not unexpected, 'Unexpected radiomics columns: ' + str(sorted(unexpected))
    vector = np.array([values.get(name, np.nan) for name in names], dtype=np.float64)
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as file:
        np.savez_compressed(file, values=vector, signature=np.array(key), errors=np.array(json.dumps(errors)))
    temp.replace(path)
    return (row['id'], errors, int(np.isnan(vector).sum()))

def feature_matrix(subset, digest):
    names = json.loads((RUN / 'features/feature_names.json').read_text())
    matrix = []
    for row in subset:
        with np.load(str(RUN / 'features' / (row['id'] + '.npz')), allow_pickle=False) as record:
            assert str(record['signature']) == signature(row, digest)
            assert not json.loads(str(record['errors']))
            matrix.append(record['values'])
    result = np.stack(matrix)
    assert result.shape == (len(subset), len(names))
    return result

def prepare_features(subset, digest, phase):
    names = json.loads((RUN / 'features/feature_names.json').read_text())
    keys = ['id', 'T1', 'T2', 'CT', 'MRI_mask', 'CT_mask']
    payloads = [({key: row[key] for key in keys}, digest, names) for row in subset]
    started = time.monotonic()
    problems, nonfinite = ([], [])
    with ProcessPoolExecutor(max_workers=4, initializer=initialise) as pool:
        pending = [pool.submit(cache_case, payload) for payload in payloads]
        for i, future in enumerate(as_completed(pending), 1):
            node_id, errors, nan_count = future.result()
            if errors:
                problems.append({'id': node_id, 'errors': errors})
            if nan_count:
                nonfinite.append({'id': node_id, 'undefined_features': nan_count})
            if i == 1 or i % 20 == 0 or i == len(subset):
                elapsed = time.monotonic() - started
                write(RUN / 'radiomics_status.json', {'state': 'extracting', 'phase': phase, 'completed': i, 'total': len(subset), 'elapsed_seconds': elapsed, 'eta_seconds': elapsed / i * (len(subset) - i), 'pid': os.getpid(), 'updated_utc': now()})
                print('Radiomics', phase, i, '/', len(subset), 'elapsed', round(elapsed, 1), flush=True)
    write(RUN / 'audit' / ('radiomics_QC_' + phase + '.json'), {'nodes': len(subset), 'features': len(names), 'extraction_errors': problems, 'nodes_with_undefined_features': nonfinite, 'undefined_individual_features_policy': 'Training-median imputation; training-all-missing columns removed by variance filter', 'entire_modality_extraction_failure_policy': 'Stop; never silently drop nodes or substitute fabricated features'})
    assert not problems, 'Radiomics extraction errors require diagnosis; see QC report'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['preflight', 'train', 'test'], required=True)
    args = parser.parse_args()
    if args.mode == 'preflight':
        initialise()
        fixture = rows('train')
        chosen = [next((row for row in fixture if row['label'] == label)) for label in [0, 1]]
        names, observations = (None, [])
        for row in chosen:
            values, errors = extract(row)
            assert not errors, (row['id'], errors)
            if names is None:
                names = sorted(values)
            assert set(values) == set(names)
            assert not any(('PET' in name or 'T1CE' in name for name in names))
            observations.append({'id': row['id'], 'features': len(values), 'nonfinite_features': sum((not np.isfinite(v) for v in values.values()))})
        write(RUN / 'features/feature_names.json', names)
        write(RUN / 'audit/preflight_radiomics.json', {'state': 'pass', 'training_nodes': observations, 'feature_count': len(names), 'pyradiomics_version': radiomics.__version__, 'SimpleITK_version': sitk.Version_VersionString(), 'settings': SETTINGS, 'image_types': ['Original', 'Wavelet'], 'classes': CLASSES + ['shape'], 'MRI_shape_included_once': True, 'PET_and_T1CE_used': False, 'CT_is_normalised_cache_not_HU': True, 'all_modalities_imagewise_zscore_scale100': True})
        print('Radiomics preflight passed:', len(names), 'features', observations, flush=True)
        return
    _, digest = validate()
    directory = RUN / 'checkpoints/radiomics_LR'
    directory.mkdir(exist_ok=True)
    if args.mode == 'test':
        assert (RUN / 'all_training_complete.json').is_file()
        selected = json.loads((directory / 'selection.json').read_text())
        assert selected['protocol_sha256'] == digest
        assert sha(directory / 'model.joblib') == selected['model_sha256']
        prepare_features(rows('test'), digest, 'test')
        model = joblib.load(str(directory / 'model.joblib'))
        probability = model.predict_proba(feature_matrix(rows('test'), digest))[:, 1]
        frame = pd.DataFrame([{key: row[key] for key in ['id', 'patient_id', 'label']} for row in rows('test')])
        frame['prob'], frame['threshold'] = (probability, selected['threshold'])
        frame['pred_threshold'] = (probability >= selected['threshold']).astype(int)
        frame.to_csv(RUN / 'predictions/radiomics_LR_test.csv', index=False)
        write(RUN / 'metrics/radiomics_LR_test.json', metrics(frame.label, probability, selected['threshold']))
        write(RUN / 'radiomics_status.json', {'state': 'test_complete', 'finished_utc': now()})
        return
    prepare_features(rows('train') + rows('val'), digest, 'development')
    x_train, x_val = (feature_matrix(rows('train'), digest), feature_matrix(rows('val'), digest))
    y_train, y_val = ([r['label'] for r in rows('train')], [r['label'] for r in rows('val')])
    candidates, best_auc, best_model, selected = ([], -1.0, None, None)
    for c in CS:
        model = Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)), ('variance', VarianceThreshold(threshold=1e-12)), ('scale', StandardScaler()), ('LR', LogisticRegression(penalty='l1', C=c, solver='liblinear', max_iter=5000, class_weight=None, random_state=SEED))])
        model.fit(x_train, y_train)
        probability = model.predict_proba(x_val)[:, 1]
        threshold = select_threshold(y_val, probability)
        performance = metrics(y_val, probability, threshold)
        candidate = {'C': c, 'validation': performance, 'selected_features': int(np.count_nonzero(model.named_steps['LR'].coef_)), 'iterations': model.named_steps['LR'].n_iter_.tolist()}
        candidates.append(candidate)
        print(json.dumps(candidate), flush=True)
        if performance['auc'] > best_auc + 1e-12:
            best_auc, best_model = (performance['auc'], model)
            selected = {'C': c, 'threshold': threshold, 'validation': performance, 'selected_features': candidate['selected_features']}
    joblib.dump(best_model, str(directory / 'model.joblib'), compress=3)
    names = np.array(json.loads((RUN / 'features/feature_names.json').read_text()))
    names = names[best_model.named_steps['variance'].get_support()]
    coefficients = best_model.named_steps['LR'].coef_.ravel()
    write(directory / 'selected_features.json', [{'feature': str(name), 'standardised_coefficient': float(coef)} for name, coef in zip(names, coefficients) if coef != 0])
    selected.update(protocol_sha256=digest, model_sha256=sha(directory / 'model.joblib'), hyperparameter_candidates=candidates, test_used_for_selection=False, fit_cohort='training split', validation_cohort='validation split')
    write(directory / 'selection.json', selected)
    write(directory / 'training_complete.json', {'protocol_sha256': digest, 'best_validation_auc': best_auc, 'model_sha256': sha(directory / 'model.joblib'), 'finished_utc': now()})
    probability = best_model.predict_proba(x_val)[:, 1]
    frame = pd.DataFrame([{key: row[key] for key in ['id', 'patient_id', 'label']} for row in rows('val')])
    frame['prob'] = probability
    frame.to_csv(RUN / 'predictions/radiomics_LR_validation_best.csv', index=False)
    write(RUN / 'radiomics_status.json', {'state': 'training_complete', 'best_validation_auc': best_auc, 'C': selected['C'], 'selected_features': selected['selected_features'], 'finished_utc': now()})
if __name__ == '__main__':
    main()
