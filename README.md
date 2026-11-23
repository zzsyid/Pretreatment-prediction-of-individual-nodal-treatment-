## Environment

Python 3.8 is the tested runtime. Use a CUDA-compatible PyTorch installation for
training; the synthetic smoke test and inference also support CPU execution.

```bash
python -m pip install -r requirements.txt
python smoke_test.py
```

The optional radiomics control additionally requires:

```bash
python -m pip install --no-build-isolation -r requirements-radiomics.txt
```

## Input contract

Provide separate `train.json`, `val.json` and `test.json` files following
`examples/manifest.example.json`. Patients, node IDs and image paths must be
mutually exclusive across splits. Each split needs both binary labels for
AUC-based model selection/evaluation. Label **1 means treatment resistant**;
label **0 means treatment sensitive**. Region indices are integers from 0 to 14.
Physical volumes are in cubic millimetres, before the loader's log1p transform.
Paths may be absolute or relative to `--data-root` (the manifest directory when
that option is omitted).

For a new cohort, a patient-stratified 7:1:2 split can be generated from a combined
manifest using `python split.py --manifest all_nodes.json --output manifests`.
Duplicate node IDs are rejected unless `--exclude-duplicates` is explicitly
specified; that option excludes all copies of each repeated ID.

Images must be preprocessed node-centred NIfTI ROI caches. MRI inputs must carry
the upstream normalisation already applied to those caches. CT caches use a
normalised intensity range and are not raw HU images. This release loads those
caches and performs the final native-grid ROI crop, augmentation and 64-cubed
resizing; raw DICOM conversion and upstream MRI/CT/PET registration are outside
this input contract. MRI-family images and the MRI mask must share a grid, as
must CT/PET and the CT mask. MRI and CT retain their separate physical grids.

The privileged teacher needs T1, contrast-enhanced T1, T2, CT and PET for source
training. The student receives **T1/T2/CT only**, plus the recorded nodal region
and MRI-derived volume. Missing contrast-enhanced T1 and PET inputs are zeroed
before the student forward pass. No PET is required for student inference.

## Train and evaluate

```bash
python run.py --train-json manifests/train.json --val-json manifests/val.json \
  --test-json manifests/test.json --data-root data --output runs/method --gpu 0
```

One implementation executes five necessary phases: teacher pretraining, student
pretraining, residual-path training, teacher adaptation and final student
training. Warm-up and adaptation use explicit preprocessing, normalisation and token-state
settings in `recipes.json`; there are no alternative source snapshots. Teacher
and initial student weights start randomly. All later initialisation comes from
the current run's own prior phases. No external checkpoint is required.

The default seed is 3407, early-stop patience is 5.
Only `best.pth` and resumable `last.pth` are stored per phase. Repeating the same
command resumes unfinished training. Use `--batch-size` to reduce GPU memory;
avoid a one-observation final training batch because of batch normalisation.
Batch overrides change the training protocol. `--max-epochs` is available for
execution checks and is not the full experiment protocol.

Model weights, the student residual scale in 0..1 (step 0.05), and all decision
thresholds are selected on validation data. Test evaluation starts afterward.
The frozen settings are written to `selection.json`; test predictions and
patient-cluster bootstrap metrics are in `predictions/` and `metrics/`.

## Student inference

```bash
python predict.py --manifest manifests/inference.json --data-root data \
  --checkpoint runs/method/checkpoints/student_final/best.pth \
  --selection runs/method/selection.json --output predictions.csv
```

Labels are optional for inference. T1/T2/CT, corresponding MRI/CT masks, nodal
region and MRI volume follow the same cache preprocessing contract.

## Classic controls

```bash
python baselines.py --train-json manifests/train.json --val-json manifests/val.json \
  --test-json manifests/test.json --data-root data --output runs/baselines --gpu 0
```

Controls are unmodified MONAI 3D ResNet-18 and DenseNet-121 architectures, trained
from scratch on the three T1/T2/CT channels, and PyRadiomics plus L1 logistic
regression. The CNNs have no explicit clinical-metadata input and use the final
ROI preprocessing and training augmentations. Their optimiser/loss, validation
selection and early stopping follow the common protocol. The radiomics control
uses original and wavelet shape/intensity/texture features, image-wise zscore
normalisation (scale 100), a fixed bin width of 25 normalised units and 1-mm
resampling. Training medians, variance filtering, scaling and L1 regression are
fitted on training nodes only; C and threshold are selected on validation.
Use `--models resnet18 densenet121` to omit the optional radiomics dependency.

```bash
python compare.py --reference method --output comparison --predictions \
  method=runs/method/predictions/student_test.csv \
  resnet=runs/baselines/predictions/resnet18_test.csv \
  densenet=runs/baselines/predictions/densenet121_test.csv \
  radiomics=runs/baselines/predictions/radiomics_LR_test.csv
```

Comparisons require identical evaluated nodes, patient mappings and labels.
The default is 2,000 paired patient-cluster bootstrap draws with percentile 95%
intervals for pooled-node AUC and paired AUC differences. These intervals are
not adjusted for multiple comparisons. CSV scores are read with round-trip
floating-point parsing to preserve equality at fixed thresholds.

## Synthetic execution example

```bash
python make_demo.py --output demo_data
python run.py --train-json demo_data/train.json --val-json demo_data/val.json \
  --test-json demo_data/test.json --output demo_run --max-epochs 1 --batch-size 2
```

The generator creates entirely artificial images and identifiers. Synthetic
results are execution checks and have no clinical or scientific interpretation.

## Third-party software

PyTorch, MONAI, NumPy/SciPy, scikit-learn, pandas, NiBabel and optional
PyRadiomics/SimpleITK/PyWavelets are external dependencies. Their source trees,
binaries and model weights are not redistributed in this package. Their
respective licences apply to those dependencies.
