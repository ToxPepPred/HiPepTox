# StrucToxNet baseline

This directory integrates the StrucToxNet model source, original preprocessing scripts, and the project's existing CSV training and evaluation interfaces. The model implementation is unchanged. The CSV interfaces retain their existing implementation, with the default DSSP executable resolved from `PATH`; an explicit path can also be supplied through `--mkdssp`.

See [SOURCE.json](SOURCE.json) for source-snapshot information and file checksums, and [README.upstream.md](README.upstream.md) for the upstream instructions.

## Files

| Path | Purpose |
| --- | --- |
| `model/StrucToxNet.py` | EGNN model and classifier |
| `scripts/train_csv.py` | Train using CSV train/validation splits and select the best validation checkpoint |
| `scripts/evaluate_csv.py` | Evaluate fixed weights and export metrics and per-sample predictions |
| `Preprocessing/` | Original structure, contact, and ProtT5 feature-processing scripts |
| `Preprocessing/inputs/x_min.npy`, `x_max.npy` | Original static ProtT5 normalization arrays |
| `configuration/`, `preprocess/`, `util/` | Original configuration, data-loading, and metric utilities |
| `main.py` | Upstream prediction entry point, using its documented relative paths and resources |
| `environment.yml` | Original environment specification |
| `requirements.txt` | Main Python dependencies for the CSV interfaces |

## Environment and resources

The CSV interfaces require Python 3.10 or later. Install PyTorch, PyG, torch-scatter, and torch-cluster versions compatible with the execution platform, followed by the remaining dependencies.

Prepare ProtT5, ESMFold, and `mkdssp` separately and supply their locations through the corresponding arguments. The normalization arrays are included. Large language-model files, trained checkpoints, PDB/DSSP/PLM caches, and historical predictions are not bundled.

Run the following commands from `model/baseline/structoxnet/`:

```bash
PROJECT_ROOT=$(cd ../../.. && pwd)

# CD-HIT 0.9: use the existing CSV train/validation split.
python scripts/train_csv.py \
  --input-csv "$PROJECT_ROOT/data/processed/toxipep09_internal_85_15/peptide_toxicity.csv" \
  --run-dir "$PROJECT_ROOT/outputs/structoxnet/cdhit09/train_features" \
  --output-dir "$PROJECT_ROOT/outputs/structoxnet/cdhit09/train" \
  --prot-t5-model /path/to/prot_t5_xl_uniref50 \
  --esmfold-model /path/to/esmfold_v1 \
  --mkdssp /path/to/mkdssp \
  --select-metric accuracy \
  --seed 42 \
  --device cuda

# Evaluate the checkpoint produced by that training run.
python scripts/evaluate_csv.py \
  --input-csv "$PROJECT_ROOT/data/processed/toxipep09_external_test/peptide_toxicity.csv" \
  --weights "$PROJECT_ROOT/outputs/structoxnet/cdhit09/train/best_valid_model.pth" \
  --run-dir "$PROJECT_ROOT/outputs/structoxnet/cdhit09/test_features" \
  --prot-t5-model /path/to/prot_t5_xl_uniref50 \
  --esmfold-model /path/to/esmfold_v1 \
  --mkdssp /path/to/mkdssp \
  --output-json "$PROJECT_ROOT/outputs/structoxnet/cdhit09/test_metrics.json" \
  --predictions-csv "$PROJECT_ROOT/outputs/structoxnet/cdhit09/test_predictions.csv" \
  --device cuda
```

For CD-HIT 0.8, replace `toxipep09` with `toxipep08` in the input paths and use a separate `cdhit08` output directory. Run `python scripts/train_csv.py --help` or `python scripts/evaluate_csv.py --help` for all arguments.

The example explicitly selects checkpoints by validation Accuracy, matching the historical comparison corresponding to manuscript Table 2. Other selection metrics can be supplied through `--select-metric`; the generic CLI default is MCC. Remaining hyperparameters are controlled by the training arguments.

Follow `README.upstream.md` when running the original preprocessing stages from `Preprocessing/`. The CSV interfaces manage their own cache directories.

Original author credits are retained in the source. The source snapshot did not contain a separate LICENSE file; this is recorded in `SOURCE.json`.
