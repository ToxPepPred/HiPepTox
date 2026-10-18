# Baseline resource preparation

This directory contains model implementations, CSV training/evaluation interfaces, original example data, and static input definitions. Released and retrained checkpoints, feature caches, historical predictions, logs, and reproduction-check directories are excluded.

Main-model configurations are in [configs/reproduction](../../configs/reproduction). Baseline parameters are supplied through their training interfaces and accompanying configuration files.

## Required resources

Paths in the following table are relative to `model/baseline/`.

| Method | Entry point | Resources to prepare |
| --- | --- | --- |
| ToxiPep | `toxipep/Code/train_eval_csv.py` | Generate atom/graph features from the CSV inputs and train the model. |
| ToxTeller | `toxteller/program_resource/train_eval_csv.py` | Extract sequence features from CSV inputs, then fit the scaler and classifiers. |
| ToxMSRC | `toxmsrc/scripts/train_eval_csv_torch.py` | Obtain the original Word2Vec model before running `build_word2vec_features.py`. Retraining Word2Vec changes the input representation. |
| PLPTP | `plptp/scripts/train_eval_csv.py` | Obtain `model.pkl` using the upstream instructions in `plptp/model/model` and supply it with `--init-weights`. Architecture and tokenizer definitions are retained in `local_models/`. |
| CAPTP | `captp/scripts/train_eval_csv.py` | The static token-to-index vocabulary `AAindex.pkl` is included. Use `--train-from-scratch` for random initialization; reproducing pretrained initialization requires the original released weights. |
| tAMPer | `tamper/scripts/train_tamper_csv.py` | Prepare PDB structures and obtain ESM2-t12-35M through `fair-esm`. To reproduce the original initialization, provide `pre_GNNs.pt` through `--pre-gnn-checkpoint`. |
| ToxGIN | `toxgin/scripts/train_toxgin_csv.py` | Prepare PDB structures. The default language model is `facebook/esm2_t36_3B_UR50D`; `aaindex1.csv` is included. |
| PeptiTox | `peptitox/train_cached.py` | Prepare PDB caches and ESM embeddings. Original feature scripts include `run_esm.py`, `run_esmfold.py`, and `preprocess.py`; see `peptitox/README.md`. |
| StrucToxNet | `structoxnet/scripts/train_csv.py`, `structoxnet/scripts/evaluate_csv.py` | Prepare ProtT5, ESMFold, and `mkdssp`. Original `x_min.npy`/`x_max.npy` arrays are included. Supply input and output paths through command-line arguments; see `structoxnet/README.md`. |
| ToxinPred3 | `toxinpred3/scripts/evaluate_csv.py` | The released prediction/evaluation implementation is included; a complete retraining workflow is not provided. Obtain model weights from the upstream distribution before evaluation. |

## Inputs and outputs

Shared CSV datasets are in `data/processed/` at the repository root. Use the existing `split` assignments and keep the held-out test sets separate.

Model implementations, feature definitions, initialization options, and training parameters are retained. Write newly generated checkpoints, graphs, PLM embeddings, and predictions to `outputs/` or another explicitly selected output directory.

Dependencies are listed in [registry.json](registry.json) and the original requirements/environment files in each baseline directory.

## Structure inputs

Structure prediction code is available in `peptitox/run_esmfold.py` and `tamper/local_implementation/src/ESMfolding.py`. The latter accepts FASTA input through `-seqs` and an output PDB directory through `-out`.

CSV sample identifiers and structure-mapping columns are retained. PDB filenames must match the `id`, `structoxnet_sample_id`, or sequence identifier expected by the selected interface.

The main model's 45-dimensional structural descriptors also require DSSP and residue-contact records. From the repository root, use `scripts/prepare_structure_inputs.py` to call `mkdssp` and prepare contact records, then use `scripts/structure_features.py` to extract the descriptors. The [main README](../../README.md#feature-preparation) describes the feature-preparation interface.

Downloading structure-prediction models and language-model weights requires network access and suitable computing resources. Newly predicted structures and stochastic training can produce differences from historical runs.
