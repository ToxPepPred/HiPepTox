# HiPepTox

**Peptide toxicity prediction with residue–atom heterogeneous graphs and multimodal representations.**

HiPepTox combines atomic and residue information with protein language model embeddings and residue-level structural descriptors. Hierarchical message passing and late fusion integrate these representations for peptide toxicity classification. This repository provides model implementations, training and evaluation scripts, configurations, benchmark datasets, and baseline code.

## Model overview

![HiPepTox framework](figures/HiPepTox_framework.png)

The framework comprises four stages:

1. **Residue–atom heterogeneous graph construction:** represent residues and heavy atoms as distinct node types, with covalent bonds, atom-to-residue membership, and residue adjacency relations.
2. **Hierarchical message passing:** aggregate information at the atomic and residue levels to obtain a peptide-level graph representation.
3. **Sequence and structural representation:** average residue embeddings from layers 28, 29, and 30 of ESM2-t30-150M and combine them with 45 structural descriptors per residue.
4. **Late fusion and classification:** concatenate the independently encoded branch representations and pass them to the classification head.

The model is implemented in [model/hipeptox.py](model/hipeptox.py). Its configuration class name is `HiPepTox`.

## Repository structure

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
├── MANIFEST.in
├── figures/
│   └── HiPepTox_framework.png
├── model/
│   ├── hipeptox.py
│   └── baseline/
│       ├── captp/
│       ├── peptitox/
│       ├── plptp/
│       ├── structoxnet/
│       ├── tamper/
│       ├── toxgin/
│       ├── toxinpred3/
│       ├── toxipep/
│       ├── toxmsrc/
│       ├── toxteller/
│       └── registry.json
├── configs/
│   ├── reproduction/
│   │   ├── cdhit08.yaml
│   │   └── cdhit09.yaml
│   ├── single_task/
│   └── examples/
├── scripts/
├── util/
│   ├── data/
│   ├── datasets/
│   └── tasks/
└── data/
    ├── raw/
    └── processed/
```

| Path | Contents |
| --- | --- |
| `model/` | HiPepTox and baseline implementations |
| `configs/reproduction/` | Main-model configurations for CD-HIT 0.8 and 0.9 |
| `configs/single_task/` | Main-model and ablation configurations |
| `scripts/` | Training, evaluation, and feature preparation scripts |
| `util/` | Graph construction, datasets, training tasks, and metrics |
| `data/raw/` | Source sequence and label files, with a provenance catalogue |
| `data/processed/` | CSV datasets, fixed splits, and structure mappings |
| `figures/` | Model architecture figure |

## Installation

The main project declares the following dependencies. See [requirements.txt](requirements.txt) and [pyproject.toml](pyproject.toml) for the full lists.

| Dependency | Declared requirement |
| --- | --- |
| Python | ≥ 3.10 |
| PyTorch | ≥ 1.12 |
| TorchDrug | 0.2.1 |
| NumPy | ≥ 1.21 and < 2 |
| Other packages | pandas, SciPy, scikit-learn, PyYAML, Jinja2, tqdm, easydict, ninja |

Run the following commands from the repository root:

```bash
git clone https://github.com/ToxPepPred/HiPepTox.git
cd HiPepTox
python -m pip install -r requirements.txt
python -m pip install --no-deps --no-build-isolation -e .
export PYTHONPATH="$PWD"
```

Use compatible versions of PyTorch, CUDA, and the compiled dependencies required by TorchDrug. Baselines have additional requirements, documented in their own `requirements.txt` or `environment.yml` files and the [baseline registry](model/baseline/registry.json).

## Datasets

The main model uses the ToxiPep CD-HIT 0.8 and 0.9 benchmarks. Training and validation assignments are stored in the internal CSV files; each held-out benchmark test set is stored separately.

| Benchmark | Training | Validation | Test | Total |
| --- | ---: | ---: | ---: | ---: |
| CD-HIT 0.8 | 4,712 | 832 | 1,388 | 6,932 |
| CD-HIT 0.9 | 5,998 | 1,058 | 1,766 | 8,822 |

```text
data/processed/
├── toxipep08_internal_85_15/
├── toxipep08_external_test/
├── toxipep09_internal_85_15/
└── toxipep09_external_test/
```

Each directory contains `peptide_toxicity.csv` with the following fields:

| Field | Description |
| --- | --- |
| `id` | Sample identifier |
| `sequence` | Peptide amino acid sequence |
| `toxicity` | Binary label: `1` for toxic and `0` for non-toxic |
| `split` | `train`, `validation`, or `test` |
| `source` | Data source |
| `orig_record_id`, `header` | Identifiers retained from the source records |

The accompanying `peptide_toxicity_structoxnet_mapping.csv` files map samples to structure identifiers. Use the existing CSV splits when running experiments. Source repositories and dataset provenance are listed in [data/raw/sources.csv](data/raw/sources.csv).

## Feature preparation

Main-model training requires ESM2 embeddings and residue-level structural features. Trained weights and generated feature caches are not bundled with this release.

| Input | Path for CD-HIT 0.9 |
| --- | --- |
| ESM2 embeddings | `data/processed/toxipep09_esm2_t28_29_30_mean_embeddings.pt` |
| Internal structural features | `data/processed/structure/cdhit09_internal.pt` |
| Test structural features | `data/processed/structure/cdhit09_external.pt` |

Prepare predicted structures and a cache containing `pdb/`, `dssp/`, and `pdb2rr/` records. The [baseline resource guide](model/baseline/PREPARATION.md) describes the available structure-processing scripts. Once these records are ready, the following example extracts ESM2 embeddings and builds the internal structural features:

```bash
bash scripts/features.sh cdhit09 \
  --device cuda \
  --structure-map data/processed/toxipep09_internal_85_15/peptide_toxicity_structoxnet_mapping.csv \
  --structure-cache /path/to/internal_structure_cache \
  --structure-output data/processed/structure/cdhit09_internal.pt
```

Repeat with the test mapping, its structure cache, and `cdhit09_external.pt` as the output to prepare test features. For CD-HIT 0.8, use the corresponding `toxipep08` and `cdhit08` paths. Run `bash scripts/features.sh --help` for all options.

## Training HiPepTox

The main configurations are:

- [configs/reproduction/cdhit08.yaml](configs/reproduction/cdhit08.yaml)
- [configs/reproduction/cdhit09.yaml](configs/reproduction/cdhit09.yaml)

### Commands

```bash
# CD-HIT 0.9
bash scripts/train_seeds.sh cdhit09 '[0]'

# CD-HIT 0.8
bash scripts/train_seeds.sh cdhit08 '[0]'

# Run both benchmarks sequentially
bash scripts/train_seeds.sh all '[0]'
```

`'[0]'` selects GPU 0; use `null` for CPU execution. The script runs seeds 0, 3407, and 2024 sequentially. An optional third argument sets the output directory; its default is `outputs/reproduction`.

```bash
bash scripts/train_seeds.sh cdhit09 null outputs/reproduction_cpu
```

### Default training settings

| Parameter | Main-model configuration |
| --- | --- |
| Epochs | 50 |
| Batch size | 128 |
| Optimizer | Adam |
| Learning rate | 0.0005 |
| Weight decay | 0.001 |
| Learning-rate scheduler | CosineAnnealingLR, `T_max=50` |
| Checkpoint selection | Validation macro F1 |

Network and classification-head settings are defined separately in each YAML file. The training script evaluates the validation set after each epoch and saves the checkpoint with the best configured validation metric.

### Generated outputs

Each training run creates a separate output directory:

| File | Contents |
| --- | --- |
| `resolved_config.yaml` | Resolved run configuration |
| `run_metadata.json` | Run metadata |
| `validation_history.csv` | Per-epoch validation metrics |
| `best_checkpoint.json` | Validation-based checkpoint selection record |
| `best_valid_epoch_<N>.pth` | Best validation checkpoint |
| `model_epoch_<N>.pth` | Per-epoch checkpoints |

## Baselines

Baseline implementations are provided in [model/baseline/](model/baseline/).

| Method | Directory | Entry point |
| --- | --- | --- |
| ToxiPep | `toxipep/` | [Code/train_eval_csv.py](model/baseline/toxipep/Code/train_eval_csv.py) |
| ToxTeller: LR, RF, SVM, XGBoost | `toxteller/` | [program_resource/train_eval_csv.py](model/baseline/toxteller/program_resource/train_eval_csv.py) |
| ToxMSRC | `toxmsrc/` | [scripts/train_eval_csv_torch.py](model/baseline/toxmsrc/scripts/train_eval_csv_torch.py) |
| PLPTP | `plptp/` | [scripts/train_eval_csv.py](model/baseline/plptp/scripts/train_eval_csv.py) |
| CAPTP | `captp/` | [scripts/train_eval_csv.py](model/baseline/captp/scripts/train_eval_csv.py) |
| tAMPer | `tamper/` | [scripts/train_tamper_csv.py](model/baseline/tamper/scripts/train_tamper_csv.py) |
| ToxGIN | `toxgin/` | [scripts/train_toxgin_csv.py](model/baseline/toxgin/scripts/train_toxgin_csv.py) |
| PeptiTox | `peptitox/` | [train_cached.py](model/baseline/peptitox/train_cached.py) |
| StrucToxNet | `structoxnet/` | [scripts/train_csv.py](model/baseline/structoxnet/scripts/train_csv.py), [scripts/evaluate_csv.py](model/baseline/structoxnet/scripts/evaluate_csv.py) |
| ToxinPred3 | `toxinpred3/` | [scripts/evaluate_csv.py](model/baseline/toxinpred3/scripts/evaluate_csv.py), using the released implementation |

Each method retains its own implementation and parameter interface. See the [registry](model/baseline/registry.json) for entry points and the [resource preparation guide](model/baseline/PREPARATION.md) for required inputs and external resources.

## Main scripts

| Script | Purpose |
| --- | --- |
| `scripts/train.py` | Train a model from a YAML configuration |
| `scripts/train_seeds.sh` | Run main-model training across seeds |
| `scripts/features.sh` | Prepare ESM2 embeddings and structural feature inputs |
| `scripts/extract_plm.py` | Extract protein language model embeddings |
| `scripts/structure_features.py` | Compute 45-dimensional residue structural descriptors |
| `scripts/prepare_structure_inputs.py` | Prepare DSSP and residue-contact records |
| `scripts/build_contacts.py` | Compute residue-contact edges from PDB files |
| `scripts/csv_to_fasta.py` | Export CSV sequences to FASTA |
| `scripts/evaluate.py` | Evaluate a trained model |
| `scripts/ensemble.py` | Combine model predictions |

## License

HiPepTox project code is distributed under the [MIT License](LICENSE). Third-party baseline implementations retain their respective licensing terms and source notices.
