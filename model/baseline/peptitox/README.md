# PeptiTox baseline

This directory contains the local PeptiTox implementation used in the comparison experiments. The model, geometric features, mean pooling, and classification head retain their original implementation. This is an integration of the existing local source snapshot; it is not identified as a verified snapshot of the latest official release.

## Files

- `PeptiTox.py`: model implementation.
- `ProteinGraphDataset.py`: original graph dataset.
- `train_cached.py`: project interface accepting explicit internal and external PDB caches.
- `train_cached_external.py`: original cached-training script retained for provenance.
- `main.py`: original training entry point.
- `structure2graph.py`, `preprocess.py`, `run_esm.py`, `run_esmfold.py`: original structure, chemical-feature, and PLM preparation scripts.

The original `main.py` and preprocessing scripts expect the upstream `Data/` directory layout. Use `train_cached.py` to supply this repository's data and cache paths explicitly. The preparation scripts may download models or generate files when executed; package import does not run them automatically.

## Environment and execution

The cached-data interface requires PyTorch, PyG, torch-scatter, torch-cluster, NumPy, and pandas; see `requirements.txt`. Install PyG extensions compatible with the selected PyTorch/CUDA versions. The standalone interface previously passed a CPU compatibility check with Python 3.8.20, PyTorch 2.4.1+cu121, PyG 2.6.1, torch-scatter 2.1.2+pt24cu121, and torch-cluster 1.6.3+pt24cu121. Requirements for installing the main project are listed in the repository's top-level README. The original ESM, ESMFold, and RDKit preparation workflows have their own additional dependencies.

Run commands from the repository root after activating a compatible environment:

```bash
python model/baseline/peptitox/train_cached.py --help
```

Example for CD-HIT 0.9 using prepared caches:

```bash
python model/baseline/peptitox/train_cached.py \
  --train-data-dir data/processed/toxipep09_internal_85_15 \
  --external-data-dir data/processed/toxipep09_external_test \
  --plm-embedding-file data/processed/toxipep09_esm2_t28_29_30_mean_embeddings.pt \
  --internal-cache-dir assets/structure/structure_cache/toxipep09_internal_85_15_cache \
  --external-cache-dir assets/structure/structure_cache/toxipep09_external_test_eval_gpu_fixed \
  --output-dir results/peptitox/cdhit09/seed42 \
  --seed 42 --device cuda
```

Use the corresponding `toxipep08` inputs, embeddings, and caches for CD-HIT 0.8. Defaults are a maximum of 100 epochs and patience of 20. Checkpoints are selected by internal validation MCC; the external set is evaluated after selection, using a classification threshold of 0.5. This command describes a new run; integrating the code did not retrain the model.

## Configuration files

`configs/` contains six YAML files covering CD-HIT 0.8 and 0.9 with seeds 42, 2024, and 3407. Their parameters were recovered from the corresponding checkpoint `args`. PLM embeddings have dimension 640; node and edge inputs have dimensions 824 and 450, respectively.

```bash
python model/baseline/peptitox/train_cached.py \
  --config model/baseline/peptitox/configs/cdhit09_seed42.yaml --device cpu
```

Command-line arguments override configuration values. When a configuration is supplied, relative input/output paths are resolved from the repository root. New outputs are written to `results/peptitox/<dataset>/seed<seed>`. Configurations retain training parameters and model input dimensions; each run selects its checkpoint using internal validation MCC. See the [configuration guide](configs/README.md).

No upstream LICENSE file was found in the local source snapshot. The HiPepTox project license does not replace the licensing terms of this third-party implementation.
