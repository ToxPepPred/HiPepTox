# Reproduction configurations

Six YAML files cover CD-HIT 0.8 and 0.9 with seeds 42, 2024, and 3407. Parameters were recovered from the corresponding historical checkpoints. Relative input/cache paths are resolved from the repository root; new outputs go to `results/peptitox/<dataset>/seed<seed>`.

Run from the repository root:

```bash
python model/baseline/peptitox/train_cached.py \
  --config model/baseline/peptitox/configs/cdhit09_seed42.yaml --device cpu
```

Command-line arguments override configuration values, including path arguments. The configurations retain training parameters and model input dimensions. Each run selects its checkpoint using internal validation MCC.

These files recover saved settings; they do not assert bitwise reproduction across devices or dependency versions.
