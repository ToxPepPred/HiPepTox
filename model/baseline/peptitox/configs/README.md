# Reproduction configurations

Six YAML files correspond to CD-HIT 0.8 / 0.9 and seeds 42, 2024, 3407. Parameters were recovered from each historical checkpoint; input/cache paths target Paper and new outputs go to results/peptitox/<dataset>/seed<seed>.

Run from the Paper root:

```bash
python model/baseline/peptitox/train_cached.py \
  --config model/baseline/peptitox/configs/cdhit09_seed42.yaml --device cpu
```

CLI options override YAML values, e.g. --device cpu. With a config, relative data/output paths are anchored to Paper, including CLI path overrides. The configurations retain training parameters and model input dimensions; each run selects its checkpoint by internal validation MCC.

These files recover saved settings; they do not assert bitwise reproduction across different devices or dependency versions.
