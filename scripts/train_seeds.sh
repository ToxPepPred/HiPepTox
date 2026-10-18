#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON="${PYTHON:-python}"

usage() {
  cat <<'EOF'
Usage: bash scripts/train_seeds.sh cdhit08|cdhit09|all [gpus] [output_root]

Runs seeds 0, 3407, 2024 sequentially using the internal train/validation split.
gpus defaults to null (CPU); pass '[0]' for one GPU.
output_root defaults to outputs/reproduction and is relative to Paper root.
PYTHON may be set to an executable path in an activated environment.
This script does not scan or select checkpoints on the external test set.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# -lt 1 || $# -gt 3 ]]; then
  usage >&2
  exit 2
fi
case "$1" in
  cdhit08) datasets=(cdhit08) ;;
  cdhit09) datasets=(cdhit09) ;;
  all) datasets=(cdhit08 cdhit09) ;;
  *) usage >&2; exit 2 ;;
esac
GPUS="${2:-null}"
OUTPUT_ROOT="${3:-outputs/reproduction}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON" >&2
  exit 1
fi
cd "$PAPER_ROOT"
export PYTHONPATH="$PAPER_ROOT:${PYTHONPATH:-}"

# Resolve all inputs before starting the first 50-epoch run.
for dataset in "${datasets[@]}"; do
  suffix="${dataset#cdhit}"
  for input in \
    "configs/reproduction/${dataset}.yaml" \
    "data/processed/toxipep${suffix}_internal_85_15/peptide_toxicity.csv" \
    "data/processed/toxipep${suffix}_esm2_t28_29_30_mean_embeddings.pt" \
    "data/processed/structure/${dataset}_internal.pt"; do
    if [[ ! -f "$input" ]]; then
      echo "Required input missing: $PAPER_ROOT/$input" >&2
      exit 1
    fi
  done
done

for dataset in "${datasets[@]}"; do
  suffix="${dataset#cdhit}"
  for seed in 0 3407 2024; do
    echo "Training $dataset, seed $seed (50 epochs; gpus=$GPUS)"
    "$PYTHON" -u scripts/train.py \
      --config "configs/reproduction/${dataset}.yaml" \
      --output_dir "${OUTPUT_ROOT}/${dataset}/seed${seed}" \
      --dataset_path "data/processed/toxipep${suffix}_internal_85_15" \
      --plm_embedding_file "data/processed/toxipep${suffix}_esm2_t28_29_30_mean_embeddings.pt" \
      --structure_feature_file "data/processed/structure/${dataset}_internal.pt" \
      --gpus "$GPUS" \
      --seed "$seed"
  done
done
echo "Training finished. Evaluate the newly selected validation checkpoints using README.md."

