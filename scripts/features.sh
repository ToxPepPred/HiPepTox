#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON="${PYTHON:-python}"

usage() {
  cat <<'EOF'
Usage: bash scripts/features.sh cdhit08|cdhit09 [options]

Options:
  --device DEVICE             PLM extraction device; default cpu
  --batch-size N              PLM extraction batch size; default 8
  --embedding-output FILE     Custom new PLM output; default canonical data cache
  --structure-map CSV         Mapping CSV with id, sequence, structoxnet_sample_id
  --structure-cache DIR        Existing cache containing pdb/, dssp/, pdb2rr/
  --structure-output FILE     New output for explicit structure rebuilding

Existing PLM files are reused without modification. Choose a new output filename
to recompute features. Structure rebuilding requires all three structure options;
it does not run structure prediction or DSSP. Paths are relative to Paper root.
PYTHON may be set to an executable path in an activated environment.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# -lt 1 ]]; then usage >&2; exit 2; fi
case "$1" in
  cdhit08|cdhit09) DATASET="$1" ;;
  *) usage >&2; exit 2 ;;
esac
shift
SUFFIX="${DATASET#cdhit}"
DEVICE="cpu"
BATCH_SIZE="8"
EMBEDDING_OUTPUT="data/processed/toxipep${SUFFIX}_esm2_t28_29_30_mean_embeddings.pt"
STRUCTURE_MAP=""
STRUCTURE_CACHE=""
STRUCTURE_OUTPUT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device|--batch-size|--embedding-output|--structure-map|--structure-cache|--structure-output)
      if [[ $# -lt 2 || -z "$2" ]]; then
        echo "A value is required for $1" >&2; exit 2
      fi
      option="$1"; value="$2"; shift 2
      case "$option" in
        --device) DEVICE="$value" ;;
        --batch-size) BATCH_SIZE="$value" ;;
        --embedding-output) EMBEDDING_OUTPUT="$value" ;;
        --structure-map) STRUCTURE_MAP="$value" ;;
        --structure-cache) STRUCTURE_CACHE="$value" ;;
        --structure-output) STRUCTURE_OUTPUT="$value" ;;
      esac
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
if [[ ! "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "--batch-size must be a positive integer" >&2; exit 2
fi
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON" >&2; exit 1
fi
cd "$PAPER_ROOT"
export PYTHONPATH="$PAPER_ROOT:${PYTHONPATH:-}"

INTERNAL_CSV="data/processed/toxipep${SUFFIX}_internal_85_15/peptide_toxicity.csv"
EXTERNAL_CSV="data/processed/toxipep${SUFFIX}_external_test/peptide_toxicity.csv"
ESM_MODEL="${ESM_MODEL:-esm2_t30_150M_UR50D}"
for input in "$INTERNAL_CSV" "$EXTERNAL_CSV"; do
  if [[ ! -f "$input" ]]; then echo "Required CSV missing: $input" >&2; exit 1; fi
done

# Validate optional structure inputs before starting an expensive PLM extraction.
if [[ -n "$STRUCTURE_MAP$STRUCTURE_CACHE$STRUCTURE_OUTPUT" ]]; then
  if [[ -z "$STRUCTURE_MAP" || -z "$STRUCTURE_CACHE" || -z "$STRUCTURE_OUTPUT" ]]; then
    echo "Specify --structure-map, --structure-cache and --structure-output together" >&2; exit 2
  fi
  if [[ ! -f "$STRUCTURE_MAP" ]]; then echo "Mapping CSV missing: $STRUCTURE_MAP" >&2; exit 1; fi
  for subdir in pdb dssp pdb2rr; do
    if [[ ! -d "$STRUCTURE_CACHE/$subdir" ]]; then
      echo "Structure cache directory missing: $STRUCTURE_CACHE/$subdir" >&2; exit 1
    fi
  done
  if [[ -e "$STRUCTURE_OUTPUT" ]]; then
    echo "Structure output exists; choose a new filename: $STRUCTURE_OUTPUT" >&2; exit 1
  fi
fi

if [[ -f "$EMBEDDING_OUTPUT" ]]; then
  echo "Reusing existing PLM cache: $EMBEDDING_OUTPUT"
else
  if [[ -e "$EMBEDDING_OUTPUT" ]]; then
    echo "PLM output is not a regular file: $EMBEDDING_OUTPUT" >&2; exit 1
  fi
  # ESM_MODEL accepts a fair-esm model name or an explicitly supplied local checkpoint.
  EMBEDDING_TEMP="${EMBEDDING_OUTPUT}.partial.$$"
  "$PYTHON" -u scripts/extract_plm.py \
    --input_csvs "$INTERNAL_CSV" "$EXTERNAL_CSV" \
    --output_file "$EMBEDDING_TEMP" \
    --backend esm \
    --model_name_or_path "$ESM_MODEL" \
    --repr_layers 28 29 30 \
    --layer_pool mean \
    --batch_size "$BATCH_SIZE" \
    --device "$DEVICE"
  if [[ -e "$EMBEDDING_OUTPUT" ]]; then
    echo "PLM destination appeared during extraction; result retained at $EMBEDDING_TEMP" >&2
    exit 1
  fi
  mv -n -- "$EMBEDDING_TEMP" "$EMBEDDING_OUTPUT"
fi

if [[ -n "$STRUCTURE_OUTPUT" ]]; then
  "$PYTHON" -u scripts/structure_features.py \
    --input-csv "$STRUCTURE_MAP" \
    --cache-dir "$STRUCTURE_CACHE" \
    --output-file "$STRUCTURE_OUTPUT" \
    --id-col id \
    --cache-id-col structoxnet_sample_id \
    --sequence-col sequence
  echo "New structure features: $STRUCTURE_OUTPUT"
else
  MISSING_STRUCTURE=0
  for split in internal external; do
    feature_file="data/processed/structure/${DATASET}_${split}.pt"
    if [[ -f "$feature_file" ]]; then
      echo "Existing structure features: $feature_file"
    else
      echo "Structure features absent: $feature_file; see README.md" >&2
      MISSING_STRUCTURE=1
    fi
  done
  if [[ "$MISSING_STRUCTURE" -ne 0 ]]; then
    echo "PLM preparation finished, but main-model structure inputs are incomplete." >&2
    exit 1
  fi
fi
echo "PLM cache: $EMBEDDING_OUTPUT"
