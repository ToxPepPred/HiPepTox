# Source datasets

This directory collects peptide toxicity datasets distributed with the source projects.

## Organization

Data are grouped by project, preserving the upstream relative directory structure where possible. [sources.csv](sources.csv) records the source repository, original path, path in this release, description, and whether each catalogued file is included.

## Included data groups

- `CAPTP/data/`: `train_data.txt`, `test1.txt`, and `test2.txt`.
- `StrucToxNet/data/`: source dataset files.
- `ToxMSRC/Raw data/` and `ToxMSRC/Example data/`: sequence, label, and example data files.
- `ToxiPep/Dataset/`: benchmark sequence and label files.
- `tAMPer/data/`: sequence files and example structures.

The catalogue also identifies source artifacts that are not included, such as generated feature archives. Static vocabularies and model resources belong to their respective implementation directories; the CAPTP token vocabulary is retained at `model/baseline/captp/data/AAindex.pkl` relative to the repository root.

Prepared CSV datasets and fixed train/validation/test assignments are in `data/processed/`. The [benchmark analysis plan](ToxiPep_benchmark_analysis_plan.md) describes dataset comparison procedures without bundling historical analysis results.
