# ToxiPep benchmark analysis plan

This document describes procedures for comparing peptide toxicity benchmarks. Historical analysis outputs and preliminary result tables are not included in the code release.

## Objectives

Compare dataset scale, class balance, redundancy control, evaluation splits, and sequence-space coverage. Report these properties separately so that conclusions follow from the measured evidence.

The comparisons concern the available benchmark collections. They do not establish that a dataset represents the full space of peptide toxicity.

## Input data and provenance

Use source sequences and labels from the project directories under `data/raw/`. Source repositories and packaged-file availability are recorded in [sources.csv](sources.csv). Prepared CSV datasets and fixed split assignments are under `data/processed/`.

Normalize sequence case and remove surrounding whitespace before exact-sequence comparisons. Retain dataset, split, label, and source identifiers. Summarize duplicate sequences separately from unique-sequence counts and check whether repeated sequences have conflicting labels.

## Dataset composition and evaluation design

For each dataset and split, report:

- Total records and unique sequences.
- Positive and negative sample counts and their ratio.
- Sequence length range and distribution.
- Available redundancy thresholds and independent test splits.
- Exact sequence overlap between training and test sets.
- Available sample-level metadata.

Suggested table: **Comparison of benchmark design across peptide toxicity datasets**.

## Sequence-space coverage

Compare length, amino acid composition, net charge, hydrophobicity, aromatic residue fraction, Cys fraction, dipeptide/tripeptide frequencies, and low-complexity sequence frequency.

Suggested visualizations include length ECDFs or violin plots, amino acid frequency heatmaps, physicochemical feature plots, and UMAP/t-SNE projections of ESM2 or k-mer TF-IDF representations.

Quantitative coverage measures can include k-nearest-neighbor coverage, cluster occupancy, and class balance within sequence clusters. Define the embedding, clustering procedure, neighborhood size, and normalization used for each measure. Where convex-hull or alpha-shape coverage is used, report the representation and dimensionality on which it is calculated.

## Redundancy and overlap

Evaluate three forms of overlap:

1. Duplicates within each dataset or split.
2. Exact training–test overlap.
3. Exact and high-similarity overlap across datasets.

Report exact-overlap ratios, sequence-set Jaccard similarity, CD-HIT-based overlap at the selected thresholds, and nearest-training-sequence identity for test samples. Suitable plots include overlap heatmaps, UpSet plots, and nearest-neighbor identity histograms.

Shared source sequences should be identified when interpreting cross-dataset comparisons. A separate test file does not, by itself, establish independence from the training data.

## Cross-dataset transfer

Construct a train-on-A/test-on-B matrix using consistent model settings and clearly defined source and target splits. Candidate comparisons include ToxiPep, CAPTP, StrucToxNet, ToxMSRC, and tAMPer, subject to the available data and overlap checks.

Use MCC as a primary transfer metric, with AUROC and balanced Accuracy as complementary measures. Select checkpoints using only the designated validation data. Report sample exclusions, duplicate handling, and uncertainty alongside the transfer results.

## Subgroup analysis

Possible sequence-length groups are ≤10, 11–20, 21–30, and 31–50 amino acids. Additional groups may be defined by Cys content, positive charge, hydrophobicity, or helix propensity.

For each subgroup, summarize dataset coverage, class balance, and predictive performance. Define subgroup thresholds before comparing outcomes and report sample counts so that sparse groups can be identified.

## Reporting

Organize the evidence around dataset composition, coverage, overlap, transfer, and subgroup consistency. Distinguish measured results from proposed analyses. Claims about broader coverage or stronger generalization should be tied to their corresponding quantitative comparisons.

Potential figures and tables include:

- Dataset composition and evaluation-design table.
- Length and physicochemical feature distributions.
- Cross-dataset overlap heatmap.
- Cross-dataset transfer performance heatmap.
- Supplementary amino acid frequency, embedding, subgroup, and nearest-neighbor analyses.

## Suggested execution order

1. Generate descriptive statistics and label-consistency checks.
2. Quantify exact overlap within and across datasets.
3. Compare length, composition, and physicochemical distributions.
4. Evaluate sequence-space coverage and nearest-neighbor similarity.
5. Run cross-dataset transfer and subgroup analyses.

Suggested generated files include `dataset_summary_table.csv`, `dataset_overlap_heatmap.svg`, `length_distribution.svg`, `physchem_distribution.svg`, and `cross_dataset_transfer_plan.md`. Write new outputs to a separate analysis directory.
