# StrucToxNet baseline

本目录整合 StrucToxNet 模型源码、原始预处理代码，以及本项目已有的 CSV 训练和评估入口。
模型实现保持原样；CSV 接口沿用原工程的实现，仅将 DSSP 默认位置改为从 PATH 查找，亦可用 `--mkdssp` 显式指定。
源码快照及文件校验值见 [SOURCE.json](SOURCE.json)，原作者说明见 [README.upstream.md](README.upstream.md)。

## 文件

| 路径 | 用途 |
| --- | --- |
| `model/StrucToxNet.py` | EGNN 模型与分类器 |
| `scripts/train_csv.py` | 按 CSV 的 train/validation 划分训练并选择验证集最优模型 |
| `scripts/evaluate_csv.py` | 固定权重评估、指标与逐样本预测导出 |
| `Preprocessing/` | 原始结构、接触及 ProtT5 特征处理脚本 |
| `Preprocessing/inputs/x_min.npy`、`x_max.npy` | 原工程的静态 ProtT5 归一化数组 |
| `configuration/`、`preprocess/`、`util/` | 原始配置、数据加载与指标工具 |
| `main.py` | 原作者的预测入口，使用其 README 中约定的相对路径与资源 |
| `environment.yml` | 原始环境文件 |
| `requirements.txt` | CSV 接口所需的主要 Python 依赖 |

## 环境和资源

CSV 接口使用 Python 3.10 或更新版本。先安装与运行平台匹配的 PyTorch、PyG、torch-scatter 和 torch-cluster，再安装其余依赖。
ProtT5、ESMFold 与 mkdssp 由使用者在环境中单独准备，并通过相应参数传入。归一化数组已随代码保留。
大模型文件、训练权重、PDB/DSSP/PLM 缓存和历史预测结果均不随本精简包复制。

以下命令从 `push_paper/model/baseline/structoxnet` 目录运行：

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

CD-HIT 0.8 对应将数据目录中的 `toxipep09` 换为 `toxipep08`，并为输出使用独立的 `cdhit08` 目录。
可用 `python scripts/train_csv.py --help` 和 `python scripts/evaluate_csv.py --help` 查看全部参数。
上述 StrucToxNet 示例使用验证集 Accuracy 选模，与正文表 2 对应的历史比较记录保持一致。脚本支持用 --select-metric 选择其他指标；CLI 的通用默认值为 MCC。其余模型超参数由显式命令参数确定。

原始分步预处理脚本按 `README.upstream.md` 从 `Preprocessing/` 目录执行；CSV 入口负责自己的缓存目录。
本次整合保留源码中的原作者署名；来源快照没有单独的 LICENSE 文件，该情况记录在 SOURCE.json 中。
