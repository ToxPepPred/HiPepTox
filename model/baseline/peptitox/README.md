# PeptiTox baseline

这里保存原服务器上用于比较实验的本地 PeptiTox 实现。模型、几何特征、平均池化和分类头沿用原代码；这是现有本地实现的接入，不将它标记为已核实的官方最新版。

## 文件

- `PeptiTox.py`：模型实现。
- `ProteinGraphDataset.py`：原始图数据集。
- `train_cached.py`：项目缓存入口，接受显式内部/外部 PDB 缓存。
- `train_cached_external.py`：原历史缓存训练脚本，保留供追溯。
- `main.py`：原始训练入口。
- `structure2graph.py`、`preprocess.py`、`run_esm.py`、`run_esmfold.py`：原结构、化学及 PLM 准备脚本。

原 `main.py` 和准备脚本期待其原有 `Data/` 目录布局。现有 Paper 数据和缓存应通过 `train_cached.py` 显式传入；原准备脚本可能下载模型或生成文件，不在包导入时自动执行。

## 环境与运行

缓存入口需要 PyTorch、PyG、torch-scatter、torch-cluster、NumPy 和 pandas，依赖列在 `requirements.txt`。使用与 PyTorch/CUDA 匹配的 PyG 扩展；缓存入口已在服务器的 `StrucToxNet` 环境完成 CPU 兼容性检查：Python 3.8.20、PyTorch 2.4.1+cu121、PyG 2.6.1、torch-scatter 2.1.2+pt24cu121、torch-cluster 1.6.3+pt24cu121。该独立基线入口可以使用此环境；整个 Paper 包的安装环境要求见顶层 README。现有 `peptitox` 环境缺少 torch-scatter 和 torch-cluster。原始 ESM、ESMFold 或 RDKit 准备流程另需各自依赖。

```bash
source /home/cns/miniconda3/etc/profile.d/conda.sh
conda activate StrucToxNet
cd /home/cns/HiPepTox/Paper
python model/baseline/peptitox/train_cached.py --help
```

CD-HIT 0.9 的缓存训练示例：

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

CD-HIT 0.8 使用对应的 `toxipep08` 数据、PLM 文件及结构缓存。默认最大 100 epoch、patience 20，按内部验证 MCC 选模型；外部集在选模后评估，分类阈值为 0.5。示例用于新运行，接入代码时没有重新训练。

## 配置文件复现

`configs/` 中包含六份 YAML，覆盖 CD-HIT 0.8、0.9 和 seed 42、2024、3407。参数从对应历史 `best_model.pth` 的 `args` 提取；PLM 维度为 640，模型节点/边输入维度为 824/450。

```bash
cd /home/cns/HiPepTox/Paper
python model/baseline/peptitox/train_cached.py \
  --config model/baseline/peptitox/configs/cdhit09_seed42.yaml --device cpu
```

命令行可覆盖配置，例如末尾加 `--device cpu`。使用配置时，相对输入/输出路径均以 Paper 为根目录；新结果写入 `results/peptitox/<dataset>/seed<seed>`。配置保留训练参数及模型输入维度，每次运行按内部验证 MCC 选模。具体文件见 [配置说明](configs/README.md)。

本地源码目录没有找到上游 LICENSE 文件；本项目许可证不代替这份第三方实现的许可来源记录。
