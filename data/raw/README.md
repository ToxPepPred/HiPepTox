# 多肽毒性总数据集

这个目录用于集中存放当前工作区中各项目自带的多肽毒性相关数据集文件。

## 目录规则

- 按项目分目录保存，尽量保留原项目中的相对路径结构。
- 原项目中的数据路径已经改为软链接，仍可继续被原脚本访问。
- 详细来源见 `来源说明.csv`。

## 当前已整理内容

- `CAPTP/data/` 中的 `train_data.txt`、`test1.txt`、`test2.txt`
- `StrucToxNet/data/` 全部数据文件
- `ToxMSRC/Raw data/` 与 `ToxMSRC/Example data/` 全部数据文件
- `ToxiPep/Dataset/` 全部数据文件
- `tAMPer/data/` 下的序列与结构数据

## 说明

- `CAPTP/data/AAindex.pkl` 和 `CAPTP/data/model_saved.pkl` 未移动，因为它们分别属于特征索引和模型权重，不属于数据集本体。
- 各项目的来源仓库已在 `来源说明.csv` 中逐项标注。
