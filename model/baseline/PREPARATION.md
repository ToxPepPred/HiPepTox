# 基线资源准备

本目录保留原有模型实现、CSV 训练/评估入口、原始示例数据及静态输入定义。
发布权重、重训练权重、特征缓存、历史预测/日志以及复现检查目录已移除。
主模型配置见 configs/reproduction；各基线的运行参数通过相应训练入口及配套配置指定。

| 方法 | 入口 | 重建或补充的资源 |
| --- | --- | --- |
| ToxiPep | toxipep/Code/train_eval_csv.py | 从 CSV 重新生成原子/图特征并训练 |
| ToxTeller | toxteller/program_resource/train_eval_csv.py | 从 CSV 重新提取序列特征、拟合 scaler 和分类器 |
| ToxMSRC | toxmsrc/scripts/train_eval_csv_torch.py | 先恢复原 Word2Vec 模型，再运行 build_word2vec_features.py；自行重新训练 Word2Vec 会改变原始表征 |
| PLPTP | plptp/scripts/train_eval_csv.py | 从 plptp/model/model 的原作者说明获取 model.pkl，使用 --init-weights；保留 local_models 中的架构与分词配置 |
| CAPTP | captp/scripts/train_eval_csv.py | AAindex.pkl 是静态 token→编号词表，已保留；可使用 --train-from-scratch，复现历史微调条件则需恢复原发布初始化权重 |
| tAMPer | tamper/scripts/train_tamper_csv.py | 重新准备 PDB；fair-esm 获取 ESM2-t12-35M；复现原初始化时另需通过 --pre-gnn-checkpoint 提供 pre_GNNs.pt |
| ToxGIN | toxgin/scripts/train_toxgin_csv.py | 重新准备 PDB；默认从 facebook/esm2_t36_3B_UR50D 获取语言模型；aaindex1.csv 已保留 |
| PeptiTox | peptitox/train_cached_external.py | 重新准备 PDB/ESM 表征；原提取代码在 run_esm.py、run_esmfold.py、preprocess.py |
| StrucToxNet | structoxnet/scripts/train_csv.py；structoxnet/scripts/evaluate_csv.py | 准备 ProtT5、ESMFold 与 mkdssp；原始 x_min.npy/x_max.npy 已保留，数据与输出通过命令参数传入；详见 structoxnet/README.md |
| ToxinPred3 | toxinpred3/scripts/evaluate_csv.py | 保留的是作者发布的预测/评估实现，不包含完整重训练流程；评估前须从原作者发行包恢复模型权重，参阅其 README.md |

## 输入与输出

共用数据位于项目根目录 data/processed。使用 CSV 中既有 split 列，不重新随机划分测试集。
主模型及基线的训练超参数、特征来源和初始化方式分别保留，删除权重不表示更改这些实验条件。
重新训练生成的权重、图、PLM 表征和预测应写入 outputs/ 或显式指定的新输出目录。
各基线依赖参阅本目录 registry.json 和各模型原有 requirements/environment 文件；无需复制第三方 Python 包源码。

## 结构资源

结构预测代码保留在 peptitox/run_esmfold.py 以及 tamper/local_implementation/src/ESMfolding.py。
后者读取 FASTA，可用 -seqs 指定输入、-out 指定输出 PDB 文件夹。CSV 的 id 和原结构 mapping 列都已保留；
PDB 文件命名须与所调用入口要求的 id、structoxnet_sample_id 或序列相匹配。
45 维主模型结构特征还需要 DSSP 与残基接触记录；可用 scripts/prepare_structure_inputs.py 从 PDB 调用 mkdssp 并生成接触记录，再由 scripts/structure_features.py 提取特征；完整命令见项目根目录 README.md。
结构预测及大模型权重首次获取需要网络与相应计算资源；新预测的结构及训练随机性可能使结果与历史记录有差异。
