# ToxiPep 代表性与全面性分析方案

## 1. 目标

这部分分析的目标，不建议直接表述为“`ToxiPep` 绝对优于所有现有数据集”，而应更稳妥地表述为：

> `ToxiPep` 在数据规模、类别平衡、冗余控制、独立评测设计以及序列空间覆盖方面，更适合作为多肽毒性预测的统一 benchmark 与方法开发基座。

这样写更容易被审稿人接受，因为“更好”本身是多维概念，而“更具有代表性、更全面、更适合作为 benchmark”则是可以通过定量证据支撑的。

## 2. 当前本地数据的直接观察

根据 `/home/cns/peptide-aa/总数据集` 中现有文件的初步核查，已经有几条非常关键的事实：

### 2.1 现有多个常用数据集并非真正独立

本地文件显示：

| 数据集对 | 交集情况 |
| --- | --- |
| `CAPTP/train_data.txt` vs `StrucToxNet/train_data.txt` | 完全一致，Jaccard = `1.0000` |
| `CAPTP/train_data.txt` vs `ToxMSRC/train_data.fasta` | 完全一致，Jaccard = `1.0000` |
| `CAPTP/test1.txt` vs `ToxMSRC/test1.fasta` | 完全一致，Jaccard = `1.0000` |
| `CAPTP/test2.txt` vs `ToxMSRC/test2.fasta` | 完全一致，Jaccard = `1.0000` |

这意味着很多“跨数据集比较”在实质上可能只是对同一批序列的重复评测，而不是对不同 benchmark 的真正外推测试。

### 2.2 ToxiPep 具备更完整的分层评测设计

本地统计如下：

| 数据集 | 划分 | 样本数 | 唯一序列数 | 正样本 | 负样本 | 平均长度 | 长度范围 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `ToxiPep` | `0.9_train` | `7056` | `7056` | `3528` | `3528` | `25.37` | `5–50` |
| `ToxiPep` | `0.9_test` | `1766` | `1766` | `883` | `883` | `24.79` | `5–50` |
| `ToxiPep` | `0.8_train` | `5544` | `5544` | `2772` | `2772` | `25.51` | `5–50` |
| `ToxiPep` | `0.8_test` | `1388` | `1388` | `694` | `694` | `25.08` | `5–50` |
| `ToxiPep` | `independent` | `958` | `958` | `479` | `479` | `21.38` | `4–49` |

相比之下：

| 数据集 | 划分 | 样本数 | 唯一序列数 | 正样本 | 负样本 | 平均长度 | 长度范围 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `CAPTP` | train | `6387` | `6387` | `1818` | `4569` | `31.63` | `11–50` |
| `CAPTP` | test1 | `1126` | `1126` | `320` | `806` | `31.56` | `11–50` |
| `CAPTP` | test2 | `582` | `582` | `46` | `536` | `29.90` | `11–50` |
| `StrucToxNet` | train | `6387` | `6387` | `1818` | `4569` | `31.63` | `11–50` |
| `ToxMSRC` | train | `6387` | `6387` | `1818` | `4569` | `31.63` | `11–50` |
| `tAMPer` | train_pos + train_neg | `4801` | `4643` | `1563` | `3238` | 正负分布不平衡 | `5–98` |

由此可见，`ToxiPep` 的直接优势至少包括：

- 正负样本严格平衡。
- 同时提供 `CD-HIT 0.9`、`CD-HIT 0.8` 和独立测试集三层评测。
- 所有划分均为唯一序列，便于公平比较。
- 长度范围覆盖短肽主流研究区间，且与模型任务边界一致。

## 3. 最推荐的证据链

如果你的目标是说服审稿人“`ToxiPep` 更具有代表性和全面性”，最有力的证据链建议按以下五层展开。

### 3.1 数据规模与评测设计层面：证明它更像 benchmark，而不只是一个训练集

这一层最容易做，也最容易说服人。

建议比较以下指标：

- 总样本数
- 唯一序列数
- 正负样本比例
- 是否提供多冗余度划分
- 是否提供独立测试集
- 是否训练集、测试集全部为唯一序列
- 是否存在明显的 train-test leakage

建议做 1 张表：

`Table X | Comparison of benchmark design across peptide-toxicity datasets`

列名可以是：

- Dataset
- Unique sequences
- Positive / Negative ratio
- Length range
- Redundancy-controlled split
- Independent test set
- Exact-overlap control
- Structured metadata availability

这一张表里，`ToxiPep` 很容易处于最完整的位置。

### 3.2 序列空间覆盖层面：证明它更“代表”这个任务，而不是偏向某一类毒肽

这是“代表性”的核心。

建议从序列分布而不是只看样本数：

- 长度分布
- 氨基酸组成分布
- 净电荷分布
- 疏水性分布
- 芳香族残基比例
- `Cys` 比例
- 二肽或三肽 motif 频率
- 低复杂度序列比例

推荐图：

1. 长度分布 ECDF 或 violin plot  
2. 20 种氨基酸频率热图  
3. 理化性质二维投影图，例如 `net charge` vs `hydrophobicity`  
4. 基于 `ESM2` embedding 或 `k-mer TF-IDF` 的 `UMAP/t-SNE` 覆盖图

要证明 `ToxiPep` 更代表性，不是要求它“分布最散”，而是要求它：

- 既覆盖已有数据集的主要区域；
- 又不过度局限于某几个窄簇；
- 在毒性与非毒性两类上都具有较均衡的覆盖。

最关键的量化指标建议用：

- Convex hull / alpha-shape coverage
- kNN coverage
- cluster occupancy
- 每个序列簇中的类别分布均衡性

一句话说，就是比较“谁覆盖了更多真实任务空间”。

### 3.3 冗余与泄漏层面：证明它更公平

这一层非常重要，而且你现在手上的本地数据已经给出了一个很强的切入点。

建议做三种重叠分析：

1. 数据集内部重复率  
2. 训练集与测试集 exact overlap  
3. 跨数据集 overlap 与近同源 overlap

推荐指标：

- Exact overlap ratio
- CD-HIT 0.9 / 0.8 近同源 overlap
- Jaccard similarity
- train-test nearest-neighbor identity

推荐图：

- UpSet plot
- overlap heatmap
- train-test nearest-identity histogram

这一部分的核心论点可以写成：

> 现有多个公开数据集在序列层面存在显著重合，导致不同方法之间的“跨 benchmark 比较”并不完全独立；相比之下，`ToxiPep` 通过分层冗余控制和独立测试集设计，提供了更严格也更公平的评测环境。

### 3.4 跨数据集外推层面：证明在 ToxiPep 上训练出的模型更具有普适性

如果你想把“代表性”做得更硬，必须加这一层。

建议做一个 cross-dataset transfer matrix：

- 在 `ToxiPep` 上训练，在 `CAPTP / StrucToxNet / ToxMSRC / tAMPer` 上测试
- 在 `CAPTP` 上训练，在 `ToxiPep independent` 上测试
- 在 `StrucToxNet` 上训练，在 `ToxiPep independent` 上测试
- 在 `ToxMSRC` 上训练，在 `ToxiPep independent` 上测试

如果出现下面这种结果，就很有说服力：

- 用 `ToxiPep` 训练的模型，在其他数据集上性能下降较小；
- 用旧数据集训练的模型，在 `ToxiPep independent` 上下降明显。

这能说明：

> `ToxiPep` 的训练分布更接近真实任务分布，因此其上训练得到的模型具有更好的跨数据集泛化能力。

推荐图：

- train-on-A, test-on-B 的性能热图
- 以 `MCC` 为主指标，辅以 `AUROC` 和 `Balanced Accuracy`

### 3.5 子群体一致性层面：证明它不是“靠某一类肽撑起来”的

这一层是高级版分析，很适合文章加分。

把所有序列分成不同子群：

- 短肽：`<=10 aa`
- 中短肽：`11–20 aa`
- 中长肽：`21–30 aa`
- 长肽：`31–50 aa`
- 高 `Cys` 肽
- 高正电荷肽
- 高疏水肽
- 富含螺旋倾向残基的肽

对每个子群比较：

- 各数据集样本覆盖数
- 正负样本平衡性
- 模型在该子群上的性能

如果 `ToxiPep` 在大多数子群上都有非稀薄覆盖，而且模型性能更稳定，那么“全面性”这个论点就会更稳。

## 4. 最终建议你主打的三个结论

如果只保留最强的三点，建议主打下面这三个：

### 4.1 `ToxiPep` 更适合作为统一 benchmark

理由：

- 同时提供 `CD-HIT 0.9`、`CD-HIT 0.8` 和独立测试集；
- 各划分均保持严格的正负平衡；
- 所有当前划分均为唯一序列；
- 评测层次比现有公开数据集更完整。

### 4.2 `ToxiPep` 的序列空间覆盖更均衡

理由：

- 不是明显偏向某一种长度、某一种富 `Cys` 模式、或某一种带电类型；
- 在毒性与非毒性两端都形成更均衡的覆盖；
- 能覆盖老数据集中的主流区域，同时保留额外的新区域。

### 4.3 `ToxiPep` 提供了更严格的泛化检验

理由：

- 旧数据集之间存在显著甚至完全重复；
- `ToxiPep` 与这些数据集只有有限重叠；
- 在 `ToxiPep independent` 上测试，更接近真正的 out-of-distribution 检验。

## 5. 不建议直接这样写

下面这些说法风险较高：

- “`ToxiPep` 是目前最好的多肽毒性数据集”
- “`ToxiPep` 完全代表真实世界多肽毒性空间”
- “其他数据集都不可靠”

更稳妥的表述是：

- “`ToxiPep` provides a more comprehensive and better controlled benchmark for peptide-toxicity prediction.”
- “`ToxiPep` offers broader and more balanced sequence-space coverage under stricter redundancy control.”
- “Compared with widely used legacy datasets, `ToxiPep` is better suited for fairer generalization assessment.”

## 6. 建议的论文图表组合

如果你只想做一套最有性价比的分析，建议如下：

### 主文

1. `Table X`  
数据集规模、平衡性、冗余控制与独立测试设计对比

2. `Figure X`  
各数据集的长度分布 + 理化性质分布

3. `Figure Y`  
跨数据集 overlap heatmap

4. `Figure Z`  
cross-dataset transfer performance heatmap

### 补充材料

1. 氨基酸频率热图  
2. `UMAP/t-SNE` 序列空间覆盖图  
3. 子群体统计表  
4. `CD-HIT` 或 nearest-neighbor identity 分析

## 7. 最省事但最有效的执行顺序

建议按下面顺序做，前两步就已经能支撑一版结果：

1. 先做数据集描述统计表  
2. 再做 overlap heatmap  
3. 再做长度与理化性质分布图  
4. 最后做 cross-dataset transfer

如果时间有限，至少把前 1 到 3 步做完，因为这三步已经足以很有力地说明：

- 老 benchmark 之间高度同源；
- `ToxiPep` 的评测设计更完整；
- `ToxiPep` 的类别和序列分布更均衡。

## 8. 下一步我建议直接做什么

如果你要我继续，我建议直接落地成一套可出图的脚本，输出以下内容：

- `dataset_summary_table.csv`
- `dataset_overlap_heatmap.svg`
- `length_distribution.svg`
- `physchem_distribution.svg`
- `cross_dataset_transfer_plan.md`

这样你就不是“有一个想法”，而是已经有一套能直接进文章的分析框架。
