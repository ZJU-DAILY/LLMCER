# 实验执行手册 —— Issue #3：Batch 内任务顺序实验

> 给实验人员：本手册说明如何一键跑完所有数据集、看哪些指标、得到结论后该替换论文里的哪张表。

---

## 0. 背景（为什么要做这个实验）

审稿人指出两点（都已核实属实）：

1. 论文 **§7.8** 研究的是「**单个 record set 内部，记录的排序**」（Similarity/Weak/Random-Ordered）；**§7.7** 只改变 **batch 的大小**。两者都**没有**回答他真正问的问题：当多个 record set（任务）被打包进**同一个 batch prompt** 时，**这些任务的先后顺序**会不会互相影响。
2. 论文 **Table 18 的 "Similarity-Ordered" 列**和 **Table 17 的 "w/o batching" 列**数字**逐位完全相同**（9 个数据集全中），等于把"非批处理"的基线换标签当成了"batch 组织策略"的结果。

本实验**专门**回答任务顺序问题，与 §7.8、§7.7 正交。

---

## 1. 准备（一次性）

1. 确认虚拟环境和依赖已装好（项目根目录）：
   ```powershell
   .\.venv\Scripts\python.exe -c "import numpy, sklearn, sentence_transformers, openai; print('ok')"
   ```
2. 确认 SBERT 权重在本地：`all-MiniLM-L6-v2/` 文件夹存在（已确认）。
3. **配置 API key**：编辑项目根目录的 `.env` 文件（不会被 git 提交）：
   ```
   OPENAI_API_KEY=sk-你的有效key
   OPENAI_MODEL=gpt-4o-mini
   # 如果用的是中转/网关而非官方 openai.com，再加一行：
   # OPENAI_BASE_URL=https://你的网关地址/v1
   ```
4. **先做一次 smoke 测试（不花钱、不调 API）**，确认环境和脚本逻辑正常：
   ```powershell
   .\.venv\Scripts\python.exe issue_experiments\batch_ordering.py --mock --all
   ```
   预期：每个数据集的 `ARI_stab = 1.0000`、`ACC std = 0.0000`（mock 的聚类与顺序无关，必然如此）。这一步只验证脚本管线没问题。

---

## 2. 正式实验（调用真实 LLM，会花 API 费用）

**一条命令跑完全部 7 个数据集：**

```powershell
.\.venv\Scripts\python.exe issue_experiments\batch_ordering.py --all
```

可选参数（一般用默认即可）：
- `--tasks 3`   每个 batch 里放几个任务（record set），默认 3
- `--records 60` 每个数据集采样多少条记录组成 block，默认 60（控制 API 成本）
- `--perms 5`   随机打乱多少种任务顺序，默认 5（加上 similarity + reverse 共 7 种）

只跑单个数据集：
```powershell
.\.venv\Scripts\python.exe issue_experiments\batch_ordering.py --dataset cora
```

**成本估算**：每个数据集 = (2 + perms) 个 batch prompt ≈ 7 次 API 调用，7 个数据集 ≈ 50 次调用，gpt-4o-mini 下成本很低（几美分量级）。

---

## 3. 看哪里的结果

所有输出落在一个带时间戳的文件夹里：
```
issue_experiments/results/batch_ordering/run_real_<时间戳>/
    ├─ cora.log            每个数据集的完整 trace（每种顺序的 ACC/FP）
    ├─ citeseer.log
    ├─ ...（7 个数据集各一个）
    ├─ summary.csv         机器可读：每数据集一行
    └─ summary.txt         ★ 人读的汇总表，直接看这个 ★
```

`summary.txt` 长这样（示例）：
```
Dataset       ACC mean  ACC std  FP mean  ARI_stab   flip
cora            0.xxxx   0.xxxx   0.xxxx    0.xxxx  0.xxxx
...
```

**每个指标的含义：**

| 指标 | 含义 | 怎么解读 |
|---|---|---|
| **ARI_stab** | 不同任务顺序产出的聚类之间的平均成对 ARI | **最关键**。=1.0 表示无论怎么换任务顺序，聚类结果完全一样 → 任务顺序无影响；<1.0 的程度 = 顺序影响有多大 |
| **ACC std** | ACC 在各顺序下的标准差 | ~0 → 顺序不影响准确率 |
| **flip** | 记录对的「同簇/异簇」状态随顺序改变的比例 | ~0 → 顺序几乎不改变聚类结构 |
| ACC mean / FP mean | 各顺序下的平均准确率 / FP-measure | 作为整体效果参考 |

---

## 4. 怎么下结论 + 替换哪张表

跑完看 `summary.txt`，按 **ARI_stab** 和 **ACC std** 判断：

### 情况 A：ARI_stab ≈ 1.0 且 ACC std ≈ 0（预期最可能）
**结论**：batch 内任务顺序对结果**没有显著影响**，模型不会因任务先后而互相干扰 → 这是一个**鲁棒性**结论。

→ **论文动作**：
1. **删除 Table 18 中重复的 "Similarity-Ordered" 列**（它本来就是 Table 17 "w/o batching" 的复制）。
2. 用本实验的结果**新增一张表 / 一段说明**，报告"任务顺序 → ACC/ARI_stab"，说明顺序无关。
3. 在正文里把"ordering 研究"的措辞**明确限定为 record set 内部排序（§7.8）**，并新增一句：batch 内任务顺序经实验验证影响可忽略（引用本表）。

### 情况 B：ARI_stab 明显 < 1.0 或 ACC std 较大
**结论**：batch 内任务顺序**确实有影响**。

→ **论文动作**：
1. 同样**删除 Table 18 重复列**。
2. **如实报告**本实验的方差数字，新增一张「任务顺序 → 性能」表，并在正文讨论这个 ordering effect（哪种顺序更好、为什么）。

### 无论 A 还是 B —— 必须做的诚实修正
**Table 18 的 "Similarity-Ordered" 列必须改**：它与 Table 17 "w/o batching" 逐位相同，审稿人有铁证。要么用本实验的真实数字替换，要么承认是表格构造错误并移除该列。**不要**把那列当成真实测量去辩护。

---

## 5. 要替换/改动的论文表格清单

| 论文位置 | 问题 | 动作 |
|---|---|---|
| **Table 18**（batch organization strategies） | "Similarity-Ordered" 列 = Table 17 "w/o batching" 列，重复 | 删除该重复列；如做了本实验，用真实「任务顺序」结果重建此表 |
| **§7.8 正文** | ordering 措辞含糊，未区分"set 内记录顺序" vs "batch 内任务顺序" | 明确限定为 set 内记录顺序 |
| **新增**（建议放在 §7.7/§7.8 附近） | 缺 batch 内任务顺序的实验 | 用本实验 `summary.txt` 的结果新增一张表 + 一段说明 |

---

## 6. 跑完后要回传给我们的东西

把整个 `issue_experiments/results/batch_ordering/run_real_<时间戳>/` 文件夹回传（含所有 `.log` + `summary.csv` + `summary.txt`）。我们据此：
- 填好给审稿人的最终回复（草稿在 `ISSUE_3_PLAN.md` 第 4 节）
- 生成替换 Table 18 的新表格

---

## 7. 常见问题

- **报 401 / invalid_api_key**：key 无效或复制错了，或它属于中转平台。换有效 key，或在 `.env` 里设 `OPENAI_BASE_URL`。先跑 `issue_experiments/check_api.py` 单独验证 key。
- **某数据集报 "block too small"**：把 `--records` 调大（如 `--records 100`）。
- **想更稳的统计**：把 `--perms` 调大（如 `--perms 10`），多几种随机顺序。
- **smoke 测试 ARI_stab 不是 1.0**：说明环境/脚本有问题，先解决再跑真实实验。
