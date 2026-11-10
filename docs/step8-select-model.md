# 第 8 步：选模型、算参数

**产出**：`work/generated/llamafactory.yaml`、`work/sft/dataset_info.json`、`work/reports/8-select-model.md`

## 跑

```powershell
cd learn
..\.venv\Scripts\python.exe pipeline\select_model.py
```

已经下载过某个模型、想离线量长度：加 `--tokenizer ..\models\Qwen3-4B-Instruct-2507`。

## 它怎么选

| 步 | 做什么 |
|---|---|
| 1 硬约束 | 预算 = `可用显存 × 0.85 − 0.6`；逐个模型算 bs=1/2/4/8 哪个放得下 |
| 2 打分 | `3×min(参数/8,1)` + `1×中文/5` + `1×(bs≥2)` + `0.5×可商用` + `0.5×能转GGUF` |
| 3 定长度 | 用**真实 tokenizer** 量全部样本，`cutoff_len = ceil(p99.5 × 1.05)`，上限 `min(ctx, 2048)` |

未实测过的模型乘 1.15 保守系数（公式只在 Qwen3-4B 上校准过）。

## 本机结果

```
选中　　　：Qwen3-4B-Instruct-2507（4.0B）
显存　　　：估 3.82 GB / 预算 5.66 GB（余 1.84）
batch　　　：4 × 累积 2 = 等效 8
cutoff_len：576（token p99.5 492，截断 1 条）
LoRA　　　：rank 16 / alpha 32 / lr 2e-4 / 2 epochs
```

8B / 7B / 14B 全部放不下（8B 在 bs=1 就要 6.19 GB）。

## 参数怎么来的

| 参数 | 值 | 依据 |
|---|---|---|
| `per_device_train_batch_size` | 4 | 预算内最大可行值，上限 4（公式只校准到 bs2） |
| `gradient_accumulation_steps` | 2 | `ceil(8 / 4)` |
| `cutoff_len` | 576 | `ceil(492 × 1.05)`，向上取 64 的倍数 |
| `lora_rank` | 16 | 样本 1,919 条 → 16（<500 用 8，<5000 用 16） |
| `lora_alpha` | 32 | `2 × rank` |
| `learning_rate` | 2e-4 | rank 16 对应值 |
| `num_train_epochs` | 2 | 样本 ≥ 500 |
| `quantization_bit` | 4 | 显存不够 bf16 |
| `template` | `qwen3_nothink` | 防止往 target 里注入思考块 |

## 报告看什么

| 项 | 要判断的 |
|---|---|
| 候选打分表 | 为什么是它，别的差多少 |
| 显存拆解 | 每项多少，余量多少 |
| 会被截断的样本 | 必须是 0 或个位数；多了 target 会被切掉 |
| 参数推导 | 每个值都有公式，能反推 |


## 参数从哪来

全部由 `lib/params.py` 推出来，换模型、换数据量会自动变，不需要手调。

| 参数 | 规则 | 为什么 |
|---|---|---|
| `learning_rate` | `5e-5 × sqrt(4B/参数量) × sqrt(16/rank)` | 大模型梯度更平滑，步长按 sqrt 递减；1e-4 量级会把 4B 练到能力坍塌 |
| `num_train_epochs` | 凑够 600 步，夹在 1~5 轮 | 步数不够学不到分布形状，够了就有最优存档兜底 |
| `lora_target` | `q_proj,k_proj,v_proj` | 加 o_proj/MLP 会改坏基座的知识和推理 |
| `lora_rank` | 样本 < 500 → 8，< 5000 → 16，否则 32 | 数据量与容量匹配 |
| `lora_dropout` | 样本 < 500 → 0.1，否则 0.05 | 小样本靠正则 |
| `save_steps` | 总步数的 1/10，最少 20 | 给第 11 步扫描留够存档 |

想验证某条规则，看 `reports/8-select-model.md` 里的推导表，再对照第 11 步的扫描结果。

## 闸门

| 闸门 | 标准 |
|---|---|
| 显存估算 × 系数 | ≤ 预算 |
| batch size | ≥ 1 |
| 每个参数有据可依 | 全部有公式 |
| 能转 GGUF | 是（否则第 13 步落不了 ollama） |

## 能改的

| 想改什么 | 改哪 |
|---|---|
| 加候选模型 | `config/catalog.yaml` |
| 允许更大 batch | `--max-batch-size 8`（显存波动风险自负） |
| 允许更小 batch | `--min-batch-size 1` |
| 指定 tokenizer | `--tokenizer 路径` |
| model 名字/路径 | 不用改，第 9 步按选中结果下载到 `models/` |

## 出问题怎么办

| 现象 | 处理 |
|---|---|
| 取不到 tokenizer | 加 `--tokenizer 本地目录`，或联网重跑 |
| 所有模型都放不下 | 换机器，或回第 7 步把 `max_context_chars` 调小重造样本 |
| 截断样本很多 | `max_context_chars` 调小，重跑第 7 步 |
| `cutoff_len` 撞到 2048 上限 | 上下文太长，`max_context_pairs` 从 3 调到 2 |

## 下一步

第 9 步：拉取模型（`pipeline/fetch_model.py`）
