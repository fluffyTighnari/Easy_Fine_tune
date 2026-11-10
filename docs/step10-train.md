# 第 10 步：训练

**产出**：`work/adapter/`、`work/reports/10-train.md`、`work/logs/train_try*.log`

## 先冒烟

```powershell
cd learn
..\.venv\Scripts\python.exe pipeline\train.py --smoke
```

6 步，约 1 分钟。看 `work/reports/10-train-smoke.md` 四个闸门全绿再往下。

## 再全量

```powershell
..\.venv\Scripts\python.exe pipeline\train.py
```

约 432 步，本机 10.5 秒/步 → **75 分钟左右**。中途可以停，加 `--smoke --steps 100` 可以先跑一百步看趋势。

## 断点续训

默认就是**自动续训**：输出目录里有 `checkpoint-N` 就接着练，没有就从头练。

| 参数 | 作用 |
|---|---|
| 不给参数 | 有 checkpoint 就续训，没有就从头训 |
| `--fresh` | 清空输出目录，从头训 |
| `--resume work/adapter/checkpoint-40` | 指定从哪个 checkpoint 续训 |
| `--steps 100` | 限制总步数（配合冒烟/试跑） |
| `--save-steps 20` | 覆盖存档间隔 |

Ctrl+C 中断后**重跑同一条命令**就接着练。存档间隔由第 8 步算：总步数的约 1/10，最少 20 步一次，只保留最近 2 个。

## 早停与最优存档

配置里 `load_best_model_at_end: true` + `metric_for_best_model: eval_loss`：训练会一直记着 eval_loss 最低的那次，结束后把**那次**的权重复原成最终适配器。

`train.py --early-stop 2`（默认）：eval_loss 连续 2 次不降就停，不用等轮数跑完。

默认 patience 2。想跑满全部数据就 `--early-stop 0`：早停只是省时间，最终权重由 `load_best_model_at_end` 决定，练多了也不会存坏的那份。

## 它做什么

| 步 | 动作 |
|---|---|
| 1 | 预检：模型分片齐不齐、数据集在不在、预计步数够不够 |
| 2 | 查输出目录：有 checkpoint → 续训；已训完 → 提示要加 `--fresh` |
| 3 | 按第 8 步的配置开跑（冒烟时改小步数、换输出目录） |
| 4 | 失败自动降档重试，最多 3 次 |
| 5 | 汇总 loss、显存峰值、耗时、checkpoint，写报告 |

## 自动降档

| 失败类型 | 动作 |
|---|---|
| 显存不够（OOM） | batch 减半，梯度累积 ×2（等效 batch 不变） |
| 系统不允许建命名管道 | `preprocessing_num_workers → 1`、`dataloader_num_workers → 0` |
| arrow 缓存被占用（WinError 1224） | 换进程重跑 |

## 为什么单开子进程

`datasets` 会把 arrow 缓存 mmap 住，同一进程里重试再往同一个缓存文件写，Windows 会报 `WinError 1224`。每次尝试开一个新进程，缓存冲突自动消失。

## 报告看什么

| 项 | 要判断的 |
|---|---|
| 步数 / 耗时 / 秒每步 | 估算全量要多久（步数 × 秒/步） |
| 显存峰值 | 分配 + 保留；保留接近设备总量就危险 |
| 起点 | 从头训还是续训 checkpoint-N |
| 现存 checkpoint | 存了几次，间隔对不对 |
| loss 首段 → 末段 | 必须下降 |
| 自动降档记录 | 有记录说明这台机器比预期紧 |
| 日志路径 | 出问题先看这个 |

## 闸门

| 闸门 | 标准 |
|---|---|
| 训练跑完 | 无异常退出 |
| loss 有效 | 无 NaN |
| loss 下降 | 末段平均 < 首段平均 |
| 适配器已存盘 | 有 `adapter_model.safetensors` |

## 本机冒烟结果

```
步数　　　：6　耗时 1.1 分钟（10.5 秒/步）
显存峰值　：5.06 GB 分配 / 5.47 GB 保留（设备 8.55 GB）
loss　　　：6.1870 → 4.2463
适配器　　：8 个文件（126 MB）
断点续训　：8 步 → checkpoint-8 → 续到 16 步，loss 接着降（已验证）
```

第 8 步算出的存档节奏：总 432 步，每 40 步存一次，留最近 2 个。

## 出问题怎么办

| 现象 | 处理 |
|---|---|
| OOM 三次都救不回来 | 回第 8 步用 `--max-batch-size 1` 重算，或调小 `max_context_chars` 重造样本 |
| loss 不下降 | 学习率太大，回第 8 步把 `lora_rank` 调到 8（对应 3e-4 以下）或手工改 yaml 的 lr |
| loss 变 NaN | 同上，并且检查第 7 步样本里有没有空 target |
| 显存保留 > 90% 设备总量 | 能把全量跑完，但建议 batch 减半再来 |
| 跑到一半断了 | 重跑同一命令，自动从最新 checkpoint 续训 |
| 想换个风格重训 | 加 `--fresh`，否则会接着旧的练 |
| 磁盘不够 | 把 `save_total_limit` 改成 1，或 `--save-steps` 调大 |

## 下一步

第 11 步：评测（`pipeline/eval.py`）
