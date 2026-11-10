#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""所有超参的唯一出处。其他地方不许再写死数字。

每条规则都要能回答两个问题：为什么是这个量级，换模型/换数据集会不会崩。

## 学习率：按参数量缩放，量级 1e-5 ~ 1e-4

`lr = 5e-5 × sqrt(4B / params) × sqrt(16 / rank)`，夹在 [1e-5, 2e-4]。

依据（同一份 1,727 条样本、同一验证集、Qwen3-4B、rank16）：

| lr | 轮 | 结果 |
|---|---|---|
| 2e-4 + all 模块 | 0.29 | 句数 1.6 但过拟合，eval 从 4.21 回涨到 4.95 |
| 1.5e-5 | 2 | 学不会断句，输出 49 字（真实 22） |
| 5e-5 | 4 | 句数 2.8、口头禅 4.8（真实 4.9），仍然收敛中 |

sqrt 缩放的理由：LoRA 更新幅度 ∝ lr × 层数 × 参数更新的相对比例，大模型的
隐藏维度更大、每层梯度更平滑，可承受的步长按 sqrt 递减，不是线性。

## 轮数：按「总步数」定，不按「几轮」定

目标 600 个优化步（LoRA 风格迁移的经验下限），再夹到 [1, 5] 轮。

步数不够 = 学不到分布形状（1.5e-5 那次 418 步，长度完全没动）；
步数过多 = 只要 lr 不炸，有最优存档兜底，不会更差。

所以轮数是**数据量的函数**，不是常数：样本少就多跑几轮凑步数，样本多见好就收。

## 为什么只训 q/k/v

外层实验：`lora_target: all`（33M 参数）→ 模型无视输入、随机输出训练集段落；
收缩到 `q_proj,v_proj`（2M）→ 风格出来且基座能力还在。中间试过加 `o_proj`，退化明显。
注意力投影负责「怎么组织表达」，MLP 和 o_proj 承载知识，动它们等于改脑子。

## dropout 随数据量走

样本 < 500 用 0.1，否则 0.05。小样本靠正则，大样本靠步数。

## 存档数量

至少留 8 个，因为第 11 步要在 checkpoint 之间扫一遍挑最好的（`eval.py --sweep`）。
选方案不靠调参试错，靠扫存档。
"""
from __future__ import annotations

import math

# 4B 模型、rank 16、约 40 万 token 数据下的实测最优学习率
BASE_LR = 5.0e-5
BASE_PARAMS_B = 4.0
BASE_RANK = 16
LR_MIN, LR_MAX = 1.0e-5, 2.0e-4

TARGET_STEPS = 600       # LoRA 风格迁移的经验总步数
EPOCHS_MIN, EPOCHS_MAX = 1, 5
SAVE_TOTAL_LIMIT = 8     # 留给 --sweep 挑
LORA_TARGET = "q_proj,k_proj,v_proj"


def lora_rank(n_samples: int) -> int:
    """样本少用低秩，避免过参数化；样本多给容量。"""
    if n_samples < 500:
        return 8
    if n_samples < 5000:
        return 16
    return 32


def lora_dropout(n_samples: int) -> float:
    return 0.10 if n_samples < 500 else 0.05


def learning_rate(params_b: float, rank: int) -> float:
    lr = BASE_LR * math.sqrt(BASE_PARAMS_B / max(0.1, params_b)) * math.sqrt(BASE_RANK / rank)
    return float(min(LR_MAX, max(LR_MIN, round(lr, 8))))


def num_epochs(steps_per_epoch: int) -> int:
    if steps_per_epoch <= 0:
        return EPOCHS_MIN
    return int(min(EPOCHS_MAX, max(EPOCHS_MIN, round(TARGET_STEPS / steps_per_epoch))))


def save_steps(total_steps: int) -> int:
    """约每 10% 存一次，最少 20 步一次。"""
    return int(max(20, min(200, round(total_steps / 100) * 10)))


def derive(*, params_b: float, n_samples: int, steps_per_epoch: int,
           bs: int, cutoff_len: int) -> dict:
    rank = lora_rank(n_samples)
    epochs = num_epochs(steps_per_epoch)
    total = int(math.ceil(steps_per_epoch * epochs))
    return {
        "cutoff_len": cutoff_len,
        "per_device_train_batch_size": bs,
        "gradient_accumulation_steps": max(1, math.ceil(8 / bs)),
        "lora_rank": rank,
        "lora_alpha": rank * 2,
        "lora_dropout": lora_dropout(n_samples),
        "lora_target": LORA_TARGET,
        "learning_rate": learning_rate(params_b, rank),
        "num_train_epochs": epochs,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "cosine",
        "save_steps": save_steps(total),
        "save_total_limit": SAVE_TOTAL_LIMIT,
    }


def explain(params_b: float, n_samples: int, steps_per_epoch: int) -> list[tuple[str, str, str]]:
    """给报告用的推导过程：(项, 值, 依据)。"""
    p = derive(params_b=params_b, n_samples=n_samples, steps_per_epoch=steps_per_epoch,
               bs=2, cutoff_len=0)
    total = int(steps_per_epoch * p["num_train_epochs"])
    return [
        ("lora_rank", str(p["lora_rank"]), f"样本 {n_samples:,} 条"),
        ("learning_rate", f"{p['learning_rate']:.1e}",
         f"5e-5 × sqrt(4/{params_b:g}B) × sqrt(16/{p['lora_rank']})"),
        ("num_train_epochs", str(p["num_train_epochs"]),
         f"{steps_per_epoch} 步/轮，凑够 {TARGET_STEPS} 步 → 共 {total} 步"),
        ("lora_target", p["lora_target"], "只训注意力，动了 MLP/o_proj 基座能力会退"),
        ("lora_dropout", str(p["lora_dropout"]), "样本少加大正则"),
        ("save_steps", str(p["save_steps"]), f"总 {total} 步，约每 10% 存一次"),
    ]
