# 第 2 步：环境体检

**产出**：`work/reports/2-doctor.md`

**先做一件事**：把 `config/persona.example.yaml` 复制成 `config/persona.yaml`，
填好 `persona.name`（要和 chat.txt 里的说话人一字不差）和 `self_names`。后面 13 步都读它。

## 跑

```powershell
cd learn
..\.venv\Scripts\python.exe pipeline\doctor.py
```

加 `--json` 额外存一份 `work/reports/2-doctor.json`。

## 报告里看这四个

| 项 | 要求 |
|---|---|
| 档位 | T0 / T1 / T2 / T3 |
| 显存可用 | 训练前实测，不是标称值 |
| 显存预算 | `可用 × 0.85 − 0.6`，第 8 步用它卡模型 |
| 可行路径 | 这台机器能训多大的模型 |

## 档位对照

| 档位 | 条件 | 能训 |
|---|---|---|
| T3 | VRAM ≥ 15 GB | 4B / 8B QLoRA |
| T2 | VRAM 7~15 GB | 1.7B / 4B QLoRA |
| T1 | VRAM 3.5~7 GB | 0.6B / 1.7B QLoRA |
| T0 | 无可用 GPU | 不训 |

## 出问题怎么办

| 现象 | 处理 |
|---|---|
| 档位 T0 | 两条路：租显卡跑第 10 步；或跳过训练，只用 persona prompt |
| 某个依赖 ❌ | `..\.venv\Scripts\python.exe -m pip install <包名>` |
| 虚拟环境 = **否** | 用了错的 python，改用 `..\.venv\Scripts\python.exe` |
| `models/` 不存在 | 正常，第 9 步会创建 |
| ⚠️ 可用内存 < 3 GB | 关掉占内存的程序；第 12 步合并模型要 2.5 GB 以上 |
| Ollama 未运行 | 正常，第 13 步才需要 |

## 下一步

第 3 步：读入与体检（`pipeline/intake.py`）
