# 第 9 步：拉取模型

**产出**：`models/<模型名>/`、`work/reports/9-fetch-model.md`

## 跑

```powershell
cd learn
..\.venv\Scripts\python.exe pipeline\fetch_model.py
```

约 7.7 GB，留 15 GB 磁盘。

## 网络

国内直连 `huggingface.co` 会超时。配置里默认走镜像：

```yaml
fetch:
  mirror: https://hf-mirror.com   # 留空则用官方 hub
```

已经有一份模型时，不用重下：

```powershell
..\.venv\Scripts\python.exe pipeline\fetch_model.py --local-source ..\models\Qwen3-4B-Instruct-2507
```

硬链接，不额外占空间，跨盘才退化成复制。

## 它做什么

| 步 | 动作 |
|---|---|
| 1 | 从第 8 步的 `llamafactory.yaml` 读模型 id，从 `config/catalog.yaml` 读仓库名 |
| 2 | 下载到 `models/<id>/`（或从 `--local-source` 硬链接过来） |
| 3 | 校验：index 里的分片是否都在 → 每个分片的 safetensors 头能否解析 → config 的层数/隐藏维度/词表是否与目录一致 → tokenizer 是否完整 |

脚本会把 `HF_HOME` 指到 `learn/cache/hf`，避免往用户目录写缓存。

## 报告看什么

| 项 | 要判断的 |
|---|---|
| 大小 | 4B 的 4bit 权重文件约 7.7 GB（bf16） |
| 分片数 / 张量数 | 3 个分片、几百个张量 |
| 耗时 | 判断网络好不好 |
| 剩余磁盘 | 别低于 20 GB，第 12 步转 GGUF 还要写一份 |
| 校验明细 | 五行必须全绿 |

## 闸门

| 闸门 | 标准 |
|---|---|
| 分片全部存在 | index 列出的一个不缺 |
| 分片可解析 | 每个 safetensors 头都能读出来 |
| tokenizer 完整 | 至少有 `tokenizer.json` 或 `vocab.json`+`merges.txt` |
| config 与目录一致 | 层数/隐藏维度/词表和目录登记的一致 |

## 出问题怎么办

| 现象 | 处理 |
|---|---|
| 下载超时 / 连接失败 | `--mirror https://hf-mirror.com` 显式指定，或换网络 |
| 小文件能下、大分片卡住不动 | 镜像对大文件不响应。已有一份模型就用 `--local-source`；没有就换时间段重跑（会续传） |
| `PermissionError` 写缓存 | 脚本已自动把 `HF_HOME` 指向工作区；手动跑时记得也设 |
| 分片缺失 | 原命令重跑，会自动续传 |
| 分片可解析 ❌ | 删掉那个分片重下 |
| config 与目录不一致 | 下错模型了，用 `--model-id` 指定正确 id |
| 磁盘不够 | 换盘：`--dest-root E:\models` |

**为什么校验 safetensors 头**：文件在 ≠ 文件完整。下载中断会留下大小不对的半截文件，读头部 1 秒就能判出来，不用等训练时炸。

## 下一步

第 10 步：训练（`pipeline/train.py`）
