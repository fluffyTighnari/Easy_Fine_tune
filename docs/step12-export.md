# 第 12 步：导出 GGUF

把训练好的适配器合进基座，转成 Ollama 能读的格式。

## 一条命令

在 `learn/` 目录下：

```powershell
..\.venv\Scripts\python.exe pipeline\export_gguf.py
```

跑完看 `work/reports/12-export.md`，三件事：合并成功、f16 生成、量化完成。

## 三个阶段分别在干什么

| 阶段 | 命令 | 产出 |
|---|---|---|
| 合并 | `pipeline/merge_lora.py` | `work/merged/`（7.5 GB） |
| 转格式 | `tools/llama/convert_hf_to_gguf.py` | `work/gguf/<模型>-f16.gguf`（8.05 GB） |
| 量化 | `tools/llama/llama-quantize.exe` | `work/gguf/<模型>-Q4_K_M.gguf`（2.50 GB） |

合并是**流式**的：逐张量读、算、写，峰值内存约 2 GB。不要改用
`llamafactory-cli export`——它要把整个模型读进内存，4B 的 bf16 要 8 GB，
小内存机器直接 OOM。

## 换量化档位

```powershell
..\.venv\Scripts\python.exe pipeline\export_gguf.py --quant Q5_K_M
```

Q4_K_M 是默认值，2.5 GB 放进 8 GB 显存很宽松；想更准用 Q5_K_M 或 Q6_K，
体积和内存都跟着涨。f16 别用来部署，8 GB 会把显存吃满。

## 中间产物可以删

`work/merged/` 只在导出时用，删了不影响 Ollama 里已经建好的模型。
`f16.gguf` 同理，只有重新量化时才需要。

## 自检

打开报告，看这三个数：合并 `applied` 等于 `total_lora`（说明 LoRA 全部对上）、
f16 约 8 GB、Q4_K_M 约 2.5 GB。数字差太多就是哪一步没跑完。
