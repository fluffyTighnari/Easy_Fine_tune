# prompt：请你扮演ta！
# fine-tune：你就是ta！

从一份聊天记录开始，到 `ollama run` 结束。
注：需要独显。

## 如何快速开始

只有一样：**聊天记录**。

1. 把微信导出整理成 `chat.txt`（格式见 `docs/step1-chat-txt.md`）
2. 放到 `work/input/chat.txt`
3. 复制 `config/persona.example.yaml` 成 `config/persona.yaml`，改名字和系统提示

然后按 `docs/` 里的顺序，一步一条命令。13 步都不用改代码。

## 目录

```
config/    persona.example.yaml（复制成 persona.yaml 改）  catalog.yaml（候选模型）
docs/      13 份步骤说明，照着做
pipeline/  13 个脚本 + _train_child.py + merge_lora.py
lib/       params.py（超参推导）  paths.py（路径）  chat_txt.py  hardware.py
prompts/   微信导出怎么变成 chat.txt
models/    第 9 步拉下来的基座模型
work/      你的产出，跑第一步时自动建（默认根目录）
test/      本地实验记录（数据、适配器、报告、GGUF），不进版本库
```

## 通用策略

**参数全部推导，不手调**，规则集中在 `lib/params.py`：

| 原则 | 做法 | 反例 |
|---|---|---|
| 学习率按参数量缩放 | `5e-5 × sqrt(4B/参数量) × sqrt(16/rank)` | 固定 2e-4 → 4B 模型能力坍塌 |
| 轮数由总步数定 | 凑够 600 步，夹在 1~5 轮 | 固定 2 轮 → 小数据集学不到形状 |
| 不碰 MLP 和 o_proj | 只训 `q_proj,k_proj,v_proj` | `lora_target: all` → 无视输入、随机输出 |

## 命令怎么跑

```powershell
cd learn
$env:PYTHONIOENCODING="utf-8"; $env:PYTHONUTF8="1"
..\.venv\Scripts\python.exe pipeline\<脚本>.py
```
