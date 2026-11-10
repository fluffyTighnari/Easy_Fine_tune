# 微调一个「像某个人」的本地模型

从一份聊天记录开始，到 `ollama run` 结束。

## 现在做到第几步

**13 / 13 全通**。另做过一次「从零开始」的演练：只准备 `chat.txt` 和配置，用默认路径
跑完 13 步，中途没改过任何路径参数。

## 仓库里放什么

只放框架：`docs/` `lib/` `pipeline/` `prompts/` `config/` 和两份说明。
聊天记录、模型、训练产物、报告、一次性脚本**都不进版本库**（见 `.gitignore`）。
所以你 clone 下来是个空壳，要自己准备 `chat.txt`。

## 你要准备什么

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

## 路径是怎么定的

每个脚本都有 `--root`，默认 `work/`，所有输入输出都在它下面：

```
work/
  input/chat.txt     原始聊天记录
  data/              中间 jsonl
  sft/               训练样本
  generated/         生成出来的训练配置
  reports/           每一步的报告
  adapter/           训练输出
  adapter_best/      扫描选出的最优存档
  gguf/              导出产物
  deploy/            Modelfile
```

想把某次实验单独放：`--root test` 就是 `test/` 下同一套结构。

## 通用策略（换人、换机器都适用）

**参数全部推导，不手调**，规则集中在 `lib/params.py`：

| 原则 | 做法 | 反例 |
|---|---|---|
| 学习率按参数量缩放 | `5e-5 × sqrt(4B/参数量) × sqrt(16/rank)` | 固定 2e-4 → 4B 模型能力坍塌 |
| 轮数由总步数定 | 凑够 600 步，夹在 1~5 轮 | 固定 2 轮 → 小数据集学不到形状 |
| 不碰 MLP 和 o_proj | 只训 `q_proj,k_proj,v_proj` | `lora_target: all` → 无视输入、随机输出 |

**内容一致性是硬约束**，三层保证：

| 层 | 机制 |
|---|---|
| 训练 | 系统提示固定写进每条样本：只接话、不清楚就问、一次一两句 |
| 评测 | 「贴题率」闸门：两个下限（真人 −5pp、原模型 −20pp） |
| 选档 | checkpoint 扫描综合分里贴题占 45% |

像不像可以慢慢追，接不接得上话不能丢。专业问答（编程、知识）不在这个框架的考核范围。

## 本次结果

| 指标 | 真实 | 原模型 | 微调后 |
|---|---|---|---|
| 平均字数 | 20 | 34 | 34 |
| 字数中位 | 19 | 27 | 19 |
| 平均句数 | 1.8 | 2.4 | 1.8 |
| 口头禅密度 | 5.3/百字 | 1.4/百字 | 3.7/百字 |
| 贴题率 | 85% | 100% | 87% |

部署在 Ollama：`ollama run persona-a`。

## 命令怎么跑

```powershell
cd learn
$env:PYTHONIOENCODING="utf-8"; $env:PYTHONUTF8="1"
..\.venv\Scripts\python.exe pipeline\<脚本>.py
```

## 计划与决策

见 `PLAN.md`。
