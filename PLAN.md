# 计划

任何人，用任何一份聊天记录，在任何一台能训的机器上，按顺序跑完 13 步，得到一个能 `ollama run`、说话像那个人的本地模型。

13 / 13 全通。模型已部署：`ollama run persona-a`。

## 可复用 = 三条

| 维度 | 判据 |
|---|---|
| 任何人 | 没碰过命令行的人只读 `docs/`，能一路跑完（已演练，见下） |
| 任何信息 | 换一份来源完全不同的材料，只改 `config/persona.yaml` 和 `work/input/chat.txt` |
| 任何机器 | 4GB / 8GB / Apple Silicon / 纯 CPU，体检步都能给出可行路径 |

## 目录

```
learn/
├── README.md  PLAN.md
├── docs/         13 份步骤说明，一步一个文件，只有怎么做
├── prompts/      微信导出怎么变成 chat.txt
├── lib/          params.py（超参推导） paths.py（路径） chat_txt.py（格式） hardware.py
├── pipeline/     13 个脚本 + _train_child.py + merge_lora.py
├── config/       persona.example.yaml · catalog.yaml
├── models/       第 9 步下载的基座
├── work/         新用户的产出根目录（第一次运行时自动建）
└── test/         本地实验记录：数据、适配器、报告、GGUF、一次性脚本（不进库）
```

规则：

1. `pipeline/` 里不许出现 `test` 字面量——路径全部从 `--root` 推，默认 `work/`
2. 一次性脚本（微信导出转换、路径迁移、演练）只放 `test/tools/`，不进框架
3. `lib/params.py` 是超参的唯一出处，别处不许写死数字

## 从零演练

`test/tools/trace_fresh.ps1` 模拟一个完全不懂的人：只准备 `chat.txt` 和
`config/persona.yaml`，然后用默认路径跑 13 步。（这两个文件都是本地实验记录，
不进版本库。）

结果：**13 步全过**，总耗时约 7 分钟（训练只跑 20 步做路径验证）。

发现并修掉的问题：

| 问题 | 修法 |
|---|---|
| 12 个脚本的默认路径全写死在 `test/out/...` | 统一成 `--root`，默认 `work/`（`lib/paths.py`） |
| `select_model` 生成的 `output_dir` 写死 | 改成从 `--root` 推 |
| `fetch_model` 没有 `--reports` 参数却在用 | 补参数 |
| 第 12 步默认要 `adapter_best`，没扫描就卡住 | 缺失时自动退回 `adapter/` |
| `--work` 和根目录重名成 `work/work` | 中间产物目录改叫 `data/` |

## 步骤

| # | 做什么 | 命令 | 产出 | 状态 |
|---|---|---|---|---|
| 1 | 拿到 `chat.txt` | 人工 / AI | `work/input/chat.txt` | ✅ |
| 2 | 环境体检：档位与可行路径 | `pipeline/doctor.py` | `work/reports/2-doctor.md` | ✅ |
| 3 | 读入与体检：条数、说话人、长度 | `pipeline/intake.py` | `work/reports/3-intake.md` | ✅ |
| 4 | 规范化：时间排序、系统消息、引用块 | `pipeline/normalize.py` | `work/data/messages.jsonl` | ✅ |
| 5 | 说话人归因、会话切分 | `pipeline/attribute.py` | `work/data/attributed.jsonl` | ✅ |
| 6 | 清洗：AI 粘贴、跨语言、超长、隐私、重发 | `pipeline/clean.py` | `work/data/clean.jsonl` | ✅ |
| 7 | 造训练样本、切分 | `pipeline/build_sft.py` | `work/sft/{train,val}.jsonl` | ✅ |
| 8 | 选模型、算参数 | `pipeline/select_model.py` | `work/generated/llamafactory.yaml` | ✅ |
| 9 | 拉取模型 | `pipeline/fetch_model.py` | `models/<name>/` | ✅ |
| 10 | 训练 | `pipeline/train.py` | `work/adapter/` | ✅ |
| 11 | 评测 | `pipeline/eval.py` | `work/reports/11-eval.md` | ✅ |
| 12 | 合并导出 GGUF | `pipeline/export_gguf.py` | `work/gguf/*.gguf` | ✅ |
| 13 | 部署到 Ollama | `pipeline/deploy_ollama.py` | `work/deploy/Modelfile` + ollama model | ✅ |

每一步的指导写在 `docs/stepN-*.md`，做到哪步写哪步。

## 每步的闸门

不达标就停，不往下走。

| # | 闸门 |
|---|---|
| 1 | 解析失败率 ≤ 5%；目标人物占比 ≥ 10% |
| 2 | 给出明确档位与可行路径 |
| 3 | 解析失败率 ≤ 5%；说话人 ≥ 2；媒体占位符 0 |
| 4 | 时间全部升序；剥离 + 结构不完整 = 含分隔符总数 |
| 5 | `me` 占比 10%~90%；`me` 条数 ≥ 300 |
| 6 | `me` 可训练 ≥ 80%；手机号与脱敏占位符清零 |
| 7 | 样本 ≥ 300；每条含 system 与人类发言；train/val 时间不交叉 |
| 8 | 显存估算 × 系数 ≤ 预算；每个参数有公式；能转 GGUF |
| 9 | index 分片全在；safetensors 头可解析；config 与目录一致 |
| 10 | 冒烟不 OOM；loss 下降无 NaN；适配器存盘 |
| 11 | val loss 降 ≥ 0.5；`train − val < 0.8`；盲测 ≥ 3/5 分不出 |
| 12 | LoRA 全部对上（applied == total_lora）；f16 ≈ 8 GB；Q4_K_M ≈ 2.5 GB |
| 13 | 5 句冒烟全非空；平均 ≤ 60 字；没有长篇大论 |

## 本次试验

| 项 | 值 |
|---|---|
| 目标人物 | 人物A |
| 输入 | 14,752 条微信导出 → 保留 13,840 条 |
| 模型 | Qwen3-4B-Instruct-2507（第 8 步选出，4bit QLoRA） |
| 产物 | 全部在 `test/`（根目录 `--root test`） |

当前最优：`test/adapter_best/`（checkpoint-240，3 轮全量训练后扫描选出）
- 长度 28 字（真实 18 / 原模型 29）✅
- 句数 1.5（真实 1.8）✅
- 口头禅 4.1/百字（真实 4.1）✅
- 贴题 87%（真实 85% / 原模型 100%）✅
- 自我重复 8% ✅

评测用 40 条样本，裁判问法经过校准（详见 `docs/step11-eval.md`）。

超参对照（同一份 1,727 条训练样本，同一验证集 192 条）：

| 试验 | lr | 目标模块 | 轮 | 步 | eval min | 均字数 | 句数 | 口头禅 |
|---|---|---|---|---|---|---|---|---|
| A | 2e-4 | all | 1（早停 0.29） | 60 | 4.214 | 19 | 1.6 | 1.8 |
| B | 1.5e-5 | q/k/v | 2 | 418 | 4.350 | 49 | 4.3 | 1.4 |
| C | 5e-5 | q/k/v | 2 | 418 | 4.240 | 57 | 4.7 | 5.3 |
| D | 5e-5 | q/k/v | 4 | 836 | **4.221** | 50 | **2.8** | **4.8** |
| 真实 | - | - | - | - | - | 22 | 1.7 | 4.9 |
| 原模型 | - | - | - | - | - | 133 | 5.6 | 1.5 |

## 决策记录

| 日期 | 决策 | 放弃的 |
|---|---|---|
| 2026-11-08 | 起始格式 = 一行一条 `说话人：内容`，时间戳可选 | 直接用微信 JSON |
| 2026-11-08 | 媒体消息整条丢弃，不留占位符；`[晕]` 与 emoji 保留 | 留 `<image>` 占位符 |
| 2026-11-08 | 一次性转换器放 `test/tools/`，框架只留指导与解析实现 | `adapters/` 目录 |
| 2026-11-08 | 长度为按单条消息统计，不按合并轮 | 合并统计 |
| 2026-11-08 | 模型不预设，由第 8 步按机器和数据算 | 写死 4B |
| 2026-11-08 | 清洗只标不删；唯一删除是同会话 60 秒内的重发 | 全局去重 |
| 2026-11-08 | 英文不当污染，只按长度排除书信体 | 按语言删 |
| 2026-11-08 | AI 粘贴用结构化打分，不用关键词 | 关键词黑名单 |
| 2026-11-09 | 显存公式的 logits 按 8 字节/元素算（交叉熵要 fp32 + 梯度） | 按 2 字节估 |
| 2026-11-09 | 训练每次尝试开新进程，回避 arrow 缓存 mmap 冲突 | 同进程重试 |
| 2026-11-09 | 默认自动续训：`overwrite_output_dir: false` + 启动时探测 checkpoint | 每次重跑从头来 |
| 2026-11-09 | 存档间隔 = 总步数 1/10（最少 20 步），只留最近 2 个 | 固定 200 步 |
| 2026-11-09 | `load_best_model_at_end` + 早停 patience 2，按 eval_loss 挑权重 | 存最后一轮 |
| 2026-11-09 | 学习率按参数量定在 1e-5 量级（≤2B 2.5e-5 / ≤5B 1.5e-5 / 更大 1e-5） | 按 LoRA rank 给 1e-4 |
| 2026-11-09 | `lora_target` 只训 q/k/v，不碰 o_proj 和 MLP | 全部线性层 |
| 2026-11-09 | eval_loss 在 4.21~4.35 之间就看不出差别，用生成指标（字数/句数/口头禅）选方案 | 只比 eval_loss |
| 2026-11-09 | 超参统一由 `lib/params.py` 推：lr 按参数量 sqrt 缩放、轮数按 600 步凑 | 各处写死数字 |
| 2026-11-09 | 「贴题」用基座模型当裁判，不用 n-gram（他本人真实回复 n-gram 只有 0.9%） | 词面重合度 |
| 2026-11-09 | 裁判问「有没有跑题/说反话」，不问「有没有回应」——后者把他判成 31% | 问「在回应吗」 |
| 2026-11-09 | 贴题闸门设两个下限：真人 −5pp、原模型 −20pp | 单一阈值 |
| 2026-11-09 | 目标只取连发的前几条、≤120 字（尾巴多是自言自语） | 整串连发都当目标 |
| 2026-11-09 | 显存公式乘 1.16 校准（两次实测都偏低 16%） | 用未校准公式自动选 batch |
| 2026-11-09 | 训练跑满 2 轮，靠 `load_best_model_at_end` 兜底 | 早停到 0.29 轮 |
| 2026-11-09 | 口头禅按「他说/对方说 ≥ 2.5 倍」对比筛，密度按每百字算 | 高频词表 + 命中率 |
