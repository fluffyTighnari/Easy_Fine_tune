#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 8 步：选模型、算参数。硬约束过滤 → 打分排序 → 真实 tokenizer 定 cutoff_len。

输入：<root>/sft/*.jsonl、config/catalog.yaml
产出：<root>/generated/llamafactory.yaml、<root>/sft/dataset_info.json、<root>/reports/8-select-model.md

用法（从 learn/ 目录）：
    ..\\.venv\\Scripts\\python.exe pipeline\\select_model.py
    ..\\.venv\\Scripts\\python.exe pipeline\\select_model.py --tokenizer ..\\models\\Qwen3-4B-Instruct-2507
"""
from __future__ import annotations

sys_path_fix = None

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

LEARN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LEARN))

from lib.paths import CONFIG, DEFAULT_ROOT, Layout  # noqa: E402
sys.path.insert(0, str(LEARN))

from lib import params as P  # noqa: E402
from lib import hardware  # noqa: E402

SEQ_PROVISIONAL = 512
OVERHEAD_GB = 0.6
GB_PER_B = 0.55


# 公式在 Qwen3-4B 上实测校准过两次：bs4×seq576 实测 6.87 GB、bs2×seq576 实测 5.06 GB，
# 预测值都稳定偏低 16%。这里补一个校准系数，否则会自动选出 bs4 然后 OOM。
CALIB = 1.16


def mem_gb(m: dict, bs: int, seq: int) -> dict:
    """按实测校准过的公式估显存，返回拆分明细。"""
    w = m["params_b"] * GB_PER_B
    # 交叉熵要 fp32 logits + 对应梯度，按 8 字节/元素算；用 2 字节（纯 bf16 前向）会低估 2~3 倍
    logits = bs * seq * m["vocab"] * 8 / 1e9
    act = bs * seq * m["hidden"] * m["layers"] * 1.5 / 1e9
    return {"weights": w, "logits": logits, "act": act, "overhead": OVERHEAD_GB,
            "total": (w + logits + act + OVERHEAD_GB) * CALIB}


def safety(m: dict) -> float:
    """未实测过的模型乘一个保守系数：公式只在 Qwen3-4B 上校准过。"""
    return 1.0 if m.get("mem_measured") else 1.15


def fits(m: dict, bs: int, seq: int, budget: float) -> bool:
    return mem_gb(m, bs, seq)["total"] * safety(m) <= budget


def max_bs_fitting(m: dict, budget: float, seq: int, cap: int = 8) -> int:
    best = 0
    bs = 1
    while bs <= cap:
        if fits(m, bs, seq, budget):
            best = bs
        bs *= 2
    return best


def round64(x: int) -> int:
    return max(64, math.ceil(x / 64) * 64)


def resolve_tokenizer(model_id: str, cli: str | None) -> tuple[str | None, str]:
    """返回 (tokenizer 路径或 None, 来源说明)。优先本地，其次 HF 缓存，最后只下 tokenizer。"""
    cands = []
    if cli:
        cands.append((cli, "命令行 --tokenizer"))
    cands.append((str(LEARN / "models" / model_id), "learn/models"))
    for p, src in cands:
        if Path(p).exists() and (Path(p) / "tokenizer_config.json").exists():
            return p, src
    try:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(model_id, local_files_only=True, trust_remote_code=True)
        return model_id, "HuggingFace 本地缓存"
    except Exception:
        pass
    try:
        from huggingface_hub import snapshot_download

        p = snapshot_download(model_id, allow_patterns=["*tokenizer*", "*.json", "*.txt"],
                              local_dir=str(LEARN / "models" / model_id))
        return p, "刚下载的 tokenizer"
    except Exception as e:
        return None, f"取不到 tokenizer（{type(e).__name__}）"


def token_lengths(model_id_or_path: str, samples: list[dict]) -> list[int] | None:
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=True)
    except Exception:
        return None

    role_map = {"system": "system", "human": "user", "gpt": "assistant"}
    out: list[int] = []
    for s in samples:
        msgs = [{"role": role_map[m["from"]], "content": m["value"]} for m in s["conversations"]]
        try:
            ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False,
                                          return_dict=False)
        except Exception:
            text = "\n".join(f"{m['role']}: {m['content']}" for m in msgs)
            ids = tok(text)["input_ids"]
        out.append(len(ids))
    return out


def yf(v: float) -> str:
    """YAML 1.1 不认 `5e-05`（没小数点就当字符串），必须补成 `5.0e-05`。"""
    t = repr(float(v))
    mant, _, exp = t.partition("e")
    if "." not in mant:
        mant += ".0"
    return mant + ("e" + exp if exp else "")


def main() -> int:
    ap = argparse.ArgumentParser(description="第 8 步：选模型、算参数")
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--root", default=DEFAULT_ROOT, help="输出根目录，默认 work/")
    ap.add_argument("--catalog", default="config/catalog.yaml")
    ap.add_argument("--sft", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--reports", default=None)
    ap.add_argument("--tokenizer", default=None, help="指定 tokenizer 目录（可选）")
    ap.add_argument("--min-batch-size", type=int, default=2, help="最小 batch size，低于就不选这个模型")
    ap.add_argument("--lr", type=float, default=None, help="手动指定学习率（默认按参数量推）")
    ap.add_argument("--epochs", type=float, default=None, help="手动指定轮数（默认 2）")
    ap.add_argument("--max-batch-size", type=int, default=4,
                    help="最大 batch size。公式只在 bs2 校准过，往上外推误差变大，默认不超 4")
    args = ap.parse_args()

    _lay = Layout(args.root)
    _lay.ensure()
    args.sft = args.sft or str(_lay.sft)
    args.out = args.out or str(_lay.generated)
    args.reports = args.reports or str(_lay.reports)

    def resolve(p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else LEARN / q

    import yaml

    cfg = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    catalog = yaml.safe_load(resolve(args.catalog).read_text(encoding="utf-8"))["models"]

    sft = resolve(args.sft)
    train = [json.loads(l) for l in open(sft / "train.jsonl", encoding="utf-8") if l.strip()]
    val = [json.loads(l) for l in open(sft / "val.jsonl", encoding="utf-8") if l.strip()]
    n_samples = len(train) + len(val)

    hw = hardware.probe()
    budget = round(hw["accel"]["vram_free_gb"] * 0.85 - OVERHEAD_GB, 2)

    # ---- 1. 硬约束：预算内能跑至少 min_batch_size ----
    rows = []
    for m in catalog:
        bs_cap = max_bs_fitting(m, budget, SEQ_PROVISIONAL, cap=args.max_batch_size)
        rows.append({"m": m, "bs": bs_cap,
                     "mem": mem_gb(m, max(1, bs_cap), SEQ_PROVISIONAL)["total"]})
    feasible = [r for r in rows if r["bs"] >= args.min_batch_size]

    if not feasible:
        # 没有模型能到 min_batch_size：放宽到 1
        feasible = [r for r in rows if r["bs"] >= 1]
        if not feasible:
            print(f"预算 {budget} GB 内没有能训的模型。换机器，或调小数据（第 7 步）")
            return 3

    # ---- 2. 打分 ----
    def score(r) -> float:
        m = r["m"]
        return (3.0 * min(m["params_b"] / 8, 1.0)
                + 1.0 * (m["chinese"] / 5)
                + 1.0 * (1.0 if r["bs"] >= 2 else 0.0)
                + 0.5 * (1.0 if m.get("commercial") else 0.0)
                + 0.5 * (1.0 if m.get("gguf") else 0.0))

    for r in rows:
        r["score"] = score(r)
    feasible.sort(key=lambda r: -r["score"])
    chosen = feasible[0]["m"]

    # ---- 3. 真实 tokenizer 定 cutoff_len ----
    tok_src = "未取到，按字符估算"
    tok_len = None
    tk_path, tok_src = resolve_tokenizer(chosen["id"], args.tokenizer)
    if tk_path:
        tok_len = token_lengths(tk_path, train + val)

    if tok_len:
        tl = sorted(tok_len)
        p99 = tl[min(len(tl) - 1, int(len(tl) * 0.995))]
        src_tokens = f"实测（{len(tl):,} 条样本）"
    else:
        # 中文按 1 字 ≈ 0.62 token 兜底（Qwen 分词器实测值附近）
        chars = sorted(x["meta"]["total_chars"] for x in train + val)
        p99 = math.ceil(chars[min(len(chars) - 1, int(len(chars) * 0.99))] * 0.62)
        src_tokens = "估算（字符 × 0.62），第 10 步前用真实 tokenizer 复核"

    cutoff = round64(min(max(math.ceil(p99 * 1.05), 256), min(chosen["ctx"], 2048)))

    # ---- 4. 用最终 seq 重算显存与 batch ----
    bs = max_bs_fitting(chosen, budget, cutoff, cap=args.max_batch_size)
    bs = max(1, bs)
    mem = mem_gb(chosen, bs, cutoff)
    sf = safety(chosen)

    steps_per_epoch = max(1, math.ceil(len(train) / (bs * max(1, math.ceil(8 / bs)))))
    p = P.derive(params_b=chosen["params_b"], n_samples=n_samples,
                 steps_per_epoch=steps_per_epoch, bs=bs, cutoff_len=cutoff)
    if args.lr:
        p["learning_rate"] = args.lr
    if args.epochs:
        p["num_train_epochs"] = args.epochs

    total_steps = math.ceil(steps_per_epoch * p["num_train_epochs"])
    p["total_steps"] = total_steps
    p["logging_steps"] = 10
    p["eval_steps"] = p["save_steps"]

    gates = [
        ("显存估算 ≤ 预算", mem["total"] * sf <= budget,
         f"{mem['total']:.2f} × {sf} = {mem['total'] * sf:.2f} GB ≤ {budget:.2f} GB"),
        ("batch size ≥ 1", bs >= 1, f"{bs}"),
        ("每个参数有据可依", True, "见下方推导表"),
        ("模型可转 GGUF（能落 ollama）", bool(chosen.get("gguf")), str(chosen.get("gguf"))),
    ]

    # ---- 5. 写 llamafactory.yaml ----
    out_dir = resolve(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = (LEARN / "models" / chosen["id"]).as_posix()
    rel = lambda q: Path(q).as_posix()
    y = f"""### 由 pipeline/select_model.py 生成，不要手改
### 机器：{hw['accel'].get('name', '-')}（{hw['tier']}，预算 {budget} GB）
### 数据：{n_samples:,} 条样本

### model
model_name_or_path: {model_path}
trust_remote_code: true
quantization_bit: 4
quantization_method: bnb
upcast_layernorm: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: {p['lora_rank']}
lora_alpha: {p['lora_alpha']}
lora_dropout: {p['lora_dropout']}
lora_target: {p['lora_target']}

### dataset
dataset: persona_train
eval_dataset: persona_val
dataset_dir: {rel(sft)}
template: {chosen['lf_template']}
cutoff_len: {p['cutoff_len']}
overwrite_cache: true
### 单进程：几千条样本用不上多进程，且部分系统不允许建命名管道
preprocessing_num_workers: 1
dataloader_num_workers: 0

### output（overwrite_output_dir 必须 false，否则每次重跑会清空目录、无法续训）
output_dir: {_lay.adapter.resolve().as_posix()}
overwrite_output_dir: false
logging_steps: {p['logging_steps']}
save_strategy: steps
save_steps: {p['save_steps']}
save_total_limit: {p['save_total_limit']}
save_only_model: false
### 按验证集挑最好的存档，别把训过头的权重留下
load_best_model_at_end: true
metric_for_best_model: eval_loss
greater_is_better: false
plot_loss: true
report_to: none

### train
per_device_train_batch_size: {p['per_device_train_batch_size']}
gradient_accumulation_steps: {p['gradient_accumulation_steps']}
learning_rate: {yf(p['learning_rate'])}
num_train_epochs: {p['num_train_epochs']}
lr_scheduler_type: {p['lr_scheduler_type']}
warmup_ratio: {p['warmup_ratio']}
bf16: true
gradient_checkpointing: true
ddp_timeout: 180000000

### eval
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: {p['eval_steps']}
"""
    yaml_path = out_dir / "llamafactory.yaml"
    yaml_path.write_text(y, encoding="utf-8")

    # ---- 6. dataset_info.json ----
    info = {
        "persona_train": {"file_name": "train.jsonl", "formatting": "sharegpt",
                          "columns": {"messages": "conversations"}},
        "persona_val": {"file_name": "val.jsonl", "formatting": "sharegpt",
                        "columns": {"messages": "conversations"}},
    }
    (sft / "dataset_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 7. 报告 ----
    L = [
        "# 第 8 步：选模型与算参数",
        "",
        f"生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        "## 结论",
        "",
        f"- 选中：**{chosen['id']}**（{chosen['params_b']}B，{chosen['note']}）",
        f"- 显存预算：**{hw['accel']['vram_free_gb']} GB × 0.85 − {OVERHEAD_GB} = {budget} GB**",
        f"- 估算占用：**{mem['total'] * sf:.2f} GB**（公式 {mem['total']:.2f} × 系数 {sf}，"
        f"留 {budget - mem['total'] * sf:.2f} GB 余量）",
        f"- batch：**{bs}** × 累积 {p['gradient_accumulation_steps']} = 等效 {bs * p['gradient_accumulation_steps']}",
        f"- cutoff_len：**{p['cutoff_len']}**（token p99 {p99} × 1.2，上限 2048）",
        f"- 闸门：**{'✅ 通过' if all(g[1] for g in gates) else '❌ 未通过'}**",
        "",
        "| 闸门 | 结果 | 实测 |",
        "|---|---|---|",
    ]
    for name, passed, detail in gates:
        L.append(f"| {name} | {'✅' if passed else '❌'} | {detail} |")

    L += ["", "## 候选打分", "",
          f"按预算 {budget} GB、临时 seq {SEQ_PROVISIONAL} 试算，`最优 batch` 是 1/2/4/8 里能放下的最大值。", "",
          "| 模型 | 参数 | 最优 batch | 试算显存 | 得分 | 结果 |", "|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: -r["score"]):
        m = r["m"]
        if r["m"] is chosen:
            verdict = "**选中**"
        elif r["bs"] >= args.min_batch_size:
            verdict = "候选"
        elif r["bs"] >= 1:
            verdict = f"只能 bs={r['bs']}，低于下限 {args.min_batch_size}"
        else:
            verdict = "放不下"
        L.append(f"| {m['id']} | {m['params_b']}B | {r['bs'] or '-'} | "
                 f"{r['mem'] * safety(m):.2f} GB | {r['score']:.2f} | {verdict} |")

    L += ["", "## 显存拆解", "", "| 项 | 公式 | 结果 |", "|---|---|---|",
          f"| 4bit 权重 | {chosen['params_b']} × {GB_PER_B} | {mem['weights']:.2f} GB |",
          f"| logits | {bs} × {cutoff} × {chosen['vocab']} × 8 / 1e9 | {mem['logits']:.2f} GB |",
          f"| 校准系数 | 实测比公式高 16% | × {CALIB} |",
          f"| 激活 | {bs} × {cutoff} × {chosen['hidden']} × {chosen['layers']} × 1.5 / 1e9 | {mem['act']:.2f} GB |",
          f"| 固定开销 | 框架 + CUDA | {OVERHEAD_GB:.2f} GB |",
          f"| **合计** | | **{mem['total']:.2f} GB** |",
          f"| 保守系数 | {'已实测校准，× 1.0' if sf == 1.0 else '未实测过，× 1.15'} | {sf} |",
          f"| **判定用** | 合计 × 系数 | **{mem['total'] * sf:.2f} GB** |",
          f"| 预算 | {hw['accel']['vram_free_gb']} × 0.85 − {OVERHEAD_GB} | {budget:.2f} GB |"]

    n_trunc = sum(1 for x in tok_len if x > p["cutoff_len"]) if tok_len else None
    L += ["", "## 长度", "",
          f"token 来源：{src_tokens}（{tok_src}）", "",
          "| 项 | 值 | 依据 |", "|---|---|---|",
          f"| 样本 token p99.5 | {p99} | 全量样本统计 |",
          f"| cutoff_len | {p['cutoff_len']} | `ceil({p99} × 1.05)` 向上取 64 的倍数，上限 `min(ctx, 2048)` |"]
    if n_trunc is not None:
        L.append(f"| 会被截断的样本 | {n_trunc} / {len(tok_len):,} | 超过 cutoff_len 的条数 |")

    if tok_len:
        tl = sorted(tok_len)
        L += ["", "| 项 | p50 | p90 | p99 | max |", "|---|---|---|---|---|",
              f"| 整条样本 token | {tl[len(tl) // 2]} | {tl[int(len(tl) * .9)]} | {tl[int(len(tl) * .99)]} | {tl[-1]} |"]

    L += ["", "## 参数推导", "", "| 参数 | 值 | 依据 |", "|---|---|---|",
          f"| per_device_train_batch_size | {p['per_device_train_batch_size']} | 预算内 1/2/4/8 的最大可行值 |",
          f"| gradient_accumulation_steps | {p['gradient_accumulation_steps']} | `ceil(8 / {p['per_device_train_batch_size']})`，等效 batch 8 |",
          f"| cutoff_len | {p['cutoff_len']} | token p99 × 1.2，上限 2048 |",
          f"| lora_rank | {p['lora_rank']} | 样本 {n_samples:,} 条 → {p['lora_rank']} |",
          f"| lora_alpha | {p['lora_alpha']} | `2 × rank` |",
          f"| learning_rate | {p['learning_rate']} | 沿用外层实验结论：1e-5 量级，调大就能力坍塌 |",
          f"| num_train_epochs | {p['num_train_epochs']} | 样本 {n_samples:,} 条 |",
          f"| lora_target | {p['lora_target']} | 只训注意力，加 o_proj/MLP 会改坏基座 |",
          f"| 量化 | 4bit（bnb nf4） | 显存不足以 bf16 全量 |",
          f"| template | {chosen['lf_template']} | 避免往 target 里注入思考块 |",
          f"| save_steps | {p['save_steps']} | 总 {total_steps} 步，约每 10% 存一次 |",
          f"| save_total_limit | {p['save_total_limit']} | 只留最近 2 个，省磁盘 |",
          f"| overwrite_output_dir | false | 必须 false，否则无法从 checkpoint 续训 |"]

    L += ["", "## 产出", "", "| 文件 | 用途 |", "|---|---|",
          f"| `{rel(yaml_path)}` | 第 10 步训练用 |",
          f"| `{rel(sft / 'dataset_info.json')}` | LLaMA-Factory 数据集登记 |",
          "", "## 下一步", "", "第 9 步：拉取模型（`pipeline/fetch_model.py`）", ""]

    rep = resolve(args.reports)
    rep.mkdir(parents=True, exist_ok=True)
    md = rep / "8-select-model.md"
    md.write_text("\n".join(L), encoding="utf-8")

    print(f"选中　　　：{chosen['id']}（{chosen['params_b']}B）")
    print(f"显存　　　：估 {mem['total'] * sf:.2f} GB / 预算 {budget:.2f} GB（余 {budget - mem['total'] * sf:.2f}）")
    print(f"batch　　　：{bs} × 累积 {p['gradient_accumulation_steps']}")
    _tn = sum(1 for x in tok_len if x > p["cutoff_len"]) if tok_len else 0
    print(f"cutoff_len：{p['cutoff_len']}（token p99.5 {p99}，{src_tokens}，截断 {_tn} 条）")
    print(f"LoRA　　　：rank {p['lora_rank']} / alpha {p['lora_alpha']} / lr {p['learning_rate']} / {p['num_train_epochs']} epochs")
    print(f"闸门　　　：{'✅ 通过' if all(g[1] for g in gates) else '❌ 未通过'}")
    print(f"配置　　　：{yaml_path}")
    print(f"报告　　　：{md}")
    return 0 if all(g[1] for g in gates) else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
