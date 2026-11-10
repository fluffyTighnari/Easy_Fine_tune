#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 7 步：造训练样本。把「一次连发」合成一个 target，取前面的对话当上下文。

输入：work/data/clean.jsonl + config
产出：work/sft/{train,val}.jsonl、work/reports/7-build-sft.md

用法（从 learn/ 目录）：
    ..\\.venv\\Scripts\\python.exe pipeline\\build_sft.py
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

LEARN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LEARN))

from lib.paths import CONFIG, DEFAULT_ROOT, Layout  # noqa: E402
sys.path.insert(0, str(LEARN))

EMOJI = re.compile(r"\[[^\[\]]{1,8}\]|[\U0001F300-\U0001FAFF\u2600-\u27BF]")


def pct(vals: list[int], q: float) -> int:
    if not vals:
        return 0
    v = sorted(vals)
    return v[min(len(v) - 1, int(len(v) * q))]


def build_turns(sess: list[dict], max_burst: int, max_target_chars: int) -> list[dict]:
    """同一会话内把连续同角色的消息合成一个 turn；me 的 turn 超长则切块。"""
    turns: list[dict] = []
    for m in sess:
        if m["role"] == "system":
            continue
        if m["role"] == "me" and not m["target_ok"]:
            continue  # 被排除的消息不进上下文也不进 target
        role = "human" if m["role"] == "them" else "gpt"
        if turns and turns[-1]["role"] == role:
            turns[-1]["msgs"].append(m)
        else:
            turns.append({"role": role, "msgs": [m]})

    out: list[dict] = []
    for t in turns:
        if t["role"] == "human":
            out.append({"role": "human", "msgs": t["msgs"],
                        "text": "\n".join(m["text"] for m in t["msgs"])})
            continue
        # gpt turn：只取紧接对方之后的前若干条。
        # 连发的后半段常常是他自己的碎碎念（换话题、翻旧账），拿它当目标会教会
        # 模型「不接话」，第 11 步的裁判能直接量到。所以块只往前切、不往后攒。
        cur: list[dict] = []
        for m in t["msgs"]:
            cand = cur + [m]
            too_many = len(cand) > max_burst
            too_long = sum(len(x["text"]) + 1 for x in cand) > max_target_chars
            if cur and (too_many or too_long):
                break  # 到上限就停，剩下的不进目标
            cur = cand
        if cur:
            out.append({"role": "gpt", "msgs": cur,
                        "text": "\n".join(x["text"] for x in cur)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="第 7 步：造训练样本")
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--root", default=DEFAULT_ROOT, help="输出根目录，默认 work/")
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--reports", default=None)
    args = ap.parse_args()

    _lay = Layout(args.root)
    _lay.ensure()
    args.data = args.data or str(_lay.data)
    args.out = args.out or str(_lay.sft)
    args.reports = args.reports or str(_lay.reports)

    def resolve(p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else LEARN / q

    import yaml

    cfg = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    sys_prompt = cfg["persona"]["system_prompt"].strip()
    max_burst = int(cfg.get("data", {}).get("max_burst", 8))
    s = cfg.get("sft", {}) or {}
    max_pairs = int(s.get("max_context_pairs", 3))
    max_ctx_chars = int(s.get("max_context_chars", 600))
    val_ratio = float(s.get("val_ratio", 0.1))
    min_val = int(s.get("min_val", 50))

    src = resolve(args.data) / "clean.jsonl"
    if not src.exists():
        print(f"输入不存在：{src}（先做第 6 步）")
        return 2
    rows = [json.loads(line) for line in open(src, encoding="utf-8") if line.strip()]

    sessions: dict[str, list[dict]] = {}
    for r in rows:
        sessions.setdefault(r["sid"], []).append(r)

    samples: list[dict] = []
    n_no_ctx = n_dropped = 0
    seen: set[str] = set()

    max_target_chars = int(cfg.get("clean", {}).get("max_target_chars", 120))
    for sid, sess in sessions.items():
        turns = build_turns(sess, max_burst, max_target_chars)
        for i, t in enumerate(turns):
            if t["role"] != "gpt":
                continue
            # 上下文：往前取若干轮，且受字数预算约束
            ctx: list[dict] = []
            budget = max_ctx_chars
            for prev in reversed(turns[max(0, i - max_pairs * 2):i]):
                if len(prev["text"]) > budget:
                    break
                ctx.insert(0, prev)
                budget -= len(prev["text"])
            while ctx and ctx[0]["role"] != "human":
                ctx.pop(0)  # 上下文必须从对方说话开始
            if not any(c["role"] == "human" for c in ctx):
                n_no_ctx += 1
                continue

            msgs = [{"from": "system", "value": sys_prompt}]
            msgs += [{"from": c["role"], "value": c["text"]} for c in ctx]
            msgs.append({"from": "gpt", "value": t["text"]})

            key = json.dumps(msgs, ensure_ascii=False)
            if key in seen:
                n_dropped += 1
                continue
            seen.add(key)

            samples.append({
                "conversations": msgs,
                "meta": {"sid": sid, "ts": t["msgs"][0]["ts"], "mid": t["msgs"][0]["mid"],
                         "n_target_msgs": len(t["msgs"]), "n_ctx_turns": len(ctx),
                         "target_chars": len(t["text"]),
                         "total_chars": sum(len(m["value"]) for m in msgs)},
            })

    samples.sort(key=lambda x: (x["meta"]["ts"] or "", x["meta"]["mid"]))

    n_val = max(min_val, math.ceil(len(samples) * val_ratio))
    n_val = min(n_val, max(0, len(samples) - 1))
    train, val = samples[:-n_val], samples[-n_val:]

    out_dir = resolve(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("val", val)):
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for x in part:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")

    # ---- 统计 ----
    tg = [x["meta"]["target_chars"] for x in samples]
    tot = [x["meta"]["total_chars"] for x in samples]
    nmsg = [x["meta"]["n_target_msgs"] for x in samples]
    split_ts = val[0]["meta"]["ts"] if val else None
    train_max_ts = train[-1]["meta"]["ts"] if train else None

    gates = [
        ("样本数 ≥ 300", len(samples) >= 300, f"{len(samples):,}"),
        ("每个样本含 system", all(x["conversations"][0]["from"] == "system" for x in samples),
         f"{len(samples):,}/{len(samples):,}"),
        ("每个样本有人类发言", all(any(m["from"] == "human" for m in x["conversations"][1:])
                                   for x in samples), "全部"),
        ("train/val 时间不交叉",
         bool(train_max_ts and split_ts and train_max_ts <= split_ts),
         f"train ≤ {train_max_ts} ＜ val ≤ {val[-1]['meta']['ts']}" if val and train else "无 val"),
        ("target 非空", all(x["conversations"][-1]["value"].strip() for x in samples), "全部"),
    ]

    L = [
        "# 第 7 步：造训练样本",
        "",
        f"生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        "## 结论",
        "",
        f"- 样本：**{len(samples):,}**（train {len(train):,} / val {len(val):,}）",
        f"- 跳过：无上下文 {n_no_ctx}｜完全重复 {n_dropped}",
        f"- 时间切分：train ≤ {train_max_ts}　val 从 {split_ts} 起",
        f"- 闸门：**{'✅ 通过' if all(g[1] for g in gates) else '❌ 未通过'}**",
        "",
        "| 闸门 | 结果 | 实测 |",
        "|---|---|---|",
    ]
    for name, passed, detail in gates:
        L.append(f"| {name} | {'✅' if passed else '❌'} | {detail} |")

    L += ["", "## 样本结构", "",
          "```",
          "system : " + sys_prompt,
          "human  : 对方的一次发言（可能多条合并）",
          "gpt    : 目标人物的一次连发（可能多条合并）   ← 训练目标",
          "```", "",
          f"上下文最多取 {max_pairs} 轮（{max_ctx_chars} 字以内），不跨会话。", "",
          "| 项 | p50 | p90 | p99 | max |", "|---|---|---|---|---|",
          f"| 目标长度（字符） | {pct(tg, .5)} | {pct(tg, .9)} | {pct(tg, .99)} | {max(tg)} |",
          f"| 整条样本长度（字符） | {pct(tot, .5)} | {pct(tot, .9)} | {pct(tot, .99)} | {max(tot)} |",
          f"| 目标消息条数 | {pct(nmsg, .5)} | {pct(nmsg, .9)} | {pct(nmsg, .99)} | {max(nmsg)} |"]

    L += ["", "| 目标消息条数 | 样本数 |", "|---|---|"]
    for k, v in sorted(Counter(nmsg).items()):
        if k <= 8 or v > 100:
            L.append(f"| {k} | {v:,} |")

    L += ["", "## 样例", ""]
    for x in (train[:1] + train[len(train) // 2:len(train) // 2 + 1] + val[-1:]):
        L += ["```"]
        for m in x["conversations"]:
            L.append(f"[{m['from']}] {m['value'].replace(chr(10), ' / ')[:110]}")
        L.append("```")
        L.append("")

    L += ["## 下一步", "", "第 8 步：选模型、算参数（`pipeline/select_model.py`）", ""]

    rep = resolve(args.reports)
    rep.mkdir(parents=True, exist_ok=True)
    md = rep / "7-build-sft.md"
    md.write_text("\n".join(L), encoding="utf-8")

    print(f"样本　　　：{len(samples):,}（train {len(train):,} / val {len(val):,}）")
    print(f"跳过　　　：无上下文 {n_no_ctx}｜重复 {n_dropped}")
    print(f"目标长度　：p50 {pct(tg, .5)}｜p90 {pct(tg, .9)}｜p99 {pct(tg, .99)}｜max {max(tg)}")
    print(f"样本长度　：p50 {pct(tot, .5)}｜p90 {pct(tot, .9)}｜p99 {pct(tot, .99)}｜max {max(tot)}")
    print(f"时间切分　：train ≤ {train_max_ts}｜val {split_ts} → {val[-1]['meta']['ts'] if val else '-'}")
    print(f"闸门　　　：{'✅ 通过' if all(g[1] for g in gates) else '❌ 未通过'}")
    for name, passed, detail in gates:
        if not passed:
            print(f"  ❌ {name}：{detail}")
    print(f"输出　　　：{out_dir}")
    print(f"报告　　　：{md}")
    return 0 if all(g[1] for g in gates) else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
