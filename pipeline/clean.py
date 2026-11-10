#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 6 步：清洗。只标不删，唯一的删除是「同一会话内 60 秒内的完全相同重发」。

输入：work/data/attributed.jsonl + config
产出：work/data/clean.jsonl、work/reports/6-clean.md

用法（从 learn/ 目录）：
    ..\\.venv\\Scripts\\python.exe pipeline\\clean.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

LEARN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LEARN))

from lib.paths import CONFIG, DEFAULT_ROOT, Layout  # noqa: E402
sys.path.insert(0, str(LEARN))

# ---- AI 粘贴判别：结构化标记打分，不靠关键词 ----
BOLD = re.compile(r"\*\*|^#{1,6} ", re.M)
NUML = re.compile(r"^\s*(?:\d+[.、)]|[①-⑩])", re.M)
SUBHD = re.compile(r"(?:^|\n)\s*(?:本质|结构|核心|定义|特点|原理|优势|劣势|步骤|方法|概念|要点|方向|场合)[：:]")
ARROW = re.compile(r"→|=>|⇒")
ORAL = re.compile(r"[啊吧嘛呢哈呀哦嗯啦诶欸]")
SECOND = re.compile(r"你|您")

CJK = re.compile(r"[\u4e00-\u9fff]")
LAT = re.compile(r"[A-Za-z]")
EFF = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")

PHONE = re.compile(r"1[3-9]\d{9}")
IDCARD = re.compile(r"\b\d{17}[\dXx]\b")
REDACTED = "[敏感数据已脱敏]"

UNUSABLE_QUOTE = {"quote_no_reply", "quote_no_quote", "quote_multiple"}


def ai_score(t: str) -> tuple[int, list[str]]:
    s, why = 0, []
    if BOLD.search(t):
        s += 3; why.append("加粗/标题")
    if len(NUML.findall(t)) >= 2:
        s += 2; why.append("编号列表")
    if SUBHD.search(t):
        s += 2; why.append("小标题冒号")
    if len(ARROW.findall(t)) >= 2:
        s += 1; why.append("箭头链")
    if not ORAL.search(t):
        s += 1; why.append("无口语词")
    if not SECOND.search(t):
        s += 1; why.append("无你/我")
    return s, why


def latin_ratio(t: str) -> float:
    c, l = len(CJK.findall(t)), len(LAT.findall(t))
    return l / (c + l) if c + l else 0.0


def mask_privacy(t: str, on: bool) -> tuple[str, int]:
    n = 0
    if not on:
        return t, 0
    t, k = PHONE.subn("<phone>", t); n += k
    t, k = IDCARD.subn("<id>", t); n += k
    return t, n


def main() -> int:
    ap = argparse.ArgumentParser(description="第 6 步：清洗")
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--root", default=DEFAULT_ROOT, help="输出根目录，默认 work/")
    ap.add_argument("--data", default=None)
    ap.add_argument("--reports", default=None)
    args = ap.parse_args()

    _lay = Layout(args.root)
    _lay.ensure()
    args.data = args.data or str(_lay.data)
    args.reports = args.reports or str(_lay.reports)

    def resolve(p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else LEARN / q

    import yaml

    cfg = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    c = cfg.get("clean", {}) or {}
    max_chars = int(c.get("max_target_chars", 300))
    max_en = int(c.get("max_english_chars", 200))
    ai_min_score = int(c.get("ai_score_min", 4))
    ai_min_chars = int(c.get("ai_min_chars", 200))
    merge_resent_sec = int(c.get("merge_resent_sec", 60))
    mask_on = bool(c.get("mask_privacy", True))

    src = resolve(args.data) / "attributed.jsonl"
    if not src.exists():
        print(f"输入不存在：{src}（先做第 5 步）")
        return 2
    rows = [json.loads(line) for line in open(src, encoding="utf-8") if line.strip()]
    n_in = len(rows)

    hits: Counter[str] = Counter()
    hit_me: Counter[str] = Counter()
    samples: dict[str, list[str]] = {}
    masked = redacted = n_merged = 0
    kept: list[dict] = []
    prev = None

    for r in rows:
        flags = [f for f in r["flags"]]
        reasons: list[str] = []
        text = r["text"]

        # R6 隐私掩码
        text, k = mask_privacy(text, mask_on)
        if k:
            masked += k
            flags.append("privacy_masked")

        # R7 脱敏占位符剔除（保留其余内容）
        if REDACTED in text:
            text = text.replace(REDACTED, "")
            redacted += 1
            flags.append("redaction_removed")
            text = re.sub(r"[ \t]{2,}", " ", text).strip() or text.strip()

        # R9 重发合并：同一会话内、与上一条完全相同、间隔 ≤ 阈值
        if (prev is not None and merge_resent_sec > 0 and text == prev["text"]
                and r["sid"] == prev["sid"] and r["ts"] and prev["ts"]):
            gap = (datetime.fromisoformat(r["ts"]) - datetime.fromisoformat(prev["ts"])).total_seconds()
            if gap <= merge_resent_sec:
                n_merged += 1
                continue

        # R1 系统消息
        if r["role"] == "system":
            reasons.append("system")
        # R2 引用结构不完整
        if any(f in UNUSABLE_QUOTE for f in flags):
            reasons.append("quote_unusable")
        # R3 外部粘贴（AI 味）
        sc, why = ai_score(text)
        if sc >= ai_min_score and len(text) > ai_min_chars:
            reasons.append("pasted_like")
            samples.setdefault("pasted_like", []).append(f"[{sc}分 {','.join(why)}] {text[:60]}")
        # R4 长英文书信
        if latin_ratio(text) > 0.7 and len(text) > max_en:
            reasons.append("long_english")
            samples.setdefault("long_english", []).append(text[:60])
        # R5 超长单条
        if len(text) > max_chars:
            reasons.append("too_long")
            samples.setdefault("too_long", []).append(f"[{len(text)}字] {text[:50]}")
        # R8 低信息
        if len(EFF.findall(text)) < 2:
            reasons.append("low_info")

        target_ok = (r["role"] == "me") and not reasons
        if r["role"] == "me":
            for x in reasons:
                hit_me[x] += 1
                hits[x] += 1
        else:
            for x in reasons:
                hits[x] += 1

        rec = {"mid": r["mid"], "sid": r["sid"], "turn": r["turn"], "role": r["role"],
               "text": text, "ts": r["ts"], "flags": flags,
               "target_ok": target_ok, "excluded_by": reasons}
        kept.append(rec)
        prev = rec

    # ---- 写出 ----
    out = resolve(args.data) / "clean.jsonl"
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---- 统计 ----
    me = [r for r in kept if r["role"] == "me"]
    me_ok = [r for r in me if r["target_ok"]]
    pool_ratio = len(me_ok) / len(me) if me else 0
    lens = sorted(len(r["text"]) for r in me_ok)
    p = lambda q: lens[min(len(lens) - 1, int(len(lens) * q))] if lens else 0

    left_phone = sum(1 for r in kept if PHONE.search(r["text"]))
    left_red = sum(1 for r in kept if REDACTED in r["text"])

    gates = [
        ("me 可训练比例 ≥ 80%", pool_ratio >= 0.80, f"{pool_ratio:.1%}（{len(me_ok):,}/{len(me):,}）"),
        ("手机号已全部脱敏", left_phone == 0, f"剩 {left_phone} 条"),
        ("脱敏占位符已清空", left_red == 0, f"剩 {left_red} 条"),
    ]

    RULES = [
        ("R1 系统消息", "role = system", "不作 target", hits["system"], hit_me["system"]),
        ("R2 引用结构不完整", "flags 含 quote_no_reply / no_quote / multiple", "不作 target",
         hits["quote_unusable"], hit_me["quote_unusable"]),
        ("R3 外部粘贴（AI 味）", f"结构化打分 ≥ {ai_min_score} 分 且 > {ai_min_chars} 字", "不作 target",
         hits["pasted_like"], hit_me["pasted_like"]),
        ("R4 长英文书信", f"拉丁字母占比 > 70% 且 > {max_en} 字", "不作 target",
         hits["long_english"], hit_me["long_english"]),
        ("R5 超长单条", f"> {max_chars} 字", "不作 target", hits["too_long"], hit_me["too_long"]),
        ("R8 低信息", "有效字符 < 2", "不作 target", hits["low_info"], hit_me["low_info"]),
    ]

    L = [
        "# 第 6 步：清洗",
        "",
        f"生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        "## 结论",
        "",
        f"- 输入 {n_in:,} 条 → 输出 **{len(kept):,}** 条（合并重发 -{n_merged}）",
        f"- `me` 可训练：**{len(me_ok):,} / {len(me):,}**（{pool_ratio:.1%}）",
        f"- 文本被改动：隐私掩码 {masked} 处，脱敏占位符剔除 {redacted} 条",
        f"- 闸门：**{'✅ 通过' if all(g[1] for g in gates) else '❌ 未通过'}**",
        "",
        "| 闸门 | 结果 | 实测 |",
        "|---|---|---|",
    ]
    for name, passed, detail in gates:
        L.append(f"| {name} | {'✅' if passed else '❌'} | {detail} |")

    L += ["", "## 规则命中", "",
          "「不作 target」= 留在数据里当上下文，但不算训练目标。「合并」= 真删除。", "",
          "| 规则 | 判据 | 处置 | 命中 | 其中 me |", "|---|---|---|---|---|"]
    for name, crit, act, n, nm in RULES:
        L.append(f"| {name} | {crit} | {act} | {n:,} | {nm:,} |")
    L.append(f"| R6 隐私掩码 | 手机号 / 身份证号 | 替换成 `<phone>` `<id>` | {masked} 处 | - |")
    L.append(f"| R7 脱敏占位 | 含 `{REDACTED}` | 剔除占位符，保留其余文字 | {redacted} | - |")
    L.append(f"| R9 重发合并 | 同会话、与上一条完全相同、≤ {merge_resent_sec} 秒 | **删除** | {n_merged} | - |")

    L += ["", "## 可训练池（清洗后）", "", "| 项 | 值 |", "|---|---|",
          f"| 条数 | {len(me_ok):,} |",
          f"| 长度 p50 / p90 / p99 / max | {p(.5)} / {p(.9)} / {p(.99)} / {lens[-1] if lens else 0} |",
          f"| 含表情 | {sum(1 for r in me_ok if '[' in r['text'] or re.search(r'[\U0001F300-\U0001FAFF]', r['text'])):,} |",
          f"| 含英文（不论长短） | {sum(1 for r in me_ok if LAT.search(r['text'])):,} |"]

    for key, label in (("pasted_like", "R3 外部粘贴"), ("long_english", "R4 长英文"),
                       ("too_long", "R5 超长")):
        if samples.get(key):
            L += ["", f"### {label} 样例（共 {len(samples[key])} 条）", "", "```"]
            L += [s.replace("\n", "\\n")[:100] for s in samples[key][:5]]
            L.append("```")

    L += ["", "## 下一步", "", "第 7 步：造训练样本（`pipeline/build_sft.py`）", ""]

    rep = resolve(args.reports)
    rep.mkdir(parents=True, exist_ok=True)
    md = rep / "6-clean.md"
    md.write_text("\n".join(L), encoding="utf-8")

    print(f"条数　　　：{n_in:,} → {len(kept):,}（合并重发 -{n_merged}）")
    print(f"可训练池　：me {len(me_ok):,} / {len(me):,}（{pool_ratio:.1%}）")
    print(f"排除明细　：超长 {hit_me['too_long']}｜长英文 {hit_me['long_english']}｜AI粘贴 {hit_me['pasted_like']}"
          f"｜低信息 {hit_me['low_info']}｜引用残留 {hit_me['quote_unusable']}")
    print(f"文本改动　：隐私 {masked} 处｜脱敏占位 {redacted} 条")
    print(f"闸门　　　：{'✅ 通过' if all(g[1] for g in gates) else '❌ 未通过'}")
    for name, passed, detail in gates:
        if not passed:
            print(f"  ❌ {name}：{detail}")
    print(f"输出　　　：{out}")
    print(f"报告　　　：{md}")
    return 0 if all(g[1] for g in gates) else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
