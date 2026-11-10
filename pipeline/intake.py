#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 3 步：读入 chat.txt，出数据体检报告。只读不改。

产出：
    work/data/messages_raw.jsonl   原始顺序的逐条记录
    work/reports/3-intake.md       体检报告

用法（从 learn/ 目录）：
    ..\\.venv\\Scripts\\python.exe pipeline\\intake.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

LEARN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LEARN))

from lib.paths import DEFAULT_ROOT, Layout  # noqa: E402
sys.path.insert(0, str(LEARN))
from lib import chat_txt  # noqa: E402

LONG_CHARS = 300
SESSION_GAP_MIN = 30


def pct(sorted_vals: list[int], q: float) -> int:
    if not sorted_vals:
        return 0
    return sorted_vals[min(len(sorted_vals) - 1, int(len(sorted_vals) * q))]


def parse_ts(s: str | None) -> datetime | None:
    if not s:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M:%S",
                "%Y/%m/%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            continue
    return None


def analyze(records: list[dict], unparsed: list[dict]) -> dict:
    total = len(records) + len(unparsed)
    speakers = Counter(r["sender"] for r in records)
    n = len(records)

    ts_list = [(i, parse_ts(r["sent_at"])) for i, r in enumerate(records)]
    with_ts = [(i, t) for i, t in ts_list if t]
    bad_ts = sum(1 for r in records if r["sent_at"] and parse_ts(r["sent_at"]) is None)

    # 顺序判定：相邻两对里「后一条比前一条早」的比例
    desc = sum(1 for a, b in zip(with_ts, with_ts[1:]) if b[1] < a[1])
    desc_ratio = desc / max(1, len(with_ts) - 1)

    # 会话估算：按时间排序后，间隔 > 30 分钟即断开
    ordered = sorted(t for _, t in with_ts)
    gaps = [(b - a).total_seconds() / 60 for a, b in zip(ordered, ordered[1:])]
    sessions = 1 + sum(1 for g in gaps if g > SESSION_GAP_MIN) if ordered else 0

    lengths = sorted(len(r["content"]) for r in records)
    months = Counter(t.strftime("%Y-%m") for _, t in with_ts)
    dup = Counter((r["sender"], r["content"]) for r in records)
    dup_extra = sum(c - 1 for c in dup.values() if c > 1)

    whole_tag = [r for r in records if r["content"].strip().startswith("[")
                 and r["content"].strip().endswith("]")
                 and len(r["content"].strip()) <= 12]
    media_left = [r for r in whole_tag if r["content"].strip()[1:-1] in chat_txt.MEDIA_TAGS]

    return {
        "total": total, "n": n, "n_unparsed": len(unparsed),
        "unparsed_ratio": len(unparsed) / total if total else 1.0,
        "speakers": speakers,
        "n_ts": len(with_ts), "bad_ts": bad_ts,
        "t_min": ordered[0] if ordered else None, "t_max": ordered[-1] if ordered else None,
        "desc_ratio": desc_ratio, "is_desc": desc_ratio > 0.9,
        "sessions": sessions,
        "lengths": lengths,
        "months": months,
        "quote": sum(1 for r in records if "------" in r["content"]),
        "wechat_emoji": sum(1 for r in whole_tag if r["content"].strip()[1:-1] not in chat_txt.MEDIA_TAGS),
        "unicode_emoji": sum(1 for r in records if chat_txt.UNICODE_EMOJI.search(r["content"])),
        "multiline": sum(1 for r in records if "\n" in r["content"]),
        "dup_extra": dup_extra, "dup_pairs": sum(1 for c in dup.values() if c > 1),
        "long": sum(1 for x in lengths if x > LONG_CHARS),
        "media_left": len(media_left),
        "blank": sum(1 for r in records if not r["content"].strip()),
    }


def render_md(a: dict, gates: list[tuple[str, bool, str]]) -> str:
    total, n = a["total"], a["n"]
    ok = all(g[1] for g in gates)
    L = [
        "# 第 3 步：数据体检",
        "",
        f"生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        "## 结论",
        "",
        f"- 条数：**{n:,}**（解析失败 {a['n_unparsed']}，共 {total:,} 行）",
        f"- 说话人：**{len(a['speakers'])}** 个",
        f"- 时间跨度：{a['t_min']:%Y-%m-%d} → {a['t_max']:%Y-%m-%d}"
        if a["t_min"] and a["t_max"] else "- 时间跨度：无可用时间戳",
        f"- 估算会话数：**{a['sessions']}**（间隔 > {SESSION_GAP_MIN} 分钟断开）",
        f"- 闸门：**{'✅ 通过' if ok else '❌ 未通过'}**",
        "",
        "| 闸门 | 结果 | 实测 |",
        "|---|---|---|",
    ]
    for name, passed, detail in gates:
        L.append(f"| {name} | {'✅' if passed else '❌'} | {detail} |")

    L += ["", "## 说话人", "", "| 说话人 | 条数 | 占比 |", "|---|---|---|"]
    for name, c in a["speakers"].most_common(20):
        L.append(f"| {name} | {c:,} | {c / n:.1%} |")

    L += ["", "## 时间", "", "| 项 | 值 |", "|---|---|"]
    L.append(f"| 有时间戳 | {a['n_ts']:,} / {n:,}（{a['n_ts'] / n:.1%}） |")
    L.append(f"| 时间格式无法解析 | {a['bad_ts']:,} |")
    L.append(f"| 顺序 | {'**倒序**（最新在前）' if a['is_desc'] else '升序' if a['desc_ratio'] < 0.1 else '混乱'}，逆序比例 {a['desc_ratio']:.1%} |")
    L += ["", "| 月份 | 条数 |", "|---|---|"]
    for m, c in sorted(a["months"].items()):
        L.append(f"| {m} | {c:,} |")

    ls = a["lengths"]
    L += ["", "## 长度（单条消息，字符）", "", "| p50 | p75 | p90 | p99 | max |", "|---|---|---|---|---|",
          f"| {pct(ls, .5)} | {pct(ls, .75)} | {pct(ls, .9)} | {pct(ls, .99)} | {ls[-1] if ls else 0} |",
          "", "| 区间 | 条数 |", "|---|---|"]
    buckets = [("<10", lambda x: x < 10), ("10~30", lambda x: 10 <= x < 30),
               ("30~100", lambda x: 30 <= x < 100), ("100~300", lambda x: 100 <= x <= 300),
               (f">{LONG_CHARS}", lambda x: x > LONG_CHARS)]
    for label, fn in buckets:
        L.append(f"| {label} | {sum(1 for x in ls if fn(x)):,} |")

    L += ["", "## 内容特征", "", "| 特征 | 条数 | 占比 |", "|---|---|---|"]
    for label, key in [("含引用块（`------`）", "quote"), ("整条就是微信表情", "wechat_emoji"),
                       ("含普通 emoji", "unicode_emoji"), ("含换行（已转义）", "multiline"),
                       (f"长度 > {LONG_CHARS}", "long"), ("内容为空", "blank")]:
        L.append(f"| {label} | {a[key]:,} | {a[key] / n:.1%} |")
    L.append(f"| 重复（多出来的条数） | {a['dup_extra']:,} | {a['dup_extra'] / n:.1%} |")

    L += ["", "## 看这份报告要判断的事", ""]
    if a["is_desc"]:
        L.append("- **消息是倒序的**，第 4 步会按时间重排")
    if a["quote"]:
        L.append(f"- 有 {a['quote']:,} 条带引用块，第 4 步剥离；剥不干净的会打标记不参与训练")
    if a["dup_extra"]:
        L.append(f"- 有 {a['dup_extra']:,} 条重复，第 6 步去重")
    if a["long"]:
        L.append(f"- 有 {a['long']:,} 条超过 {LONG_CHARS} 字符，第 6 步按长文处理")
    if a["media_left"]:
        L.append(f"- ⚠️ 还有 {a['media_left']:,} 条媒体占位符没删，回第 1 步处理")
    if len(a["speakers"]) < 2:
        L.append("- ⚠️ 只有一个说话人，检查是否只导出了单方消息")
    if not a["is_desc"] and not a["quote"] and not a["dup_extra"] and not a["long"]:
        L.append("- 没有检出明显脏点")

    L += ["", "## 下一步", "", "第 4 步：规范化（`pipeline/normalize.py`）", ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description="第 3 步：读入与体检")
    ap.add_argument("--root", default=DEFAULT_ROOT, help="输出根目录，默认 work/")
    ap.add_argument("--input", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--reports", default=None)
    args = ap.parse_args()

    _lay = Layout(args.root)
    _lay.ensure()
    args.input = args.input or str(_lay.chat)
    args.data = args.data or str(_lay.data)
    args.reports = args.reports or str(_lay.reports)

    def resolve(p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else LEARN / q

    src = resolve(args.input)
    if not src.exists():
        print(f"输入不存在：{src}（先做第 1 步）")
        return 2

    records, unparsed = chat_txt.parse_file(src)
    if not records:
        print("一条都没解析出来，回第 1 步检查格式")
        return 2

    for i, r in enumerate(records, 1):
        r["mid"] = f"m{i:06d}"

    work = resolve(args.data)
    work.mkdir(parents=True, exist_ok=True)
    raw_path = work / "messages_raw.jsonl"
    with open(raw_path, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps({"mid": r["mid"], "lineno": r["lineno"], "sender": r["sender"],
                                "content": r["content"], "sent_at": r["sent_at"]},
                               ensure_ascii=False) + "\n")

    a = analyze(records, unparsed)
    gates = [
        ("解析失败率 ≤ 5%", a["unparsed_ratio"] <= 0.05, f"{a['unparsed_ratio']:.1%}"),
        ("说话人 ≥ 2", len(a["speakers"]) >= 2, f"{len(a['speakers'])} 个"),
        ("媒体占位符已清空", a["media_left"] == 0, f"剩 {a['media_left']} 条"),
    ]

    rep = resolve(args.reports)
    rep.mkdir(parents=True, exist_ok=True)
    md = rep / "3-intake.md"
    md.write_text(render_md(a, gates), encoding="utf-8")

    print(f"条数　　　：{a['n']:,}（解析失败 {a['n_unparsed']}）")
    print(f"说话人　　：{len(a['speakers'])} 个 —— " +
          "，".join(f"{k} {v:,}" for k, v in a["speakers"].most_common(4)))
    if a["t_min"]:
        print(f"时间跨度　：{a['t_min']:%Y-%m-%d} → {a['t_max']:%Y-%m-%d}，顺序 {'倒序' if a['is_desc'] else '升序'}")
    print(f"长度　　　：p50={pct(a['lengths'], .5)} p90={pct(a['lengths'], .9)} max={a['lengths'][-1]}")
    print(f"待处理　　：引用块 {a['quote']:,}｜重复 {a['dup_extra']:,}｜长文 {a['long']:,}")
    print(f"闸门　　　：{'✅ 通过' if all(g[1] for g in gates) else '❌ 未通过'}")
    for name, passed, detail in gates:
        if not passed:
            print(f"  ❌ {name}：{detail}")
    print(f"逐条记录　：{raw_path}")
    print(f"报告　　　：{md}")
    return 0 if all(g[1] for g in gates) else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
