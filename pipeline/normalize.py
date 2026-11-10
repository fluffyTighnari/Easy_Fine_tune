#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 4 步：规范化。时间重排、系统消息归位、引用块剥离。

输入：work/data/messages_raw.jsonl
产出：work/data/messages.jsonl、work/reports/4-normalize.md

用法（从 learn/ 目录）：
    ..\\.venv\\Scripts\\python.exe pipeline\\normalize.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

LEARN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LEARN))

from lib.paths import DEFAULT_ROOT, Layout  # noqa: E402
sys.path.insert(0, str(LEARN))

SYSTEM_SENDERS = {"系统", "system", "System", "SYSTEM", "通知"}
SYSTEM_PATTERNS = re.compile(
    r"^(通话时长|已拒绝|对方未接听|已取消|对方已取消|你撤回了一条消息|"
    r".{1,12}撤回了一条消息|.{1,12}拍了拍.{0,12}|以上是打招呼的内容)"
)
QUOTE_SEP = re.compile(r"\n\s*-{3,}\s*\n")
TS_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M:%S",
              "%Y/%m/%d %H:%M", "%Y-%m-%d")


def sys_kind(t: str) -> str:
    if t.startswith("通话时长"):
        return "通话（通话时长 xx:xx）"
    if "撤回" in t:
        return "撤回消息"
    if "拍了拍" in t:
        return "拍一拍"
    if t in ("已取消", "对方已取消", "已拒绝", "对方未接听"):
        return "通话结果（取消/拒绝/未接听）"
    return t[:16]


def parse_ts(s: str | None) -> datetime | None:
    if not s:
        return None
    for fmt in TS_FORMATS:
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            continue
    return None


def split_quote(text: str) -> tuple[str, str | None, str | None]:
    """返回 (状态, 正文, 引用)。

    状态：none 无引用块 / stripped 已剥离 / no_reply 引用块后无正文 /
          no_quote 分隔符前不是引用 / multiple 多个分隔符
    """
    if "------" not in text:
        return "none", text, None

    parts = QUOTE_SEP.split(text, maxsplit=1)
    if len(parts) != 2:
        # 分隔符存在但不在「换行+分隔符+换行」的位置
        if QUOTE_SEP.split(text)[0] != text and len(QUOTE_SEP.findall(text)) == 0:
            return "no_quote", text, None
        return "no_reply", text, None

    head, tail = parts[0].strip(), parts[1].strip()
    if not tail:
        return "no_reply", text, None
    if not head or head[0] not in "\"“「":
        return "no_quote", text, None
    if QUOTE_SEP.search(tail):
        return "multiple", text, None
    return "stripped", tail, head


def main() -> int:
    ap = argparse.ArgumentParser(description="第 4 步：规范化")
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

    src = resolve(args.data) / "messages_raw.jsonl"
    if not src.exists():
        print(f"输入不存在：{src}（先做第 3 步）")
        return 2

    rows = [json.loads(line) for line in open(src, encoding="utf-8") if line.strip()]
    n_in = len(rows)
    for r in rows:  # 第 3 步写的是 content，这里统一叫 text
        r["text"] = r.pop("content")

    # ---- 1. 时间解析 ----
    for r in rows:
        r["_ts"] = parse_ts(r["sent_at"])
    no_ts = sum(1 for r in rows if r["_ts"] is None)

    # ---- 2. 重排：有时间的按升序；无时间戳的保持原序放最后 ----
    timed = sorted([r for r in rows if r["_ts"]], key=lambda r: (r["_ts"], r["lineno"]))
    untimed = [r for r in rows if not r["_ts"]]
    ordered = timed + untimed

    # ---- 3. 系统消息 ----
    n_system = 0
    for r in ordered:
        r["flags"] = []
        is_sys = r["sender"] in SYSTEM_SENDERS or bool(SYSTEM_PATTERNS.match(r["text"]))
        if is_sys:
            r["role"] = "system"
            n_system += 1
        else:
            r["role"] = "pending"  # 第 5 步决定 me / them
        r["text"] = r["text"] if not is_sys else r["text"].strip()

    # ---- 4. 引用块剥离（系统消息不处理）----
    stats = {"none": 0, "stripped": 0, "no_reply": 0, "no_quote": 0, "multiple": 0}
    stripped_chars = 0
    examples: list[tuple[str, str, str]] = []

    for r in ordered:
        if r["role"] == "system":
            stats["none"] += 1
            continue
        state, body, quote = split_quote(r["text"])
        stats[state] += 1
        if state == "stripped":
            stripped_chars += len(r["text"]) - len(body)
            r["quote"] = quote
            r["flags"].append("quote_stripped")
            r["text"] = body
            if len(examples) < 3:
                examples.append((r["sender"], quote, body))
        elif state != "none":
            r["flags"].append(f"quote_{state}")

    # ---- 5. 重新编号 ----
    for i, r in enumerate(ordered, 1):
        r["mid"] = f"m{i:06d}"

    # ---- 6. 写出 ----
    out = resolve(args.data) / "messages.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for r in ordered:
            rec = {"mid": r["mid"], "sender": r["sender"], "role": r["role"],
                   "text": r["text"], "ts": r["_ts"].isoformat(sep=" ") if r["_ts"] else None,
                   "flags": r["flags"]}
            if r.get("quote"):
                rec["quote"] = r["quote"]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---- 7. 校验：时间是否升序 ----
    ts_only = [r["_ts"] for r in ordered if r["_ts"]]
    ascending = all(a <= b for a, b in zip(ts_only, ts_only[1:]))

    # ---- 8. 报告 ----
    quote_total = stats["stripped"] + stats["no_reply"] + stats["no_quote"] + stats["multiple"]
    gates = [
        ("时间全部升序", ascending, f"{'是' if ascending else '否'}"),
        ("引用块处置可解释", quote_total > 0,
         f"含分隔符 {quote_total:,} 条 = 剥离 {stats['stripped']:,} + 结构不完整 {quote_total - stats['stripped']:,}"),
        ("系统消息已归位", n_system > 0, f"{n_system} 条"),
    ]

    L = [
        "# 第 4 步：规范化",
        "",
        f"生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        "## 结论",
        "",
        f"- 输入 {n_in:,} 条 → 输出 {len(ordered):,} 条（条数不变）",
        f"- 时间范围：{ts_only[0]:%Y-%m-%d %H:%M} → {ts_only[-1]:%Y-%m-%d %H:%M}" if ts_only else "- 无时间戳",
        f"- 系统消息：{n_system} 条",
        f"- 引用块：剥离 {stats['stripped']:,} 条，结构不完整 {quote_total - stats['stripped']:,} 条",
        f"- 闸门：**{'✅ 通过' if all(g[1] for g in gates) else '❌ 未通过'}**",
        "",
        "| 闸门 | 结果 | 实测 |",
        "|---|---|---|",
    ]
    for name, passed, detail in gates:
        L.append(f"| {name} | {'✅' if passed else '❌'} | {detail} |")

    L += ["", "## 时间重排", "", "| 项 | 值 |", "|---|---|",
          f"| 原顺序 | 倒序（最新在前） |" if rows and rows[0]["_ts"] and rows[-1]["_ts"] and rows[0]["_ts"] > rows[-1]["_ts"] else "| 原顺序 | 升序 |",
          f"| 现顺序 | {'升序' if ascending else '**非升序**'} |",
          f"| 无时间戳 | {no_ts} 条（排在最后，保持原相对顺序） |",
          f"| 首条 | {ts_only[0]:%Y-%m-%d %H:%M:%S} |" if ts_only else "| 首条 | - |",
          f"| 末条 | {ts_only[-1]:%Y-%m-%d %H:%M:%S} |" if ts_only else "| 末条 | - |"]

    L += ["", "## 系统消息", "", "| 项 | 值 |", "|---|---|",
          f"| 归为 system | {n_system} 条 |", ""]
    sys_kinds: dict[str, int] = {}
    for r in ordered:
        if r["role"] == "system":
            k = sys_kind(r["text"])
            sys_kinds[k] = sys_kinds.get(k, 0) + 1
    if sys_kinds:
        L += ["| 内容 | 条数 |", "|---|---|"]
        for k, v in sorted(sys_kinds.items(), key=lambda x: -x[1]):
            L.append(f"| {k} | {v} |")

    L += ["", "## 引用块剥离", "", "| 状态 | 条数 | 说明 |", "|---|---|---|",
          f"| 剥离成功 | {stats['stripped']:,} | 保留正文，引用存进 `quote` 字段 |",
          f"| 后段无正文 | {stats['no_reply']:,} | 原样保留，打 `quote_no_reply` |",
          f"| 前段不是引用 | {stats['no_quote']:,} | 分隔符不像引用块，原样保留 |",
          f"| 多个分隔符 | {stats['multiple']:,} | 结构不明，原样保留 |",
          f"| 无分隔符 | {stats['none']:,} | 不需要处理 |",
          "", f"合计含分隔符 **{quote_total:,}** 条，剥离掉的字符共 **{stripped_chars:,}** 个。",
          "后三种会打 `quote_*` 标记，第 7 步组样本时排除。"]
    if examples:
        L += ["", "剥离样例（前 3 条）：", ""]
        for sender, quote, body in examples:
            q = quote.replace("\n", "\\n")[:60]
            b = body.replace("\n", "\\n")[:40]
            L.append(f"- `{sender}`　引用 `{q}` → 正文 `{b}`")

    L += ["", "## 下一步", "", "第 5 步：说话人归因与会话切分（`pipeline/attribute.py`）", ""]

    rep = resolve(args.reports)
    rep.mkdir(parents=True, exist_ok=True)
    md = rep / "4-normalize.md"
    md.write_text("\n".join(L), encoding="utf-8")

    print(f"条数　　　：{n_in:,} → {len(ordered):,}")
    if ts_only:
        print(f"时间　　　：{ts_only[0]:%Y-%m-%d} → {ts_only[-1]:%Y-%m-%d}，升序 {'是' if ascending else '**否**'}")
    print(f"系统消息　：{n_system} 条")
    print(f"引用块　　：剥离 {stats['stripped']:,}｜后段空 {stats['no_reply']:,}｜非引用 {stats['no_quote']:,}｜多分隔符 {stats['multiple']:,}")
    print(f"闸门　　　：{'✅ 通过' if all(g[1] for g in gates) else '❌ 未通过'}")
    print(f"输出　　　：{out}")
    print(f"报告　　　：{md}")
    return 0 if all(g[1] for g in gates) else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
