#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 5 步：说话人归因与会话切分。

输入：work/data/messages.jsonl + config（self_names、session_gap_min）
产出：work/data/attributed.jsonl、work/reports/5-attribute.md

用法（从 learn/ 目录）：
    ..\\.venv\\Scripts\\python.exe pipeline\\attribute.py
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

from lib.paths import CONFIG, DEFAULT_ROOT, Layout  # noqa: E402
sys.path.insert(0, str(LEARN))


def load_cfg(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def pct(vals: list[int], q: float) -> int:
    if not vals:
        return 0
    v = sorted(vals)
    return v[min(len(v) - 1, int(len(v) * q))]


def main() -> int:
    ap = argparse.ArgumentParser(description="第 5 步：说话人归因与会话切分")
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--root", default=DEFAULT_ROOT, help="输出根目录，默认 work/")
    ap.add_argument("--data", default=None)
    ap.add_argument("--reports", default=None)
    ap.add_argument("--as", dest="as_name", default=None,
                    help="数据里的说话人和配置不一致时，直接指定目标人物")
    args = ap.parse_args()

    _lay = Layout(args.root)
    _lay.ensure()
    args.data = args.data or str(_lay.data)
    args.reports = args.reports or str(_lay.reports)

    def resolve(p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else LEARN / q

    cfg = load_cfg(resolve(args.config))
    self_names = set(cfg["persona"]["self_names"])
    gap_min = float(cfg.get("data", {}).get("session_gap_min", 30))

    src = resolve(args.data) / "messages.jsonl"
    if not src.exists():
        print(f"输入不存在：{src}（先做第 4 步）")
        return 2
    rows = [json.loads(line) for line in open(src, encoding="utf-8") if line.strip()]

    # ---- 1. 归因 ----
    natural = Counter(r["sender"] for r in rows if r["role"] != "system")
    # 配置里的名字和数据对不上时（改名、匿名的数据），直接问一次，别让人猜
    if natural and not (set(natural) & self_names):
        print(f"配置里的 self_names={sorted(self_names)} 在数据里一个都没出现。")
        print("数据里的说话人：" + "、".join(f"{n}（{c:,} 条）" for n, c in natural.most_common()))
        print("用 --as 指定目标人物，或改配置里的 self_names。")
        as_name = getattr(args, "as_name", None)
        if as_name and as_name in natural:
            self_names = {as_name}
            print(f"按 --as 改成 {as_name}")
        else:
            return 2
    is_group = len(natural) > 2
    for r in rows:
        if r["role"] == "system":
            continue
        r["role"] = "me" if r["sender"] in self_names else "them"
        if is_group:
            r["flags"].append("group")

    # ---- 2. 会话切分 + 轮次 ----
    sid_no = 0
    prev_ts: datetime | None = None
    prev_role: str | None = None
    turn = 0
    for r in rows:
        ts = datetime.fromisoformat(r["ts"]) if r["ts"] else prev_ts
        new_session = (sid_no == 0) or (prev_ts is not None and ts is not None
                                        and (ts - prev_ts).total_seconds() / 60 > gap_min)
        if new_session:
            sid_no += 1
            turn = 0
            prev_role = None
        if r["role"] != "system" and r["role"] != prev_role:
            turn += 1
        r["sid"] = f"s{sid_no:04d}"
        r["turn"] = turn
        if ts is not None:
            prev_ts = ts
        if r["role"] != "system":
            prev_role = r["role"]

    # ---- 3. 连发统计 ----
    bursts: list[int] = []
    cur_role, cur_len = None, 0
    me_bursts: list[int] = []
    for r in rows:
        if r["role"] == "system":
            continue
        if r["role"] == cur_role:
            cur_len += 1
        else:
            if cur_role is not None:
                bursts.append(cur_len)
                if cur_role == "me":
                    me_bursts.append(cur_len)
            cur_role, cur_len = r["role"], 1
    if cur_role is not None:
        bursts.append(cur_len)
        if cur_role == "me":
            me_bursts.append(cur_len)

    # ---- 4. 会话规模 ----
    per_session = Counter(r["sid"] for r in rows)
    sess_sizes = list(per_session.values())
    sess_turns = [max((r["turn"] for r in rows if r["sid"] == s), default=0) for s in per_session]

    # ---- 5. 写出 ----
    out = resolve(args.data) / "attributed.jsonl"
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps({"mid": r["mid"], "sid": r["sid"], "turn": r["turn"],
                                "role": r["role"], "sender": r["sender"], "text": r["text"],
                                "ts": r["ts"], "flags": r["flags"],
                                **({"quote": r["quote"]} if r.get("quote") else {})},
                               ensure_ascii=False) + "\n")

    # ---- 6. 闸门 ----
    roles = Counter(r["role"] for r in rows)
    non_sys = roles["me"] + roles["them"]
    me_ratio = roles["me"] / non_sys if non_sys else 0
    gates = [
        ("me 占比 10%~90%", 0.10 <= me_ratio <= 0.90, f"{me_ratio:.1%}"),
        ("me 条数 ≥ 300", roles["me"] >= 300, f"{roles['me']:,} 条"),
        ("有会话切分", len(per_session) >= 1, f"{len(per_session):,} 个会话"),
    ]

    # ---- 7. 报告 ----
    L = [
        "# 第 5 步：说话人归因与会话切分",
        "",
        f"生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        "## 结论",
        "",
        f"- 目标人物：**{cfg['persona']['name']}**（数据里认作 me 的名字：{sorted(self_names)}）",
        f"- `me`（目标人物）：**{roles['me']:,}** 条，占非系统消息 **{me_ratio:.1%}**",
        f"- `them`：{roles['them']:,} 条　`system`：{roles['system']:,} 条",
        f"- 会话：**{len(per_session):,}** 个（间隔 > {gap_min:g} 分钟切分）",
        f"- 闸门：**{'✅ 通过' if all(g[1] for g in gates) else '❌ 未通过'}**",
        "",
        "| 闸门 | 结果 | 实测 |",
        "|---|---|---|",
    ]
    for name, passed, detail in gates:
        L.append(f"| {name} | {'✅' if passed else '❌'} | {detail} |")

    L += ["", "## 归因", "", "| 说话人 | 原始条数 | 判定 |", "|---|---|---|"]
    for name, c in natural.most_common():
        L.append(f"| {name} | {c:,} | {'**me**（目标）' if name in self_names else 'them'} |")
    if roles["system"]:
        L.append(f"| 系统 | {roles['system']:,} | system |")
    if is_group:
        L += ["", f"> ⚠️ 检出 {len(natural)} 个自然人，按群聊处理（每条打 `group` 标记）。"]

    L += ["", "## 会话", "", "| 项 | 值 |", "|---|---|",
          f"| 会话数 | {len(per_session):,} |",
          f"| 每会话消息数 | p50 {pct(sess_sizes, .5)}　p90 {pct(sess_sizes, .9)}　max {max(sess_sizes) if sess_sizes else 0} |",
          f"| 每会话轮次 | p50 {pct(sess_turns, .5)}　p90 {pct(sess_turns, .9)}　max {max(sess_turns) if sess_turns else 0} |",
          f"| 平均每会话消息 | {len(rows) / max(1, len(per_session)):.1f} |"]

    L += ["", "## 连发", "",
          "连续同一个人说话算一次连发。第 7 步会把一次连发合并成一个训练目标。", "",
          "| 项 | 全部连发 | 其中 me 的连发 |", "|---|---|---|",
          f"| 次数 | {len(bursts):,} | {len(me_bursts):,} |",
          f"| p50 | {pct(bursts, .5)} | {pct(me_bursts, .5)} |",
          f"| p90 | {pct(bursts, .9)} | {pct(me_bursts, .9)} |",
          f"| max | {max(bursts) if bursts else 0} | {max(me_bursts) if me_bursts else 0} |",
          f"| 合计条数 | {sum(bursts):,} | {sum(me_bursts):,} |"]
    max_burst_cfg = int(cfg.get("data", {}).get("max_burst", 0) or 0)
    if max_burst_cfg and me_bursts and max(me_bursts) > max_burst_cfg:
        L += ["", f"> 最长一次 me 连发 {max(me_bursts)} 条，超过 `max_burst`（{max_burst_cfg}）。"
                  f"第 7 步会按 {max_burst_cfg} 条截断，余下的并进下一个样本。"]

    L += ["", "## 下一步", "", "第 6 步：清洗（`pipeline/clean.py`）", ""]

    rep = resolve(args.reports)
    rep.mkdir(parents=True, exist_ok=True)
    md = rep / "5-attribute.md"
    md.write_text("\n".join(L), encoding="utf-8")

    print(f"目标人物　：{cfg['persona']['name']}")
    print(f"归因　　　：me {roles['me']:,}（{me_ratio:.1%}）｜them {roles['them']:,}｜system {roles['system']:,}")
    print(f"会话　　　：{len(per_session):,} 个，每会话 p50 {pct(sess_sizes, .5)} 条")
    print(f"连发　　　：{len(bursts):,} 次，me 的连发 {len(me_bursts):,} 次（p90 {pct(me_bursts, .9)} 条，max {max(me_bursts) if me_bursts else 0} 条）")
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
