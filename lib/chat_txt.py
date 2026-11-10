#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""chat.txt 的编解码器 —— 该格式的唯一权威实现。

格式定义与做法见 docs/step1-chat-txt.md。

所有使用者都只依赖这一个模块读写 chat.txt：

    from lib import chat_txt
    records, unparsed = chat_txt.parse_file("work/input/chat.txt")
    chat_txt.render(records, "out/chat.txt")
    chat_txt.verify(records, "out/chat.txt")     # 往返校验

也能直接当命令行检查器用（**第一步做完就靠它自检**）：

    python lib/chat_txt.py --check work/input/chat.txt

设计要点：
  * 解析**绝不修改**内容，只做「拆字段 + 还原转义」。
  * 解析**绝不排序**、绝不去重、绝不丢弃。认不出的行原样交回，由调用方记账。
  * 时间戳缺失是合法的，不报错。
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

# [时间戳] 前缀。允许 6~32 字符（"2026-4-1 9:05" 到 "2026-04-01 11:21:32.123" 都能吃）
_TS_PREFIX = re.compile(r"^\[(?P<ts>[^\[\]]{6,32})\]\s+(?P<rest>.*)$", re.DOTALL)
# 说话人：第一个全角或半角冒号之前的部分
_SPEAKER_SEP = re.compile(r"[：:]")
# 转义还原：\n -> 换行，\\ -> 反斜杠；其它 \x 原样保留
_UNESCAPE = re.compile(r"\\(.)", re.DOTALL)
_ESCAPES = {"n": "\n", "\\": "\\"}

WECHAT_EMOJI = re.compile(r"\[([^\[\]]{1,10})\]")
UNICODE_EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F\u200D]")

# 媒体类标签闭集：只有这些词出现在方括号里才算「非文本消息」，其余一律当表情
MEDIA_TAGS = {
    "图片", "视频", "语音", "文件", "链接", "位置", "动画表情", "表情",
    "语音通话", "视频通话", "名片", "转账", "红包", "小程序", "聊天记录",
    "音乐", "游戏", "合并转发", "接龙", "视频号", "直播", "公告", "收藏",
}


def escape(text: str) -> str:
    """内容 -> 单行安全文本。先转义反斜杠，再转义换行，顺序不能反。"""
    return text.replace("\\", "\\\\").replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n")


def unescape(text: str) -> str:
    """单行安全文本 -> 内容。单趟扫描，避免 \\\\n 被解释两次。"""
    return _UNESCAPE.sub(lambda m: _ESCAPES.get(m.group(1), m.group(0)), text)


def format_line(sender: str, content: str, sent_at: str | None = None) -> str:
    """一条消息 -> 一行 chat.txt。不改变 content 内容，只做转义。"""
    body = f"{sender}：{escape(content)}"
    return f"[{sent_at}] {body}" if sent_at else body


def parse_line(line: str) -> dict:
    """一行 chat.txt -> {sender, content, sent_at, flags}。

    永远不抛异常：认不出的行返回 ok=False 并带上原文，让调用方决定怎么记账。
    """
    raw = line.rstrip("\n").rstrip("\r")
    if not raw.strip():
        return {"ok": False, "reason": "blank", "raw": raw, "flags": []}

    sent_at = None
    body = raw
    m = _TS_PREFIX.match(raw)
    if m:
        sent_at, body = m.group("ts").strip(), m.group("rest")

    sep = _SPEAKER_SEP.search(body)
    if not sep:
        return {"ok": False, "reason": "no_colon", "raw": raw, "flags": []}

    sender = body[: sep.start()].strip()
    content = body[sep.end():]

    flags: list[str] = []
    if not sender:
        return {"ok": False, "reason": "empty_sender", "raw": raw, "flags": []}
    if len(sender) > 24:
        return {"ok": False, "reason": "sender_too_long", "raw": raw, "flags": []}
    if " " in sender and len(sender) > 12:
        flags.append("suspicious_sender")

    return {
        "ok": True,
        "sender": sender,
        "content": unescape(content),
        "sent_at": sent_at,
        "flags": flags,
        "raw": raw,
    }


def parse_file(path: str | Path) -> tuple[list[dict], list[dict]]:
    """读整个 chat.txt -> (records, unparsed)。records 保持文件中的原始顺序。"""
    records: list[dict] = []
    unparsed: list[dict] = []
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            rec = parse_line(line)
            rec["lineno"] = lineno
            (records if rec["ok"] else unparsed).append(rec)
    return records, unparsed


def render(records: list[dict], path: str | Path) -> int:
    """把 records 写成 chat.txt。返回行数。UTF-8、LF 换行、结尾无多余空行。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [format_line(r["sender"], r["content"], r.get("sent_at")) for r in records]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")
    return len(lines)


def verify(records_in: list[dict], path: str | Path) -> dict:
    """往返校验：写出去再读回来，逐条比对 sender / content / sent_at / 顺序。

    这是「起始格式足够表达原始数据」的证明。任何一项不等都算失败。
    """
    back, unparsed = parse_file(path)
    problems: list[str] = []

    if len(back) != len(records_in):
        problems.append(f"条数不等：写入 {len(records_in)}，读回 {len(back)}")
    if unparsed:
        problems.append(f"有 {len(unparsed)} 行解析失败，首个：{unparsed[0]['raw'][:80]!r}")

    diff = 0
    for i, (a, b) in enumerate(zip(records_in, back)):
        if a["sender"] != b["sender"]:
            problems.append(f"第 {i + 1} 条 sender 不符：{a['sender']!r} != {b['sender']!r}")
            diff += 1
        elif a["content"] != b["content"]:
            problems.append(f"第 {i + 1} 条 content 不符：{a['content'][:40]!r} != {b['content'][:40]!r}")
            diff += 1
        elif (a.get("sent_at") or None) != (b.get("sent_at") or None):
            problems.append(f"第 {i + 1} 条 sent_at 不符：{a.get('sent_at')!r} != {b.get('sent_at')!r}")
            diff += 1
        if diff > 5:
            problems.append("……（差异过多，已截断）")
            break

    return {"ok": not problems, "n_written": len(records_in), "n_read": len(back),
            "n_unparsed": len(unparsed), "problems": problems[:10]}


# ---------------------------------------------------------------------------
# 命令行：第一步做完就用它自检
# ---------------------------------------------------------------------------
def cli_check(path: Path) -> int:
    records, unparsed = parse_file(path)
    total = len(records) + len(unparsed)
    ratio = len(unparsed) / total if total else 1.0

    print(f"文件：{path}")
    print(f"总行数：{total:,}　成功解析：{len(records):,}　失败：{len(unparsed):,}")
    print()

    if records:
        speakers = Counter(r["sender"] for r in records)
        print("说话人分布：")
        for name, n in speakers.most_common(12):
            print(f"  {name:<14} {n:>6,}  {n / len(records):6.1%}")
        print()

        with_ts = sum(1 for r in records if r["sent_at"])
        print(f"有时间戳：{with_ts:,} / {len(records):,}（{with_ts / len(records):.1%}）")
        print(f"含微信方括号表情的消息：{sum(1 for r in records if WECHAT_EMOJI.search(r['content']) if r['content']):,}")
        print(f"含普通 Unicode emoji 的消息：{sum(1 for r in records if UNICODE_EMOJI.search(r['content'])):,}")
        print(f"含转义换行（\\n）的消息：{sum(1 for r in records if chr(10) in r['content']):,}")
        print()

        whole = [r for r in records if r["content"].strip().startswith("[") and r["content"].strip().endswith("]")]
        media = [r for r in whole if r["content"].strip()[1:-1] in MEDIA_TAGS]
        print(f"整条就是一个方括号标签的：{len(whole):,} 条，其中命中媒体闭集（本应已丢弃）：{len(media):,} 条")
        if media:
            print(f"  ⚠️ 这些媒体消息还在，第一步没做完：{sorted({r['content'].strip() for r in media})[:8]}")
        print()

        print("冒烟（前 3 条 / 后 3 条）：")
        for r in records[:3] + records[-3:]:
            print(f"  [{r['sent_at'] or '-'}] {r['sender']}：{r['content'][:44]!r}")
    print()

    if unparsed:
        print(f"❌ 有 {len(unparsed)} 行没通过解析（最多列 10 条）：")
        for r in unparsed[:10]:
            print(f"  第 {r['lineno']} 行 [{r['reason']}]：{r['raw'][:90]!r}")
        print("\n  最常见原因：说话人和内容之间少了冒号、或该行是微信复制时的多行结构没拼起来。")
        print("  看 docs/step1-chat-txt.md 的「报错对照」。")
    else:
        print("✅ 所有行都解析成功")

    if ratio > 0.05:
        print(f"\n❌ 解析失败率 {ratio:.1%} 超过 5% 的闸门，第一步不通过，先修数据。")
        return 1
    if unparsed:
        print(f"\n⚠️ 解析失败率 {ratio:.1%} 在 5% 容忍线内，可以继续，但建议修掉。")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="chat.txt 编解码器 / 自检工具")
    ap.add_argument("--check", metavar="FILE", help="检查一个 chat.txt 是否合规")
    args = ap.parse_args(argv)
    if not args.check:
        ap.print_help()
        return 2
    path = Path(args.check).resolve()
    if not path.exists():
        print(f"文件不存在：{path}")
        return 2
    return cli_check(path)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
