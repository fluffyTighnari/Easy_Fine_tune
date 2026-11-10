#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 2 步：环境体检。输出档位与可行路径。

用法（从 learn/ 目录）：
    ..\\.venv\\Scripts\\python.exe pipeline\\doctor.py
    ..\\.venv\\Scripts\\python.exe pipeline\\doctor.py --out test\\out\\reports
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

LEARN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LEARN))

from lib.paths import DEFAULT_ROOT, Layout  # noqa: E402
sys.path.insert(0, str(LEARN))
from lib import hardware  # noqa: E402

TIER_DESC = hardware.TIERS


def render_md(d: dict) -> str:
    accel = d["accel"]
    label, crit, paths = TIER_DESC[d["tier"]]
    L: list[str] = [
        "# 第 2 步：环境体检",
        "",
        f"生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        "## 结论",
        "",
        f"- 档位：**{d['tier']} {label}**（{crit}）",
        f"- 依据：{d['tier_reason']}",
        f"- 可行路径：{'　/　'.join(paths)}",
    ]
    if d["vram_budget_gb"] is not None:
        L.append(f"- 显存预算：**{accel['vram_free_gb']} GB × 0.85 − 0.6 = {d['vram_budget_gb']} GB**")

    L += ["", "## 机器", "", "| 项 | 值 |", "|---|---|",
          f"| 系统 | {d['os']} |",
          f"| Python | `{d['python']}` |",
          f"| 虚拟环境 | {'是' if d['in_venv'] else '**否**'} |",
          f"| CPU | {d['cpu']} |",
          f"| 内存 | {d['ram_free_gb']} GB 可用 / {d['ram_gb']} GB 总 |",
          f"| 磁盘 | {d['disk_free_gb']} GB 可用 / {d['disk_total_gb']} GB 总 |",
          f"| models/ | {'存在' if d['models_dir_exists'] else '**不存在**'} |"]

    L += ["", "## 加速器", "", "| 项 | 值 |", "|---|---|",
          f"| 后端 | `{accel.get('backend')}` |",
          f"| 设备 | {accel.get('name', '-')} |"]
    if accel.get("cc"):
        L.append(f"| 算力 | {accel['cc']} |")
    if accel.get("vram_total_gb"):
        L += [f"| 显存总量 | {accel['vram_total_gb']} GB |",
              f"| 显存可用 | {accel['vram_free_gb']} GB |"]
    if accel.get("error"):
        L.append(f"| 错误 | {accel['error']} |")

    L += ["", "## 依赖", "", "| 包 | 版本 | 状态 |", "|---|---|---|"]
    for name, ver in d["packages"].items():
        L.append(f"| `{name}` | {ver or '-'} | {'✅' if ver else '❌ 未安装'} |")

    o = d["ollama"]
    L += ["", "## Ollama", "", "| 项 | 值 |", "|---|---|",
          f"| 已安装 | {'✅ ' + str(o['path']) if o['installed'] else '❌ 未安装'} |",
          f"| 服务运行中 | {'✅ 版本 ' + str(o['version']) if o['server'] else '❌ 未运行（第 13 步才需要）'} |"]

    if d["ram_free_gb"] is not None and d["ram_free_gb"] < 8:
        L += ["", f"> ⚠️ 可用内存只有 {d['ram_free_gb']} GB。第 12 步合并模型需要 8 GB 以上空闲内存，"
                  "训练前请先关掉占内存的程序。"]

    L += ["", "## 下一步", "", "第 3 步：读入与体检（`pipeline/intake.py`）", ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description="第 2 步：环境体检")
    ap.add_argument("--root", default=DEFAULT_ROOT, help="输出根目录，默认 work/")
    ap.add_argument("--out", default=None, help="报告目录")
    ap.add_argument("--json", action="store_true", help="同时输出机器可读的 json")
    args = ap.parse_args()

    _lay = Layout(args.root)
    _lay.ensure()
    args.out = args.out or str(_lay.reports)

    d = hardware.probe()
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = LEARN / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "2-doctor.md"
    md_path.write_text(render_md(d), encoding="utf-8")

    accel = d["accel"]
    label, crit, paths = TIER_DESC[d["tier"]]
    print(f"档位　　　：{d['tier']} {label}（{crit}）")
    print(f"依据　　　：{d['tier_reason']}")
    print(f"加速器　　：{accel.get('backend')} / {accel.get('name', '-')}")
    if d["vram_budget_gb"] is not None:
        print(f"显存预算　：{accel['vram_free_gb']} GB 可用 → 预算 {d['vram_budget_gb']} GB")
    print(f"可行路径　：{'　/　'.join(paths)}")
    missing = [k for k, v in d["packages"].items() if not v]
    if missing:
        print(f"缺失依赖　：{', '.join(missing)}")
    print(f"Ollama　　：{'已安装' if d['ollama']['installed'] else '未安装'}，服务{'运行中' if d['ollama']['server'] else '未运行'}")
    if d["ram_free_gb"] is not None and d["ram_free_gb"] < 8:
        print(f"⚠️ 　　　：可用内存仅 {d['ram_free_gb']} GB（第 12 步需要 8 GB 以上）")
    print(f"报告　　　：{md_path}")

    if args.json:
        jp = out_dir / "2-doctor.json"
        jp.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"机器快照　：{jp}")

    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
