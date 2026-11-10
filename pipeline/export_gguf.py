#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 12 步：导出 GGUF。合并 LoRA → 转 GGUF → 量化。

输入：work/adapter_best/（或 work/adapter/）、models/<name>/
产出：work/merged/、work/gguf/<name>-f16.gguf、<name>-Q4_K_M.gguf

用法（从 learn/ 目录）：
    ..\\.venv\\Scripts\\python.exe pipeline\export_gguf.py
    ..\\.venv\\Scripts\\python.exe pipeline\export_gguf.py --quant Q5_K_M
    ..\\.venv\\Scripts\\python.exe pipeline\export_gguf.py --skip-merge   # 已合并过
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

LEARN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LEARN))

from lib.paths import DEFAULT_ROOT, Layout  # noqa: E402
sys.path.insert(0, str(LEARN))

# llama.cpp 工具在外层 tools/ 下，换机器时指到自己的 llama.cpp 目录
LLAMA_DIR = LEARN.parent / "tools" / "llama"
CONVERT = LLAMA_DIR / "convert_hf_to_gguf.py"
QUANTIZE = LLAMA_DIR / "llama-quantize.exe"


def setup_env() -> None:
    tmp = LEARN / "cache" / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    os.environ["TMP"] = str(tmp)
    os.environ["TEMP"] = str(tmp)
    os.environ.setdefault("HF_HOME", str(LEARN / "cache" / "hf"))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["WANDB_MODE"] = "disabled"


def mem_gb() -> float:
    try:
        import psutil

        return round(psutil.virtual_memory().available / 1e9, 1)
    except Exception:
        return 0.0


def run(cmd: list[str], log: Path) -> int:
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    with open(log, "a", encoding="utf-8", errors="replace") as fh:
        fh.write(f"\n$ {' '.join(str(c) for c in cmd)}\n")
        fh.flush()
        rc = subprocess.call([str(c) for c in cmd], stdout=fh, stderr=subprocess.STDOUT,
                             cwd=str(LEARN.parent))
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description="第 12 步：导出 GGUF")
    ap.add_argument("--root", default=DEFAULT_ROOT, help="输出根目录，默认 work/")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--model", default="models/Qwen3-4B-Instruct-2507")
    ap.add_argument("--merged", default=None)
    ap.add_argument("--gguf-dir", default=None)
    ap.add_argument("--quant", default="Q4_K_M")
    ap.add_argument("--template", default="qwen3_nothink")
    ap.add_argument("--reports", default=None)
    ap.add_argument("--skip-merge", action="store_true")
    ap.add_argument("--skip-f16", action="store_true")
    args = ap.parse_args()

    _lay = Layout(args.root)
    _lay.ensure()
    args.adapter = args.adapter or str(_lay.adapter_best)
    args.merged = args.merged or str(_lay.merged)
    args.gguf_dir = args.gguf_dir or str(_lay.gguf)
    args.reports = args.reports or str(_lay.reports)

    def resolve(p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else LEARN / q

    setup_env()
    adapter, model = resolve(args.adapter), resolve(args.model)
    # 没跑扫描就导出时，用训练目录里最后那一份，别直接报错卡住
    if not adapter.exists() and Path(args.adapter).name == "adapter_best":
        fallback = Layout(args.root).adapter
        if fallback.exists():
            print(f"没有 adapter_best，改用 {fallback}")
            adapter = fallback
    merged, gdir = resolve(args.merged), resolve(args.gguf_dir)
    gdir.mkdir(parents=True, exist_ok=True)
    logs = _lay.logs
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / "export.log"
    if log.exists():
        log.unlink()

    checks: list[tuple[str, bool, str]] = []
    checks.append(("适配器存在", (adapter / "adapter_config.json").exists(), adapter.as_posix()))
    checks.append(("基座模型存在", (model / "config.json").exists(), model.name))
    checks.append(("转换脚本存在", CONVERT.exists(), CONVERT.as_posix()))
    checks.append(("量化程序存在", QUANTIZE.exists(), QUANTIZE.name))
    avail = mem_gb()
    need = 2.5  # 流式合并，峰值约 2 GB，和模型大小无关
    checks.append(("内存够合并", avail >= need, f"{avail} GB 可用 / 建议 ≥ {need} GB（流式合并）"))
    for name, ok, detail in checks:
        print(f"  {'✅' if ok else '❌'} {name}：{detail}")
    if not all(c[1] for c in checks):
        print("预检未通过")
        return 2

    tag = model.name.replace("-Instruct-2507", "")
    times: dict[str, float] = {}
    err = ""

    # ---- 1. 合并 LoRA ----
    if not args.skip_merge:
        print(f"\n[1/3] 合并 LoRA → {merged}（流式，内存占用约 2 GB）")
        t0 = time.time()
        rc = run([sys.executable, LEARN / "pipeline" / "merge_lora.py", model, adapter, merged], log)
        times["merge"] = time.time() - t0
        if rc != 0:
            err = "合并失败，看 export.log"
            print(f"❌ {err}")
            return 1
    else:
        print(f"\n[1/3] 跳过合并，用现成的 {merged}")

    # ---- 2. 转 F16 GGUF ----
    f16 = gdir / f"{tag}-f16.gguf"
    if not args.skip_f16:
        print(f"\n[2/3] 转 GGUF（f16）→ {f16.name}")
        t0 = time.time()
        rc = run([sys.executable, CONVERT, merged, "--outfile", f16, "--outtype", "f16"], log)
        times["f16"] = time.time() - t0
        if rc != 0:
            err = "转 GGUF 失败，看 export.log"
            print(f"❌ {err}")
            return 1
    else:
        print(f"\n[2/3] 跳过转换，用现成的 {f16.name}")

    # ---- 3. 量化 ----
    q = gdir / f"{tag}-{args.quant}.gguf"
    print(f"\n[3/3] 量化 {args.quant} → {q.name}")
    t0 = time.time()
    rc = run([QUANTIZE, f16, q, args.quant], log)
    times["quant"] = time.time() - t0
    if rc != 0:
        err = "量化失败，看 export.log"
        print(f"❌ {err}")
        return 1

    # ---- 校验 ----
    sz = {p.name: p.stat().st_size for p in (f16, q) if p.exists()}
    magic = {}
    for p in (f16, q):
        if p.exists():
            with open(p, "rb") as fh:
                magic[p.name] = fh.read(4) == b"GGUF"
    gates = [
        ("合并产物完整", (merged / "config.json").exists() and
         len(list(merged.glob("*.safetensors"))) >= 1,
         f"{len(list(merged.glob('*.safetensors')))} 个分片"),
        ("f16 GGUF 有 GGUF 头", magic.get(f16.name, False), f"{sz.get(f16.name, 0) / 1e9:.2f} GB"),
        (f"{args.quant} 已生成", q.exists() and magic.get(q.name, False),
         f"{sz.get(q.name, 0) / 1e9:.2f} GB"),
        ("量化后不小于 f16 的 1/4", sz.get(q.name, 0) > sz.get(f16.name, 0) * 0.25,
         f"{sz.get(q.name, 0) / max(1, sz.get(f16.name, 1)):.0%}"),
    ]

    L = ["# 第 12 步：导出 GGUF", "", f"生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}", "",
         "## 结论", "",
         f"- 基座：`{model.name}` + `{Path(adapter).name}`",
         f"- 合并：`{merged.as_posix()}`",
         f"- f16：**{sz.get(f16.name, 0) / 1e9:.2f} GB**　量化 {args.quant}：**{sz.get(q.name, 0) / 1e9:.2f} GB**",
         f"- 耗时：合并 {times.get('merge', 0) / 60:.1f} 分钟 / 转换 {times.get('f16', 0) / 60:.1f} 分钟 / "
         f"量化 {times.get('quant', 0) / 60:.1f} 分钟",
         f"- 闸门：**{'✅ 通过' if all(g[1] for g in gates) else '❌ 未通过'}**", "",
         "| 闸门 | 结果 | 实测 |", "|---|---|---|"]
    for name, ok, detail in gates:
        L.append(f"| {name} | {'✅' if ok else '❌'} | {detail} |")
    L += ["", "## 预检", "", "| 检查项 | 结果 | 说明 |", "|---|---|---|"]
    for name, ok, detail in checks:
        L.append(f"| {name} | {'✅' if ok else '❌'} | {detail} |")
    L += ["", "## 文件", "", "| 文件 | 大小 |", "|---|---|"]
    for name, b in sz.items():
        L.append(f"| `{name}` | {b / 1e9:.2f} GB |")
    if err:
        L += ["", "## 错误", "", f"```\n{err}\n```", "", f"日志：`{log.as_posix()}`"]
    L += ["", "## 下一步", "", "第 13 步：写 Modelfile 并 `ollama create`。", ""]

    rep = resolve(args.reports)
    rep.mkdir(parents=True, exist_ok=True)
    md = rep / "12-export.md"
    md.write_text("\n".join(L), encoding="utf-8")

    print(f"\nf16　　　：{sz.get(f16.name, 0) / 1e9:.2f} GB")
    print(f"{args.quant}　：{sz.get(q.name, 0) / 1e9:.2f} GB")
    print(f"闸门　　　：{'✅ 通过' if all(g[1] for g in gates) else '❌ 未通过'}")
    print(f"报告　　　：{md}")
    return 0 if all(g[1] for g in gates) else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
