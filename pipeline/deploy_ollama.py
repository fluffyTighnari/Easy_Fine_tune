#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 13 步：部署到 Ollama。写 Modelfile → ollama create → 调 API 验证。

用法（从 learn/ 目录）：
    ..\\.venv\\Scripts\\python.exe pipeline\deploy_ollama.py
    ..\\.venv\\Scripts\\python.exe pipeline\deploy_ollama.py --name persona-me --no-create
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

LEARN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LEARN))

from lib.paths import CONFIG, DEFAULT_ROOT, Layout  # noqa: E402
sys.path.insert(0, str(LEARN))

import yaml  # noqa: E402

OLLAMA_API = "http://127.0.0.1:11434"

# Qwen3 系列的对话壳子（非思考模式）
QWEN_TEMPLATE = """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ range .Messages }}{{ if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
<|im_start|>assistant
{{ else if eq .Role "assistant" }}{{ .Content }}<|im_end|>
{{ end }}{{ end }}"""

PROBES = [
    "明天要去面试好紧张 你说我要不要提前一天去看看场地",
    "我今天一共吃了三碗饭",
    "在吗",
    "刚跑完五公里 腿废了",
    "你觉得我该不该换工作",
]


def slug(s: str) -> str:
    s = re.sub(r"[^a-z0-9_.-]+", "-", s.lower()).strip("-")
    return s or "persona"


def ollama_available() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_API}/api/tags", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def chat(model: str, system: str, prompt: str, opts: dict) -> tuple[str, float]:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "stream": False,
        "options": opts,
    }).encode("utf-8")
    req = urllib.request.Request(f"{OLLAMA_API}/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        data = json.loads(r.read().decode("utf-8"))
    return (data.get("message", {}).get("content") or "").strip(), time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser(description="第 13 步：部署到 Ollama")
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--root", default=DEFAULT_ROOT, help="输出根目录，默认 work/")
    ap.add_argument("--gguf", default=None)
    ap.add_argument("--quant", default="Q4_K_M")
    ap.add_argument("--name", default=None, help="Ollama 模型名，默认取 config 的 deploy.model_name")
    ap.add_argument("--out", default=None)
    ap.add_argument("--reports", default=None)
    ap.add_argument("--no-create", action="store_true", help="只写 Modelfile，不执行 ollama create")
    args = ap.parse_args()

    _lay = Layout(args.root)
    _lay.ensure()
    args.gguf = args.gguf or str(_lay.gguf)
    args.out = args.out or str(_lay.deploy)
    args.reports = args.reports or str(_lay.reports)

    def resolve(p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else LEARN / q

    cfg = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8")) or {}
    persona = cfg.get("persona", {})
    dep = cfg.get("deploy", {})
    system = (persona.get("system_prompt") or "").strip()
    name = args.name or slug(dep.get("model_name") or persona.get("name") or "persona")
    opts = {
        "num_ctx": int(dep.get("num_ctx", 4096)),
        "temperature": float(dep.get("temperature", 0.7)),
        "top_p": float(dep.get("top_p", 0.9)),
        "repeat_penalty": float(dep.get("repeat_penalty", 1.1)),
        "num_predict": int(dep.get("num_predict", 160)),
    }

    gdir = resolve(args.gguf)
    tag = None
    cands = sorted(gdir.glob(f"*-{args.quant}.gguf")) or sorted(gdir.glob("*.gguf"))
    if not cands:
        print(f"{gdir} 里没有 gguf，先跑第 12 步")
        return 2
    tag = cands[0]
    # Ollama 的 FROM 相对路径是按 Modelfile 所在目录算的，找不到就当成模型名报
    # 「invalid model name」。一律用绝对路径。
    gguf_rel = tag.resolve().as_posix()

    checks: list[tuple[str, bool, str]] = []
    checks.append(("GGUF 存在", tag.exists(), f"{tag.name} {tag.stat().st_size / 1e9:.2f} GB"))
    checks.append(("Ollama 在运行", ollama_available(), OLLAMA_API))
    for n, ok, d in checks:
        print(f"  {'✅' if ok else '❌'} {n}：{d}")
    if not all(c[1] for c in checks):
        print("预检未通过")
        return 2

    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    modelfile = out / "Modelfile"
    lines = [f"FROM {gguf_rel}", "", f'TEMPLATE """{QWEN_TEMPLATE}"""', "",
             f'SYSTEM """{system}"""', "", 'PARAMETER stop "<|im_end|>"']
    for k in ("num_ctx", "temperature", "top_p", "repeat_penalty", "num_predict"):
        lines.append(f"PARAMETER {k} {opts[k]}")
    modelfile.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nModelfile 已写：{modelfile}")

    rc = 0
    if not args.no_create:
        print(f"创建模型 {name} …")
        cmd = ["ollama", "create", name, "-f", str(modelfile)]
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            print(f"❌ ollama create 失败：{(proc.stderr or proc.stdout)[-800:]}")
            rc = 1
        else:
            print("✅ 创建成功")

    results = []
    if rc == 0 and not args.no_create:
        print("验证回复 …")
        for p in PROBES:
            try:
                text, sec = chat(name, system, p, opts)
                results.append({"prompt": p, "reply": text, "sec": round(sec, 2),
                                "chars": len(text), "ok": bool(text)})
                print(f"  · {p[:16]}… → {text[:40].replace(chr(10), ' / ')}")
            except Exception as e:
                results.append({"prompt": p, "reply": "", "sec": 0, "chars": 0, "ok": False,
                                "error": str(e)})
                print(f"  ❌ {e}")

    chars = [r["chars"] for r in results if r["chars"]]
    long_reply = sum(1 for c in chars if c > 120)
    gates = [
        ("模型已创建", rc == 0 and not args.no_create, name if not args.no_create else "跳过"),
        ("全部回复非空", bool(results) and all(r["ok"] for r in results), f"{len(results)} 条"),
        ("平均字数 ≤ 60", bool(chars) and sum(chars) / len(chars) <= 60,
         f"{sum(chars) / max(1, len(chars)):.0f} 字"),
        ("没有长篇大论", long_reply == 0, f"{long_reply} 条超 120 字"),
    ]

    L = ["# 第 13 步：部署到 Ollama", "", f"生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}", "",
         f"- 模型名：`{name}`", f"- 来源：`{tag.name}`（{tag.stat().st_size / 1e9:.2f} GB）",
         f"- Modelfile：`{modelfile.as_posix()}`",
         f"- 闸门：**{'✅ 通过' if all(g[1] for g in gates) else '❌ 未通过'}**", "",
         "| 闸门 | 结果 | 实测 |", "|---|---|---|"]
    for n, ok, d in gates:
        L.append(f"| {n} | {'✅' if ok else '❌'} | {d} |")
    L += ["", "## 参数", "", "| 参数 | 值 |", "|---|---|"]
    for k, v in opts.items():
        L.append(f"| {k} | {v} |")
    L += ["", "## 系统提示（会和训练时同一份）", "", "```", system, "```", ""]
    if results:
        L += ["## 实测回复", "", "| 对方说 | 回复 | 字数 | 耗时 |", "|---|---|---|---|"]
        for r in results:
            L.append(f"| {r['prompt']} | {r['reply'].replace(chr(10), ' / ') or '（空）'} | "
                     f"{r['chars']} | {r['sec']}s |")
        L.append("")
    L += ["## 怎么用", "", "```powershell", f"ollama run {name}", "```", "",
          "或走接口：", "", "```powershell",
          f'curl http://127.0.0.1:11434/api/chat -d \'{{"model":"{name}","messages":[{{"role":"user","content":"在吗"}}]}}\'',
          "```", ""]

    rep = resolve(args.reports)
    rep.mkdir(parents=True, exist_ok=True)
    md = rep / "13-deploy.md"
    md.write_text("\n".join(L), encoding="utf-8")
    print(f"\n闸门　　　：{'✅ 通过' if all(g[1] for g in gates) else '❌ 未通过'}")
    print(f"报告　　　：{md}")
    return 0 if all(g[1] for g in gates) else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
