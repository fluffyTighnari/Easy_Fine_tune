#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 9 步：拉取模型。下载到 models/<id>，然后校验分片完整性与配置一致性。

输入：work/generated/llamafactory.yaml（第 8 步选中的结果）、config/catalog.yaml
产出：models/<id>/、work/reports/9-fetch-model.md

用法（从 learn/ 目录）：
    ..\\.venv\\Scripts\\python.exe pipeline\\fetch_model.py
    ..\\.venv\\Scripts\\python.exe pipeline\\fetch_model.py --local-source ..\\models\\Qwen3-4B-Instruct-2507
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys
import time
from datetime import datetime
from pathlib import Path

LEARN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LEARN))

from lib.paths import CONFIG, DEFAULT_ROOT, Layout  # noqa: E402
sys.path.insert(0, str(LEARN))

REQUIRED_ANY = ("config.json",)
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")


def read_yaml_model(path: Path) -> str | None:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("model_name_or_path:"):
            return line.split(":", 1)[1].strip().strip("'\"")
    return None


def safetensors_info(p: Path) -> tuple[int, int]:
    """读 safetensors 头部，返回 (张量个数, 文件声明的字节长度)。失败抛异常。"""
    with open(p, "rb") as f:
        head = f.read(8)
        if len(head) < 8:
            raise ValueError("文件太短")
        n = struct.unpack("<Q", head)[0]
        if n <= 0 or n > 200 * 1024 * 1024:
            raise ValueError(f"头部长度不合理：{n}")
        meta = json.loads(f.read(n))
    tensors = [k for k in meta if k != "__metadata__"]
    end = max((v["data_offsets"][1] for k, v in meta.items() if k != "__metadata__"), default=0)
    return len(tensors), 8 + n + end


def check_model(dest: Path, cfg_expected: dict | None) -> dict:
    """返回 {ok, checks:[(名称, 通过, 说明)], shards, tensors, bytes}"""
    checks: list[tuple[str, bool, str]] = []
    shards: list[Path] = []
    tensors = total_bytes = 0

    for name in REQUIRED_ANY:
        p = dest / name
        checks.append((name, p.exists(), f"{p.stat().st_size} B" if p.exists() else "缺失"))

    idx = dest / "model.safetensors.index.json"
    if idx.exists():
        wm = json.loads(idx.read_text(encoding="utf-8"))["weight_map"]
        names = sorted(set(wm.values()))
        missing = [n for n in names if not (dest / n).exists()]
        checks.append(("index 列出的分片", not missing,
                       f"{len(names)} 个" + (f"，缺 {missing}" if missing else "，全部存在")))
        shards = [dest / n for n in names if (dest / n).exists()]
    else:
        single = dest / "model.safetensors"
        if single.exists():
            shards = [single]
            checks.append(("model.safetensors", True, "单文件"))
        else:
            checks.append(("权重文件", False, "既无 index 也无 model.safetensors"))

    bad = []
    for s in shards:
        try:
            n, _ = safetensors_info(s)
            tensors += n
        except Exception as e:
            bad.append(f"{s.name}: {e}")
        total_bytes += s.stat().st_size
    checks.append(("分片可解析", not bad, "全部可读" if not bad else "; ".join(bad)))

    cfg_path = dest / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if cfg_expected:
            same = all(cfg.get(k) == cfg_expected.get(k)
                       for k in ("num_hidden_layers", "hidden_size", "vocab_size") if k in cfg_expected)
            checks.append(("config 与目录一致", same,
                           f"layers={cfg.get('num_hidden_layers')} hidden={cfg.get('hidden_size')} "
                           f"vocab={cfg.get('vocab_size')}"))

    tok = [f for f in TOKENIZER_FILES if (dest / f).exists()]
    checks.append(("tokenizer 文件", bool(tok), ", ".join(tok) or "缺失"))

    return {"ok": all(c[1] for c in checks), "checks": checks,
            "shards": len(shards), "tensors": tensors, "bytes": total_bytes}


def main() -> int:
    ap = argparse.ArgumentParser(description="第 9 步：拉取模型")
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--root", default=DEFAULT_ROOT, help="输出根目录，默认 work/")
    ap.add_argument("--catalog", default="config/catalog.yaml")
    ap.add_argument("--yaml", default=None)
    ap.add_argument("--reports", default=None)
    ap.add_argument("--dest-root", default="models")
    ap.add_argument("--model-id", default=None, help="覆盖自动识别")
    ap.add_argument("--local-source", default=None, help="已经有一份模型时，从本地目录复制（硬链接，不占空间）")
    ap.add_argument("--mirror", default=None, help="HF 镜像，如 https://hf-mirror.com")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    _lay = Layout(args.root)
    _lay.ensure()
    args.yaml = args.yaml or str(_lay.yaml)
    args.reports = args.reports if getattr(args, "reports", None) else str(_lay.reports)

    _lay = Layout(args.root)
    _lay.ensure()
    args.yaml = args.yaml or str(_lay.yaml)

    def resolve(p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else LEARN / q

    import yaml

    cfg = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    catalog = {m["id"]: m for m in yaml.safe_load(resolve(args.catalog).read_text(encoding="utf-8"))["models"]}

    yaml_model = read_yaml_model(resolve(args.yaml))
    model_id = args.model_id or (Path(yaml_model).name if yaml_model else None)
    if not model_id:
        print("识别不到模型，先做第 8 步，或用 --model-id 指定")
        return 2

    entry = catalog.get(model_id, {})
    repo = entry.get("repo", model_id)
    dest = resolve(args.dest_root) / model_id

    mirror = args.mirror if args.mirror is not None else (cfg.get("fetch", {}) or {}).get("mirror", "")
    if mirror:
        os.environ["HF_ENDPOINT"] = mirror
    os.environ.setdefault("HF_HOME", str(LEARN / "cache" / "hf"))
    os.environ.pop("HF_HUB_OFFLINE", None)

    print(f"模型　　　：{model_id}")
    print(f"仓库　　　：{repo}" + (f"（镜像 {mirror}）" if mirror else "（官方）"))
    print(f"目标目录　：{dest}")

    if args.dry_run:
        print("dry-run，不下载")
        return 0

    t0 = time.time()
    src_desc = ""

    # ---- 1. 取文件 ----
    if args.local_source:
        src = Path(args.local_source)
        if not src.is_absolute():
            src = LEARN / src
        if not src.exists():
            print(f"本地源不存在：{src}")
            return 2
        dest.mkdir(parents=True, exist_ok=True)
        linked = copied = 0
        for f in src.iterdir():
            if not f.is_file():
                continue
            d = dest / f.name
            if d.exists():
                continue
            try:
                os.link(f, d)
                linked += 1
            except OSError:
                shutil.copy2(f, d)
                copied += 1
        src_desc = f"本地 {src}（硬链接 {linked}，复制 {copied}）"
    else:
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(repo_id=repo, local_dir=str(dest), max_workers=4,
                              ignore_patterns=["*.pt", "*.pth", "original/*", "*.md"])
            src_desc = f"HuggingFace{'（镜像 ' + mirror + '）' if mirror else ''}"
        except Exception as e:
            print(f"下载失败：{type(e).__name__}: {str(e)[:200]}")
            print("可试：--mirror https://hf-mirror.com　或　--local-source <已下载的目录>")
            return 3

    elapsed = time.time() - t0

    # ---- 2. 校验 ----
    expected = {k: entry.get(k) for k in ("num_hidden_layers", "hidden_size", "vocab_size")}
    expected = {k: v for k, v in expected.items() if v}
    # catalog 用的键名和 config.json 不同，映射一下
    if entry:
        expected = {"num_hidden_layers": entry.get("layers"), "hidden_size": entry.get("hidden"),
                    "vocab_size": entry.get("vocab")}
        expected = {k: v for k, v in expected.items() if v}

    r = check_model(dest, expected)
    size_gb = r["bytes"] / 1e9

    gates = [
        ("分片全部存在", any(c[0] == "index 列出的分片" and c[1] for c in r["checks"])
         or any(c[0] == "model.safetensors" and c[1] for c in r["checks"]),
         next((c[2] for c in r["checks"] if c[0].startswith("index")), "单文件")),
        ("分片可解析", any(c[0] == "分片可解析" and c[1] for c in r["checks"]),
         next((c[2] for c in r["checks"] if c[0] == "分片可解析"), "-")),
        ("tokenizer 完整", any(c[0] == "tokenizer 文件" and c[1] for c in r["checks"]),
         next((c[2] for c in r["checks"] if c[0] == "tokenizer 文件"), "-")),
        ("config 与目录一致", any(c[0] == "config 与目录一致" and c[1] for c in r["checks"]) or not expected,
         next((c[2] for c in r["checks"] if c[0] == "config 与目录一致"), "无期望值，跳过")),
    ]

    # ---- 3. 报告 ----
    files = sorted([f for f in dest.iterdir() if f.is_file()], key=lambda p: -p.stat().st_size)[:12]
    disk_free = round(shutil.disk_usage(dest).free / 1e9, 1)

    L = [
        "# 第 9 步：拉取模型",
        "",
        f"生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        "## 结论",
        "",
        f"- 模型：**{model_id}**",
        f"- 来源：{src_desc}",
        f"- 大小：**{size_gb:.2f} GB**（{r['shards']} 个分片，{r['tensors']:,} 个张量）",
        f"- 耗时：{elapsed:.0f} 秒",
        f"- 目录：`{dest.as_posix()}`",
        f"- 剩余磁盘：{disk_free} GB",
        f"- 闸门：**{'✅ 通过' if all(g[1] for g in gates) else '❌ 未通过'}**",
        "",
        "| 闸门 | 结果 | 实测 |",
        "|---|---|---|",
    ]
    for name, passed, detail in gates:
        L.append(f"| {name} | {'✅' if passed else '❌'} | {detail} |")

    L += ["", "## 校验明细", "", "| 检查项 | 结果 | 说明 |", "|---|---|---|"]
    for name, ok, detail in r["checks"]:
        L.append(f"| {name} | {'✅' if ok else '❌'} | {detail} |")

    L += ["", "## 文件", "", "| 文件 | 大小 |", "|---|---|"]
    for f in files:
        L.append(f"| `{f.name}` | {f.stat().st_size / 1e6:.1f} MB |")

    L += ["", "## 下一步", "", "第 10 步：训练（`pipeline/train.py`）", ""]

    rep = Path(args.reports)
    rep.mkdir(parents=True, exist_ok=True)
    md = rep / "9-fetch-model.md"
    md.write_text("\n".join(L), encoding="utf-8")

    print(f"大小　　　：{size_gb:.2f} GB（{r['shards']} 分片，{r['tensors']:,} 张量）")
    print(f"耗时　　　：{elapsed:.0f} 秒　剩余磁盘 {disk_free} GB")
    print(f"闸门　　　：{'✅ 通过' if all(g[1] for g in gates) else '❌ 未通过'}")
    for name, passed, detail in gates:
        if not passed:
            print(f"  ❌ {name}：{detail}")
    print(f"报告　　　：{md}")
    return 0 if all(g[1] for g in gates) else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
