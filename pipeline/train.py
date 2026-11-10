#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 10 步：训练。先冒烟再全量；OOM 或多进程受限时自动降档重试。

输入：work/generated/llamafactory.yaml、work/sft/
产出：work/adapter/（或 adapter_smoke/）、work/reports/10-train*.md

用法（从 learn/ 目录）：
    ..\\.venv\\Scripts\\python.exe pipeline\\train.py --smoke        # 冒烟 6 步
    ..\\.venv\\Scripts\\python.exe pipeline\\train.py                # 全量训练
"""
from __future__ import annotations

import argparse
import json
import re
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

CHILD = LEARN / "pipeline" / "_train_child.py"
LOG_DIR = LEARN / "test" / "out" / "logs"


def tail_of_error(text: str) -> str:
    """从日志里挑出真正的报错段，别把进度条和 atexit 噪声当结论。"""
    lines = text.splitlines()
    idx = None
    for i, ln in enumerate(lines):
        low = ln.lower()
        if ("error" in low or "traceback" in low) and "winerror 5" not in low:
            idx = i
    if idx is None:
        return "\n".join(lines[-8:])
    return "\n".join(lines[idx:idx + 12])


def classify(text: str) -> str:
    low = text.lower()
    if "out of memory" in low or "outofmemoryerror" in low:
        return "oom"
    if "createfile" in low or "simplequeue" in low or "multiprocess" in low and "winerror 5" in low:
        return "multiprocess"
    if "winerror 1224" in low or "user-mapped" in low or "用户映射" in text:
        return "mmap"
    return "other"


def downgrade(cfg: dict, kind: str, see: str) -> str | None:
    """按失败类型降一档，返回说明；None 表示没救。"""
    if kind == "oom":
        bs = int(cfg.get("per_device_train_batch_size", 1))
        if bs <= 1:
            return None
        cfg["per_device_train_batch_size"] = bs // 2
        cfg["gradient_accumulation_steps"] = int(cfg.get("gradient_accumulation_steps", 1)) * 2
        return f"OOM：batch {bs} → {bs // 2}，梯度累积 ×2"
    if kind == "multiprocess":
        if int(cfg.get("preprocessing_num_workers", 1) or 1) > 1:
            cfg["preprocessing_num_workers"] = 1
            return "系统不允许建命名管道：preprocessing_num_workers → 1"
        if int(cfg.get("dataloader_num_workers", 0) or 0) > 0:
            cfg["dataloader_num_workers"] = 0
            return "系统不允许建命名管道：dataloader_num_workers → 0"
        return None
    if kind == "mmap":
        # 换一个进程已经能解决；若仍失败，把循环压到最小
        if int(cfg.get("preprocessing_num_workers", 1) or 1) > 1:
            cfg["preprocessing_num_workers"] = 1
            return "arrow 缓存被占用：改用单进程重跑"
        return None
    return None


def preflight(cfg: dict) -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []
    mp = Path(cfg["model_name_or_path"])
    idx = mp / "model.safetensors.index.json"
    if idx.exists():
        names = sorted(set(json.loads(idx.read_text(encoding="utf-8"))["weight_map"].values()))
        missing = [n for n in names if not (mp / n).exists()]
        checks.append(("模型分片", not missing, f"{len(names)} 个" + (f"，缺 {missing}" if missing else "")))
    else:
        ok = (mp / "model.safetensors").exists()
        checks.append(("模型权重", ok, "单文件" if ok else "缺 model.safetensors"))

    ddir = Path(cfg.get("dataset_dir", ""))
    checks.append(("dataset_info.json", (ddir / "dataset_info.json").exists(), str(ddir)))
    n_train = 0
    for name in ("train.jsonl", "val.jsonl"):
        p = ddir / name
        n = sum(1 for _ in open(p, encoding="utf-8")) if p.exists() else 0
        if name == "train.jsonl":
            n_train = n
        checks.append((name, p.exists() and n > 0, f"{n:,} 条"))

    steps = n_train * float(cfg.get("num_train_epochs", 1)) / (
        int(cfg.get("per_device_train_batch_size", 1)) * int(cfg.get("gradient_accumulation_steps", 1)))
    checks.append(("预计步数", steps >= 20, f"{steps:.0f} 步"))
    return checks


def vram() -> tuple[float, float]:
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return round(free / 1e9, 2), round(total / 1e9, 2)
    except Exception:
        pass
    return 0.0, 0.0


def read_log(out_dir: Path) -> list[dict]:
    """loss 有两个来源：pipeline 的 trainer_log.jsonl 和 HF 的 trainer_state.json。合并去重。"""
    rows: dict[int, dict] = {}
    p = out_dir / "trainer_log.jsonl"
    if p.exists():
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if isinstance(r.get("loss"), (int, float)):
                rows[int(r.get("current_steps") or r.get("global_step") or len(rows))] = r
    st = out_dir / "trainer_state.json"
    if st.exists():
        try:
            for h in json.loads(st.read_text(encoding="utf-8")).get("log_history", []):
                if isinstance(h.get("loss"), (int, float)):
                    step = int(h.get("step") or 0)
                    if step not in rows:
                        rows[step] = {**h, "current_steps": step}
        except Exception:
            pass
    return [rows[k] for k in sorted(rows)]


def main() -> int:
    ap = argparse.ArgumentParser(description="第 10 步：训练")
    ap.add_argument("--config", default=None)
    ap.add_argument("--root", default=DEFAULT_ROOT, help="输出根目录，默认 work/")
    ap.add_argument("--smoke", action="store_true", help="只跑几步，验证不 OOM 且能存盘")
    ap.add_argument("--steps", type=int, default=None, help="限制总步数（冒烟默认 6，全量不设）")
    ap.add_argument("--save-steps", type=int, default=None, help="覆盖存档间隔")
    ap.add_argument("--reports", default=None)
    ap.add_argument("--max-retry", type=int, default=3)
    ap.add_argument("--fresh", action="store_true", help="清空输出目录，从头训")
    ap.add_argument("--resume", default=None, help="指定 checkpoint 目录续训")
    ap.add_argument("--early-stop", type=int, default=2,
                    help="eval_loss 连续多少次不降就停（0 = 不早停）")
    args = ap.parse_args()

    _lay = Layout(args.root)
    _lay.ensure()
    args.config = args.config or str(_lay.yaml)
    args.reports = args.reports or str(_lay.reports)

    def resolve(p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else LEARN / q

    import yaml

    yaml_path = resolve(args.config)
    if not yaml_path.exists():
        print(f"配置不存在：{yaml_path}（先做第 8 步）")
        return 2
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

    checks = preflight(cfg)
    free, total = vram()
    print(f"显存　　　：{free} GB 可用 / {total} GB 总")
    for name, ok, detail in checks:
        print(f"  {'✅' if ok else '❌'} {name}：{detail}")
    if not all(c[1] for c in checks):
        print("预检未通过，先修上面对应的项")
        return 2

    run_cfg = dict(cfg)
    tag = "全量"
    if args.smoke:
        tag = f"冒烟 {args.steps or 6} 步"
        smoke_dir = LEARN / "test" / "out" / "adapter_smoke"
        if smoke_dir.exists():
            shutil.rmtree(smoke_dir, ignore_errors=True)
        run_cfg.update({
            "output_dir": smoke_dir.as_posix(),
            "overwrite_output_dir": True,
            "max_steps": args.steps or 6,
            "logging_steps": 1,
            "save_steps": args.save_steps or (args.steps or 6),
            "save_total_limit": 1,
            "eval_strategy": "no",
            "report_to": "none",
        })

    out_dir = Path(run_cfg["output_dir"])
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # ---- checkpoint 判定 ----
    ckpts = sorted([p for p in out_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")],
                   key=lambda p: int(p.name.split("-")[-1])) if out_dir.exists() else []
    done_before = (out_dir / "train_results.json").exists() and not ckpts
    resume_from = None
    if not args.smoke:
        if args.fresh and out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)
            ckpts = []
            print("--fresh：已清空输出目录")
        elif args.resume:
            rp = Path(args.resume)
            rp = rp if rp.is_absolute() else LEARN / rp
            if not rp.exists():
                print(f"指定的 checkpoint 不存在：{rp}")
                return 2
            resume_from = str(rp)
        elif done_before:
            print(f"这个目录已经训完过（{out_dir}）。要重训加 --fresh，要接着训加 --resume <checkpoint>")
            return 2
        elif ckpts:
            resume_from = str(ckpts[-1])
        run_cfg["overwrite_output_dir"] = False

    if args.steps and not args.smoke:
        run_cfg["max_steps"] = args.steps
    if args.save_steps:
        run_cfg["save_steps"] = args.save_steps
    # 短跑时自动加密日志，否则一条 loss 都拿不到
    if run_cfg.get("max_steps"):
        ms = int(run_cfg["max_steps"])
        run_cfg["logging_steps"] = max(1, min(int(run_cfg.get("logging_steps", 10)), max(1, ms // 10)))

    if resume_from:
        run_cfg["resume_from_checkpoint"] = resume_from
        print(f"检测到 checkpoint，续训：{Path(resume_from).name}")
    elif not args.smoke:
        print("没有可用 checkpoint，从头训")
    runtime = _lay.generated / "_runtime.yaml"

    print(f"\n开始{tag}，输出到 {out_dir}\n")
    t0 = time.time()
    notes: list[str] = []
    err_tail = ""
    ok = False

    for attempt in range(1, args.max_retry + 2):
        runtime.write_text(yaml.safe_dump(run_cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        log_path = LOG_DIR / f"train_try{attempt}.log"
        print(f"--- 第 {attempt} 次尝试，日志 {log_path.name} ---")
        interrupted = False
        with open(log_path, "w", encoding="utf-8", errors="replace") as fh:
            try:
                rc = subprocess.call([sys.executable, "-u", str(CHILD), str(runtime), str(args.early_stop)],
                                     stdout=fh, stderr=subprocess.STDOUT, cwd=str(LEARN))
            except KeyboardInterrupt:
                interrupted = True
                rc = -1
        if interrupted:
            ck = sorted([p.name for p in out_dir.iterdir()
                         if p.is_dir() and p.name.startswith("checkpoint-")]) if out_dir.exists() else []
            print(f"\n已中断。最近 checkpoint：{ck[-1] if ck else '无（还没到第一次存档）'}")
            print("重跑同一命令会自动从这个 checkpoint 续训。")
            return 130
        if rc == 0:
            ok = True
            break
        text = log_path.read_text(encoding="utf-8", errors="replace")
        # 分类看全文：日志尾部往往是 wandb 的 atexit 噪声，会把真正的报错挤出去
        kind = classify(text)
        err_tail = tail_of_error(text)
        note = downgrade(run_cfg, kind, err_tail)
        print(f"失败（{kind}）" + (f" → {note}" if note else "，无法自动降档"))
        if not note:
            break
        notes.append(f"{note}（第 {attempt} 次后）")

    elapsed = time.time() - t0
    peak = reserved = 0.0
    vf = out_dir / "_vram.json"
    if vf.exists():
        try:
            v = json.loads(vf.read_text(encoding="utf-8"))
            peak = v.get("peak_alloc_gb", 0.0)
            reserved = v.get("peak_reserved_gb", 0.0)
            total = v.get("device_total_gb", total)
        except Exception:
            pass

    logs = read_log(out_dir)
    losses = [r["loss"] for r in logs]
    steps_done = 0
    st = out_dir / "trainer_state.json"
    if st.exists():
        try:
            steps_done = json.loads(st.read_text(encoding="utf-8")).get("global_step", 0)
        except Exception:
            pass
    if not steps_done and logs:
        steps_done = logs[-1].get("current_steps") or logs[-1].get("global_step") or 0
    seg = max(1, len(losses) // 4)
    first, last = losses[:seg], losses[-seg:]
    m = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    no_nan = all(x == x for x in losses)
    dropped = m(last) < m(first) if losses else False

    adapter_files: list[str] = []
    if out_dir.exists():
        adapter_files = [p.name for p in out_dir.iterdir()
                         if p.is_file() and p.suffix in (".safetensors", ".json")]
        ck = sorted([p for p in out_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")])
        if ck and not adapter_files:
            adapter_files = [f"{ck[-1].name}/{p.name}" for p in ck[-1].iterdir()
                             if p.suffix in (".safetensors", ".json")]

    gates = [
        ("训练跑完（无异常退出）", ok, "正常结束" if ok else err_tail.strip().splitlines()[-1][:90]),
        ("loss 有效（无 NaN）", no_nan and bool(losses), f"{len(losses)} 条记录"),
        ("loss 下降", dropped or len(losses) < 4,
         f"首段 {m(first):.4f} → 末段 {m(last):.4f}" if len(losses) >= 4
         else f"只有 {len(losses)} 个点，不足以判断趋势"),
        ("适配器已存盘", bool(adapter_files), f"{len(adapter_files)} 个文件" if adapter_files else "无"),
    ]

    L = [
        f"# 第 10 步：训练（{tag}）",
        "",
        f"生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        "## 结论",
        "",
        f"- 模型：`{Path(cfg['model_name_or_path']).name}`",
        f"- 步数：**{steps_done}**　耗时 **{elapsed / 60:.1f} 分钟**"
        + (f"（{elapsed / max(1, steps_done):.1f} 秒/步）" if steps_done else ""),
        f"- 显存峰值：{peak} GB 分配 / {reserved} GB 保留（设备共 {total} GB）",
        f"- 输出：`{out_dir.as_posix()}`",
        f"- 闸门：**{'✅ 通过' if all(g[1] for g in gates) else '❌ 未通过'}**",
        "",
        "| 闸门 | 结果 | 实测 |",
        "|---|---|---|",
    ]
    for name, passed, detail in gates:
        L.append(f"| {name} | {'✅' if passed else '❌'} | {detail} |")

    L += ["", "## 预检", "", "| 检查项 | 结果 | 说明 |", "|---|---|---|"]
    for name, good, detail in checks:
        L.append(f"| {name} | {'✅' if good else '❌'} | {detail} |")

    L += ["", "## 实际配置", "", "| 项 | 值 |", "|---|---|"]
    for k in ("per_device_train_batch_size", "gradient_accumulation_steps", "cutoff_len",
              "learning_rate", "num_train_epochs", "lora_rank", "lora_alpha", "template",
              "quantization_bit", "preprocessing_num_workers", "dataloader_num_workers"):
        if k in run_cfg:
            L.append(f"| `{k}` | {run_cfg[k]} |")
    if "max_steps" in run_cfg:
        L.append(f"| `max_steps` | {run_cfg['max_steps']} |")
    ckpts_after = sorted([p.name for p in out_dir.iterdir()
                          if p.is_dir() and p.name.startswith("checkpoint-")]) if out_dir.exists() else []
    L += ["", "## checkpoint", "", "| 项 | 值 |", "|---|---|",
          f"| 起点 | {'续训 ' + Path(resume_from).name if resume_from else '从头训'} |",
          f"| save_steps | {run_cfg.get('save_steps', '-')}（每 {run_cfg.get('save_steps', '-')} 步存档）|",
          f"| save_total_limit | {run_cfg.get('save_total_limit', '-')}（只留最近 N 个）|",
          f"| 现存 checkpoint | {', '.join(ckpts_after) or '无'} |",
          f"| 中断后续训 | 重跑同一命令即可自动续训 |"]

    if notes:
        L += ["", "**自动降档记录**", ""] + [f"- {n}" for n in notes]

    final_train_loss = final_eval_loss = None
    if st.exists():
        try:
            for h in reversed(json.loads(st.read_text(encoding="utf-8")).get("log_history", [])):
                if final_train_loss is None and isinstance(h.get("train_loss"), (int, float)):
                    final_train_loss = h["train_loss"]
                if final_eval_loss is None and isinstance(h.get("eval_loss"), (int, float)):
                    final_eval_loss = h["eval_loss"]
                if final_train_loss is not None and final_eval_loss is not None:
                    break
        except Exception:
            pass

    if losses:
        L += ["", "## loss", "", "| 项 | 值 |", "|---|---|",
              f"| 最终 train_loss | {final_train_loss:.4f} |" if final_train_loss else "| 最终 train_loss | - |",
              f"| 最终 eval_loss | {final_eval_loss:.4f} |" if final_eval_loss else "| 最终 eval_loss | - |",
              f"| 首段平均 | {m(first):.4f} |",
              f"| 末段平均 | {m(last):.4f} |",
              f"| 最低 | {min(losses):.4f} |"]
        step_n = max(1, len(losses) // 10)
        L += ["", "| 步 | loss |", "|---|---|"]
        for i in range(0, len(losses), step_n):
            L.append(f"| {logs[i].get('current_steps', i) or i} | {losses[i]:.4f} |")
    else:
        L += ["", "## loss", "", "没有记录到 loss，看日志：", "", "```",
              (err_tail.strip().splitlines()[-6:] and "\n".join(err_tail.strip().splitlines()[-6:])) or "无",
              "```"]

    L += ["", "## 日志", ""] + [f"- `{p.as_posix()}`" for p in sorted(LOG_DIR.glob("train_try*.log"))]
    L += ["", "## 下一步", "",
          "冒烟通过后跑全量 `pipeline/train.py`，再做第 11 步评测（`pipeline/eval.py`）", ""]

    rep = resolve(args.reports)
    rep.mkdir(parents=True, exist_ok=True)
    md = rep / ("10-train-smoke.md" if args.smoke else "10-train.md")
    md.write_text("\n".join(L), encoding="utf-8")

    print(f"\n步数　　　：{steps_done}　耗时 {elapsed / 60:.1f} 分钟")
    if losses:
        print(f"loss　　　：首段 {m(first):.4f} → 末段 {m(last):.4f}（最低 {min(losses):.4f}）")
    print(f"闸门　　　：{'✅ 通过' if all(g[1] for g in gates) else '❌ 未通过'}")
    for name, passed, detail in gates:
        if not passed:
            print(f"  ❌ {name}：{detail}")
    print(f"报告　　　：{md}")
    return 0 if all(g[1] for g in gates) else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
