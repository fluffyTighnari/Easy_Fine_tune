#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练子进程：只负责把一份 yaml 跑完。由 pipeline/train.py 调用，不要直接用。

为什么要单开一个进程：datasets 会把 arrow 缓存 mmap 住，同一进程里重试会在
Windows 上报 WinError 1224（无法在用户映射区域打开的文件上执行）。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

LEARN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LEARN))


def setup_env() -> None:
    tmp = LEARN / "cache" / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    os.environ["TMP"] = str(tmp)
    os.environ["TEMP"] = str(tmp)
    os.environ.setdefault("HF_HOME", str(LEARN / "cache" / "hf"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(LEARN / "cache" / "hf_datasets"))
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def main() -> int:
    if len(sys.argv) < 2:
        print("用法：_train_child.py <llamafactory.yaml> [早停 patience]")
        return 2
    setup_env()
    import yaml

    cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
    from llamafactory.train.tuner import run_exp

    callbacks = []
    if len(sys.argv) > 2 and int(sys.argv[2]) > 0:
        from transformers import EarlyStoppingCallback

        callbacks = [EarlyStoppingCallback(early_stopping_patience=int(sys.argv[2]))]

    run_exp(cfg, callbacks=callbacks)

    # 把显存峰值留给父进程（跨进程拿不到 torch 统计）
    try:
        import torch

        if torch.cuda.is_available():
            out = Path(cfg["output_dir"])
            out.mkdir(parents=True, exist_ok=True)
            _f, total = torch.cuda.mem_get_info()
            (out / "_vram.json").write_text(
                json.dumps({
                    "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                    "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2),
                    "device_total_gb": round(total / 1e9, 2),
                }), encoding="utf-8")
    except Exception:
        pass

    print("TRAIN-CHILD-OK")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
