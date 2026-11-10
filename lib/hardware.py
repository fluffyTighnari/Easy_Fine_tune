#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""硬件与依赖探测。只采集事实，不做任何训练相关决策（选型在第 8 步）。"""
from __future__ import annotations

import importlib
import json
import os
import platform
import shutil
import socket
import sys
import urllib.request
from pathlib import Path

LEARN = Path(__file__).resolve().parents[1]
PACKAGES = ["torch", "transformers", "peft", "datasets", "bitsandbytes", "llamafactory", "safetensors"]

TIERS = {
    "T3": ("宽裕", "VRAM ≥ 15 GB", ["4B QLoRA", "8B QLoRA", "4B 全参 LoRA"]),
    "T2": ("标准", "VRAM 7~15 GB", ["1.7B QLoRA", "4B QLoRA（bs=1~2）"]),
    "T1": ("能训小模型", "VRAM 3.5~7 GB", ["0.6B QLoRA", "1.7B QLoRA（bs=1）"]),
    "T0": ("不能训", "无可用 GPU", ["租卡训练", "不训：改用 API + persona prompt"]),
}


def _pkg_version(name: str) -> str | None:
    try:
        return getattr(importlib.import_module(name), "__version__", "?")
    except Exception:
        return None


def _ram_gb() -> tuple[float | None, float | None]:
    try:
        import psutil

        vm = psutil.virtual_memory()
        return round(vm.total / 1e9, 1), round(vm.available / 1e9, 1)
    except Exception:
        return None, None


def _disk_gb(path: Path) -> tuple[float | None, float | None]:
    try:
        import psutil

        du = psutil.disk_usage(str(path))
        return round(du.free / 1e9, 1), round(du.total / 1e9, 1)
    except Exception:
        return None, None


def _accel() -> dict:
    """返回后端与显存。torch 导入失败也不抛异常。"""
    try:
        import torch
    except Exception as e:
        return {"backend": "cpu", "error": f"torch 不可用：{e}"}

    if torch.cuda.is_available():
        try:
            p = torch.cuda.get_device_properties(0)
            try:
                free, total = torch.cuda.mem_get_info()
            except Exception:
                free, total = 0, p.total_memory
            return {
                "backend": "cuda",
                "name": p.name,
                "cc": f"{p.major}.{p.minor}",
                "vram_total_gb": round(total / 1e9, 2),
                "vram_free_gb": round(free / 1e9, 2),
            }
        except Exception as e:
            return {"backend": "cuda", "error": f"读取 GPU 信息失败：{e}"}

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return {"backend": "mps", "name": "Apple Silicon (unified memory)"}

    return {"backend": "cpu", "name": platform.processor() or "unknown"}


def _ollama() -> dict:
    """不执行 ollama 命令（它可能尝试拉起服务并写日志），只查可执行文件与 HTTP 端口。"""
    exe = shutil.which("ollama")
    out = {"installed": bool(exe), "path": exe, "server": False, "version": None}
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/version", timeout=2) as r:
            out["server"] = True
            out["version"] = json.loads(r.read().decode()).get("version")
    except Exception:
        pass
    return out


def _tier(accel: dict, ram_gb: float | None) -> tuple[str, str]:
    backend = accel.get("backend")
    if backend == "cuda":
        free = accel.get("vram_free_gb") or 0
        if free >= 15:
            return "T3", f"可用显存 {free} GB"
        if free >= 7:
            return "T2", f"可用显存 {free} GB"
        if free >= 3.5:
            return "T1", f"可用显存 {free} GB"
        return "T0", f"可用显存仅 {free} GB"
    if backend == "mps":
        if ram_gb and ram_gb >= 24:
            return "T2", f"统一内存 {ram_gb} GB"
        if ram_gb and ram_gb >= 16:
            return "T1", f"统一内存 {ram_gb} GB"
        return "T0", f"统一内存仅 {ram_gb} GB"
    return "T0", "没有可用的 GPU"


def probe() -> dict:
    ram_total, ram_free = _ram_gb()
    disk_free, disk_total = _disk_gb(LEARN)
    accel = _accel()
    tier, tier_reason = _tier(accel, ram_total)

    budget = None
    if accel.get("backend") == "cuda" and accel.get("vram_free_gb"):
        budget = round(accel["vram_free_gb"] * 0.85 - 0.6, 2)

    return {
        "os": f"{platform.system()} {platform.release()}",
        "python": f"{sys.version.split()[0]} ({sys.executable})",
        "in_venv": sys.prefix != sys.base_prefix,
        "cpu": platform.processor() or "unknown",
        "ram_gb": ram_total,
        "ram_free_gb": ram_free,
        "disk_free_gb": disk_free,
        "disk_total_gb": disk_total,
        "accel": accel,
        "packages": {n: _pkg_version(n) for n in PACKAGES},
        "ollama": _ollama(),
        "models_dir": str(LEARN / "models"),
        "models_dir_exists": (LEARN / "models").exists(),
        "tier": tier,
        "tier_reason": tier_reason,
        "vram_budget_gb": budget,
        "paths": TIERS[tier][2],
    }
