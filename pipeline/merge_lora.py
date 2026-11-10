#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""流式合并 LoRA 到基座：逐张量读、算、写，峰值内存约 2 GB。

为什么不用 `llamafactory-cli export`：它要把整个模型读进内存（4B 的 bf16 要 8 GB），
小内存机器直接失败。这里按分片流式处理，1 GB 一组落盘，内存占用跟模型大小无关。

输入：基座目录（HF 格式，可分片）、适配器目录（LoRA，peft 格式）
产出：合并后的 HF 目录（可直接喂给 convert_hf_to_gguf.py）
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

CHUNK_BYTES = 1_000_000_000  # 每写出约 1 GB 落一次盘


def load_adapter(adapter_dir: Path) -> tuple[dict, float]:
    """读 LoRA 权重，返回 {(层名, 模块): (A, B)} 与缩放系数。"""
    from safetensors import safe_open

    cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    r = int(cfg.get("r", 16))
    alpha = float(cfg.get("lora_alpha", r * 2))
    scale = alpha / r

    pairs: dict[str, dict] = {}
    f = (adapter_dir / "adapter_model.safetensors")
    with safe_open(str(f), framework="pt") as sf:
        for k in sf.keys():
            # base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
            if ".lora_A." in k:
                base = k.split(".lora_A.")[0]
                pairs.setdefault(base, {})["A"] = sf.get_tensor(k)
            elif ".lora_B." in k:
                base = k.split(".lora_B.")[0]
                pairs.setdefault(base, {})["B"] = sf.get_tensor(k)
    return pairs, scale


def to_base_name(peft_name: str) -> str:
    """peft 的键名转成基座权重名。

    base_model.model.model.layers.0.self_attn.q_proj -> model.layers.0.self_attn.q_proj
    """
    n = peft_name
    for pref in ("base_model.model.", "base_model."):
        if n.startswith(pref):
            n = n[len(pref):]
    return n


def merge(base_dir: Path, adapter_dir: Path, out_dir: Path) -> dict:
    from safetensors import safe_open
    from safetensors.torch import save_file

    pairs, scale = load_adapter(adapter_dir)
    deltas: dict[str, tuple] = {}
    for peft_name, ab in pairs.items():
        if "A" not in ab or "B" not in ab:
            continue
        # peft 的键到 .lora_A.weight 为止，基座权重名还要带 .weight
        modal = to_base_name(peft_name)
        deltas[modal if modal.endswith(".weight") else modal + ".weight"] = (ab["A"], ab["B"])

    idx_path = base_dir / "model.safetensors.index.json"
    if idx_path.exists():
        weight_map = json.loads(idx_path.read_text(encoding="utf-8"))["weight_map"]
        shards = sorted(set(weight_map.values()))
    else:
        single = base_dir / "model.safetensors"
        if not single.exists():
            raise SystemExit(f"{base_dir} 里没有 safetensors")
        shards = [single.name]
        weight_map = {}

    out_dir.mkdir(parents=True, exist_ok=True)
    applied = 0
    buf: dict = {}
    buf_bytes = 0
    new_map: dict[str, str] = {}
    part = 0

    def flush() -> None:
        nonlocal buf, buf_bytes, part
        if not buf:
            return
        part += 1
        name = f"model-merged-{part:05d}.safetensors"
        save_file(buf, str(out_dir / name), metadata={"format": "pt"})
        for k in buf:
            new_map[k] = name
        buf = {}
        buf_bytes = 0

    for shard in shards:
        with safe_open(str(base_dir / shard), framework="pt") as sf:
            for name in sf.keys():
                w = sf.get_tensor(name)
                if name in deltas:
                    a, b = deltas[name]
                    # delta = (B @ A) * scale，形状与原权重一致
                    w = (w.float() + (b.float() @ a.float()) * scale).to(w.dtype)
                    applied += 1
                buf[name] = w.contiguous()
                buf_bytes += w.numel() * w.element_size()
                if buf_bytes >= CHUNK_BYTES:
                    flush()
    flush()

    index = {"metadata": {"total_size": 0}, "weight_map": new_map}
    (out_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")

    # 非权重文件照原样搬过去
    for p in base_dir.iterdir():
        if p.is_file() and p.suffix not in (".safetensors", ".bin") and p.name != "model.safetensors.index.json":
            shutil.copy2(p, out_dir / p.name)

    return {"applied": applied, "total_lora": len(deltas), "shards_out": part, "scale": scale}


def main() -> int:
    if len(sys.argv) < 4:
        print("用法：merge_lora.py <基座目录> <适配器目录> <输出目录>")
        return 2
    base, adapter, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    for d in (base, adapter):
        if not d.exists():
            print(f"目录不存在：{d}")
            return 2
    if out.exists():
        shutil.rmtree(out, ignore_errors=True)
    info = merge(base, adapter, out)
    print(json.dumps(info, ensure_ascii=False))
    if info["applied"] != info["total_lora"]:
        print(f"⚠️ 有 {info['total_lora'] - info['applied']} 个 LoRA 模块没对上基座权重")
        return 1
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
