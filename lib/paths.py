#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一处定义所有路径，别在每个脚本里重复拼。

新用户只需要一个根目录概念：所有输入输出都在 `--root` 下面。
默认 `work/`；本次实验的根目录叫 `test/`。

    root/
      input/chat.txt      原始聊天记录（唯一要准备的输入）
      data/              中间产物 *.jsonl
      sft/               训练样本 train/val.jsonl
      generated/         llamafactory.yaml 等生成物
      reports/           每一步的报告
      adapter/           训练输出
      adapter_best/      扫描选出来的最优存档
      merged/            合并后的模型
      gguf/              导出产物
      deploy/            Modelfile
      logs/              子进程日志
"""
from __future__ import annotations

from pathlib import Path

DEFAULT_ROOT = "work"
CONFIG = "config/persona.yaml"


class Layout:
    """把根目录展开成各阶段用的子目录。"""

    def __init__(self, root: str | Path = DEFAULT_ROOT) -> None:
        self.root = Path(root)

    def __truediv__(self, name: str) -> Path:
        return self.root / name

    @property
    def input(self) -> Path:
        return self.root / "input"

    @property
    def chat(self) -> Path:
        return self.input / "chat.txt"

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def clean_jsonl(self) -> Path:
        return self.data / "clean.jsonl"

    @property
    def sft(self) -> Path:
        return self.root / "sft"

    @property
    def train_jsonl(self) -> Path:
        return self.sft / "train.jsonl"

    @property
    def val_jsonl(self) -> Path:
        return self.sft / "val.jsonl"

    @property
    def generated(self) -> Path:
        return self.root / "generated"

    @property
    def yaml(self) -> Path:
        return self.generated / "llamafactory.yaml"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def adapter(self) -> Path:
        return self.root / "adapter"

    @property
    def adapter_best(self) -> Path:
        return self.root / "adapter_best"

    @property
    def merged(self) -> Path:
        return self.root / "merged"

    @property
    def gguf(self) -> Path:
        return self.root / "gguf"

    @property
    def deploy(self) -> Path:
        return self.root / "deploy"

    @property
    def eval(self) -> Path:
        return self.root / "eval"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    def ensure(self) -> "Layout":
        for d in (self.input, self.data, self.sft, self.generated,
                  self.reports, self.deploy, self.logs):
            d.mkdir(parents=True, exist_ok=True)
        return self

    def __repr__(self) -> str:
        return f"Layout(root={self.root})"
