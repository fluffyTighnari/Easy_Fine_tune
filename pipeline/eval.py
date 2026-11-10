#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 11 步：评测。拿验证集的真实对话开头，让原模型和微调后的模型各回一次，对比。

输入：work/sft/val.jsonl、work/adapter/、models/<name>/
产出：work/reports/11-eval.md、work/eval/samples.md

用法（从 learn/ 目录）：
    ..\\.venv\\Scripts\\python.exe pipeline\\eval.py
    ..\\.venv\\Scripts\\python.exe pipeline\\eval.py --n 20 --which tuned
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

LEARN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LEARN))

from lib.paths import CONFIG, DEFAULT_ROOT, Layout  # noqa: E402
sys.path.insert(0, str(LEARN))

SENT_SPLIT = re.compile(r"[。！？!?~～\n]+")
PUNCT = set("，。！？、；：\u201c\u201d\u2018\u2019（）【】《》…—～~,.!?;:\"'()[]<>")
STOP_PHRASES = {"然后", "就是", "什么", "这个", "那个", "我们", "你们", "他们", "可以", "没有",
                "如果", "因为", "所以", "但是", "现在", "什么", "觉得", "知道", "真的", "有点"}


def setup_env() -> None:
    tmp = LEARN / "cache" / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    os.environ["TMP"] = str(tmp)
    os.environ["TEMP"] = str(tmp)
    os.environ.setdefault("HF_HOME", str(LEARN / "cache" / "hf"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(LEARN / "cache" / "hf_datasets"))
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def bigrams(t: str) -> set[str]:
    t = re.sub(r"\s+", "", t or "")
    return {t[i:i + 2] for i in range(len(t) - 1)}


def ground_pair(reply: str, ctx: str) -> float:
    return overlap(reply, ctx)


def overlap(a: str, b: str) -> float:
    """回复和上下文共享了多少字词（2-gram Jaccard）。接没接话看这个。"""
    ga, gb = bigrams(a), bigrams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def self_repeat(t: str) -> float:
    """回复内部 2-gram 重复率。自说自话、绕圈子的典型症状。"""
    g = [t[i:i + 2] for i in range(max(0, len(t) - 1))]
    if not g:
        return 0.0
    return 1 - len(set(g)) / len(g)


def sent_count(t: str) -> int:
    return len([s for s in SENT_SPLIT.split(t or "") if s.strip()])


def latin_ratio(t: str) -> float:
    letters = [c for c in (t or "") if c.isascii() and c.isalpha()]
    return len(letters) / max(1, len(t or ""))


def build_catchphrases(clean_path: Path, topn: int = 120, min_cnt: int = 8,
                       ratio: float = 2.5) -> list[str]:
    """口头禅 = 他说得明显比对方多的短语，不是「中文里常见」的短语。

    只看高频会在原模型身上刷到 100%：它一写 130 字，总有几个常用词撞上。
    所以按「他说 / 对方说」的频次比来筛，长 2~5 字。
    """
    mine: Counter[str] = Counter()
    theirs: Counter[str] = Counter()
    if not clean_path.exists():
        return []
    for line in open(clean_path, encoding="utf-8"):
        r = json.loads(line)
        t = re.sub(r"\s+", "", r.get("text") or "")
        if len(t) < 2:
            continue
        bag = mine if r.get("role") == "me" else theirs
        for n in (2, 3, 4, 5):
            for i in range(0, max(0, len(t) - n + 1)):
                g = t[i:i + n]
                if g.isascii() or g in STOP_PHRASES or g[0] in PUNCT or g[-1] in PUNCT:
                    continue
                bag[g] += 1
    keep = []
    for g, c in mine.most_common(topn * 6):
        if c < min_cnt:
            continue
        if c / max(1, theirs.get(g, 0)) < ratio:
            continue
        keep.append(g)
        if len(keep) >= topn:
            break
    return keep


def resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else LEARN / q


def load_samples(val_path: Path, n: int, seed: int = 42) -> list[dict]:
    rows = [json.loads(line) for line in open(val_path, encoding="utf-8")]
    rows = [r for r in rows if len(r["conversations"]) >= 3]
    random.Random(seed).shuffle(rows)
    # 长度分层：短/中/长各取三分之一，避免只看长回复
    rows.sort(key=lambda r: r["meta"]["target_chars"])
    k = max(1, n // 3)
    short = rows[: max(k, len(rows) // 3)][:k]
    mid = rows[len(rows) // 3: 2 * len(rows) // 3][:k]
    long_ = rows[2 * len(rows) // 3:][:k]
    picked = short + mid + long_
    return picked[:n]


def generate(model_dir: Path, adapter: Path | None, samples: list[dict], template: str,
             max_new: int, temp: float) -> tuple[list[str], float]:
    import time

    from llamafactory.chat import ChatModel

    args = {
        "model_name_or_path": str(model_dir),
        "template": template,
        "infer_backend": "huggingface",
        "quantization_bit": 4,
    }
    if adapter:
        args["adapter_name_or_path"] = str(adapter)
    t0 = time.time()
    cm = ChatModel(args)
    load_s = time.time() - t0

    outs = []
    for s in samples:
        conv = s["conversations"]
        system = next((c["value"] for c in conv if c["from"] == "system"), "")
        msgs = []
        for c in conv:
            if c["from"] == "human":
                msgs.append({"role": "user", "content": c["value"]})
            elif c["from"] == "gpt":
                msgs.append({"role": "assistant", "content": c["value"]})
        msgs = msgs[:-1]  # 去掉最后一条真实回复，留给模型说
        while msgs and msgs[0]["role"] != "user":
            msgs.pop(0)
        try:
            res = cm.chat(messages=msgs, system=system, temperature=temp, top_p=0.9,
                          max_new_tokens=max_new)
            outs.append((res[0].response_text or "").strip())
        except Exception as e:  # noqa: BLE001
            outs.append(f"<生成失败：{type(e).__name__}>")
    del cm
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    return outs, load_s


def load_config(path: Path) -> dict:
    try:
        import yaml

        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def other_person_names(clean_path: Path, config: dict) -> list[str]:
    """「对方」的称呼：优先用配置里写的，其次从第 5 步的 sender 字段取。"""
    names = {n for n in (config.get("persona", {}).get("other_names") or []) if n}
    path = clean_path.parent / "attributed.jsonl"
    if not path.exists():
        return sorted(names)
    self_names = set(config.get("persona", {}).get("self_names") or [])
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("role") == "them" and r.get("sender"):
                names.add(r["sender"])
    return sorted(names - self_names)


def stats(name: str, texts: list[str], real: list[str], phrases: list[str],
          ctxs: list[str] | None = None) -> dict:
    ctxs = ctxs or real
    def agg(xs: list[str]) -> dict:
        lens = sorted(len(x) for x in xs)
        sents = [sent_count(x) for x in xs]
        hit_n = sum(sum(x.count(p) for p in phrases) for x in xs)
        hit = hit_n * 100 / max(1, sum(len(x) for x in xs))  # 每 100 字命中次数
        return {
            "n": len(xs),
            "ground": sum(ground_pair(x, c) for x, c in zip(xs, ctxs)) / max(1, len(xs)),
            "repeat": sum(self_repeat(x) for x in xs) / max(1, len(xs)),
            "len_mean": sum(lens) / max(1, len(lens)),
            "len_p50": lens[len(lens) // 2] if lens else 0,
            "sent_mean": sum(sents) / max(1, len(sents)),
            "catch_hit": hit,
            "latin": sum(latin_ratio(x) for x in xs) / max(1, len(xs)),
            "empty": sum(1 for x in xs if len(x.strip()) < 2) / max(1, len(xs)),
        }

    return {"name": name, **agg(texts)}



def score_of(r: dict, real: dict, base: dict | None) -> tuple[float, dict]:
    """综合分：接话度权重最高（像不像可以让步，接不接得上话不行）。"""
    def rel(x, y):
        return 1 - min(1.5, abs(x - y) / max(1e-6, y))

    parts = {
        "长度": rel(r["len_mean"], real["len_mean"]),
        "句数": rel(r["sent_mean"], real["sent_mean"]),
        "口头禅": min(1.0, r["catch_hit"] / max(1e-6, real["catch_hit"])),
        "接话": min(1.0, r["ground"] / max(1e-6, (base or real)["ground"])),
    }
    w = {"长度": 0.20, "句数": 0.15, "口头禅": 0.20, "接话": 0.45}
    return round(sum(parts[k] * w[k] for k in parts), 4), {k: round(v, 3) for k, v in parts.items()}


def sweep(args, samples: list[dict], ctxs: list[str], real: list[str], phrases: list[str],
          model_dir: Path, adapter_dir: Path, outdir: Path) -> int:
    """扫所有 checkpoint，按综合分挑一个，不用靠调参试错。"""
    import shutil

    ckpts = sorted([d for d in adapter_dir.iterdir()
                    if d.is_dir() and d.name.startswith("checkpoint-")],
                   key=lambda d: int(d.name.split("-")[-1]))
    if not ckpts:
        print(f"{adapter_dir} 里没有 checkpoint，先跑第 10 步")
        return 2
    if args.sweep_n:
        args.n = args.sweep_n
    samples = load_samples(resolve(args.val), args.sweep_n or args.n)
    ctxs = ["\n".join([c["value"] for c in s_["conversations"][:-1] if c["from"] == "human"][-2:])
            for s_ in samples]
    real = [s_["conversations"][-1]["value"] for s_ in samples]
    real_s = stats("真实", real, real, phrases, ctxs)

    base_s = None
    if args.which == "both":
        print("先用原模型生成基准 …")
        base_outs, _ = generate(model_dir, None, samples, args.template, args.max_new, args.temp)
        base_s = stats("原模型", base_outs, real, phrases, ctxs)

    rows = []
    for ck in ckpts:
        print(f"评估 {ck.name} …")
        outs_ck, _ = generate(model_dir, ck, samples, args.template, args.max_new, args.temp)
        r = stats(ck.name, outs_ck, real, phrases, ctxs)
        sc, parts = score_of(r, real_s, base_s)
        rows.append((sc, parts, r))

    rows.sort(key=lambda x: -x[0])
    best_sc, best_parts, best_r = rows[0]
    best_dir = None
    for ck in ckpts:
        if ck.name == best_r["name"]:
            best_dir = ck
    target = adapter_dir.parent / "adapter_best"
    if best_dir and args.apply:
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        shutil.copytree(best_dir, target)
        print(f"已复制 {best_dir.name} → {target}")

    L = ["# 第 11 步：checkpoint 扫描", "",
         f"生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}　每档 {len(samples)} 条", "",
         f"选中：**{best_r['name']}**（综合分 {best_sc}）", "",
         "综合分 = 接话度 45% + 长度 20% + 口头禅 20% + 句数 15%", "",
         "| checkpoint | 综合分 | 接话 | 长度 | 句数 | 口头禅 | 均字数 | 句数实测 | 口头禅/百字 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for sc, parts, r in rows:
        L.append(f"| {r['name']} | **{sc}** | {parts['接话']} | {parts['长度']} | {parts['句数']} | "
                 f"{parts['口头禅']} | {r['len_mean']:.0f} | {r['sent_mean']:.1f} | {r['catch_hit']:.1f} |")
    if base_s:
        L += ["", f"原模型基准：接话 {base_s['ground']:.1%}，均 {base_s['len_mean']:.0f} 字，"
                  f"句 {base_s['sent_mean']:.1f}，口头禅 {base_s['catch_hit']:.1f}/百字", ""]
    L += ["", "真实值对照：" + f"均 {real_s['len_mean']:.0f} 字，句 {real_s['sent_mean']:.1f}，"
          f"口头禅 {real_s['catch_hit']:.1f}/百字，接话 {real_s['ground']:.1%}", ""]
    rep = resolve(args.reports)
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "11-sweep.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[3:12]))
    print(f"报告：{rep / '11-sweep.md'}")
    return 0



JUDGE_PROMPT = """对方刚说：
{ctx}

我回：
{reply}

这条回复有没有引入跟对话完全无关的新话题，或者说了和对方说的相反的情况？
有 → 回答「跑题」；没有 → 回答「贴题」。
对方说的话不需要被逐字回应，简短、情绪化的反应也算贴题。
只输出一个词：贴题 或 跑题。"""


def judge_grounding(samples: list[dict], texts: list[str], model_dir: Path,
                    template: str) -> tuple[list[bool], float]:
    """用基座模型当裁判，判断每条回复有没有接住上下文。

    为什么不用 n-gram：他本人的真实回复接话度只有 0.9%（「好嘟」「可恶」这种
    应答本来就不重复对方的词），词面重叠量不出「有没有在回应」。
    """
    import time

    from llamafactory.chat import ChatModel

    t0 = time.time()
    cm = ChatModel({
        "model_name_or_path": str(model_dir),
        "template": template,
        "infer_backend": "huggingface",
        "quantization_bit": 4,
    })
    load_s = time.time() - t0
    flags = []
    for smp, txt in zip(samples, texts):
        turns = [c["value"] for c in smp["conversations"][:-1] if c["from"] == "human"]
        prompt = JUDGE_PROMPT.format(ctx="\n".join(turns[-2:]), reply=txt)
        try:
            r = cm.chat(messages=[{"role": "user", "content": prompt}],
                        system="你是一个严格的中文对话质检员。", temperature=0.0,
                        max_new_tokens=8)
            ans = (r[0].response_text or "").strip()
            # 只说「贴题」才算贴题；没解析出来一律算跑题，宁可严
            flags.append(ans.startswith("贴题"))
        except Exception:
            flags.append(False)
    del cm
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    return flags, load_s


def main() -> int:
    ap = argparse.ArgumentParser(description="第 11 步：评测")
    ap.add_argument("--root", default=DEFAULT_ROOT, help="输出根目录，默认 work/")
    ap.add_argument("--val", default=None)
    ap.add_argument("--train", default=None)
    ap.add_argument("--clean", default=None)
    ap.add_argument("--reuse", action="store_true", help="复用上次生成结果，只重算指标")
    ap.add_argument("--sweep", action="store_true", help="扫所有 checkpoint，按综合分挑最好的")
    ap.add_argument("--sweep-n", type=int, default=8, help="扫描时每档用几条样本")
    ap.add_argument("--apply", action="store_true", help="把选中的 checkpoint 复制成 adapter_best/")
    ap.add_argument("--judge", action="store_true", default=True,
                    help="用基座模型当裁判判断有没有接住话（默认开）")
    ap.add_argument("--no-judge", dest="judge", action="store_false")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--model", default="models/Qwen3-4B-Instruct-2507")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--which", choices=["tuned", "base", "both"], default="both")
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--template", default="qwen3_nothink")
    ap.add_argument("--reports", default=None)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    _lay = Layout(args.root)
    _lay.ensure()
    args.val = args.val or str(_lay.val_jsonl)
    args.train = args.train or str(_lay.train_jsonl)
    args.clean = args.clean or str(_lay.clean_jsonl)
    args.adapter = args.adapter or str(_lay.adapter)
    args.reports = args.reports or str(_lay.reports)
    args.outdir = args.outdir or str(_lay.eval)

    setup_env()
    val_path, model_dir = resolve(args.val), resolve(args.model)
    adapter = resolve(args.adapter)
    if not val_path.exists():
        print(f"验证集不存在：{val_path}（先做第 7 步）")
        return 2
    if args.which in ("tuned", "both") and not (adapter / "adapter_config.json").exists():
        print(f"适配器不存在：{adapter}（先做第 10 步）")
        return 2

    outdir = resolve(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cache = outdir / "gen.json"
    phrases = build_catchphrases(resolve(args.clean))

    if args.sweep:
        return sweep(args, [], [], [], phrases, model_dir, adapter, outdir)

    samples = load_samples(val_path, args.n)
    real = [s["conversations"][-1]["value"] for s in samples]

    def ctx_of(smp: dict) -> str:
        turns = [c["value"] for c in smp["conversations"][:-1] if c["from"] == "human"]
        return "\n".join(turns[-2:])

    ctxs = [ctx_of(x) for x in samples]
    outs: dict[str, list[str]] = {}
    load_time = 0.0

    # 缓存要认适配器，换了模型不能复用旧生成
    ad_mtime = int((adapter / "adapter_model.safetensors").stat().st_mtime) \
        if (adapter / "adapter_model.safetensors").exists() else 0
    reused = False
    if args.reuse and cache.exists():
        saved = json.loads(cache.read_text(encoding="utf-8"))
        if (saved.get("n") == len(samples) and saved.get("temp") == args.temp
                and saved.get("adapter_mtime") == ad_mtime):
            outs = saved["outs"]
            reused = True
            print(f"复用上次生成结果（{len(samples)} 条）")

    if not reused:
        if args.which in ("tuned", "both"):
            print(f"用微调后的模型生成 {len(samples)} 条 …")
            outs["tuned"], t = generate(model_dir, adapter, samples, args.template, args.max_new, args.temp)
            load_time = t
        if args.which in ("base", "both"):
            print(f"用原模型生成 {len(samples)} 条 …")
            outs["base"], t = generate(model_dir, None, samples, args.template, args.max_new, args.temp)
            load_time = max(load_time, t)
        cache.write_text(json.dumps({"n": len(samples), "temp": args.temp,
                                     "adapter_mtime": ad_mtime, "outs": outs},
                                    ensure_ascii=False), encoding="utf-8")

    rows = [stats("真实（他本人）", real, real, phrases, ctxs)]
    for k in ("base", "tuned"):
        if k in outs:
            rows.append(stats("原模型" if k == "base" else "微调后", outs[k], real, phrases, ctxs))

    # AI 腔：复用第 6 步的结构化打分
    ai_flags = {}
    try:
        from pipeline.clean import ai_score

        for k, xs in outs.items():
            ai_flags[k] = sum(1 for x in xs if ai_score(x)[0] >= 4 and len(x) > 200)
    except Exception:
        pass

    # 名字泄漏：对方的名字如果凭空出现在回复里，说明模型把上下文当成了填空
    others = other_person_names(resolve(args.clean), load_config(args.config))
    name_leak = {}
    for k, xs in outs.items():
        bad = 0
        for s_, x in zip(samples, xs):
            ctx = "".join(c["value"] for c in s_["conversations"][:-1])
            if any(n in x and n not in ctx for n in others):
                bad += 1
        name_leak[k] = bad

    # ---- 接话判定（模型当裁判）----
    judge_res = {}
    if args.judge and "tuned" in outs:
        print("用基座模型判定接话情况 …")
        for k in ("base", "tuned"):
            if k in outs:
                flags, _ = judge_grounding(samples, outs[k], model_dir, args.template)
                judge_res[k] = flags
        real_flags, _ = judge_grounding(samples, real, model_dir, args.template)
        judge_res["real"] = real_flags

    # ---- 闸门 ----
    tuned = next((r for r in rows if r["name"] == "微调后"), None)
    base = next((r for r in rows if r["name"] == "原模型"), None)
    real_s = rows[0]
    gates = []
    if tuned:
        gates += [
            ("生成非空", tuned["empty"] <= 0.2, f"空回复 {tuned['empty']:.0%}"),
            ("长度像真人", 0.4 * real_s["len_mean"] <= tuned["len_mean"] <= 2.5 * real_s["len_mean"],
             f"微调 {tuned['len_mean']:.0f} 字 / 真实 {real_s['len_mean']:.0f} 字"),
            ("没有 AI 腔", ai_flags.get("tuned", 0) == 0, f"{ai_flags.get('tuned', 0)} 条"),
            ("没有串名字", name_leak.get("tuned", 0) == 0, f"{name_leak.get('tuned', 0)} 条"),
        ]
        if base:
            gates.append(("口头禅密度不低于原模型", tuned["catch_hit"] >= base["catch_hit"] * 0.9,
                          f"微调 {tuned['catch_hit']:.1f} / 原模型 {base['catch_hit']:.1f} 次每百字"))
            # 硬要求：像不像可以退让，接不接得上话不能退
            if judge_res:
                tb = sum(judge_res["tuned"]) / max(1, len(judge_res["tuned"]))
                bb = sum(judge_res["base"]) / max(1, len(judge_res["base"]))
                rr = sum(judge_res["real"]) / max(1, len(judge_res["real"]))
                # 两个下限：像真人（不比本人更跑题）、别退化（比助手低太多就是练废了）
                floor_human = rr - 0.05
                floor_base = bb - 0.20
                gates.append(("贴题率：不像真人跑题，也不比原模型差太多",
                              tb >= floor_human and tb >= floor_base,
                              f"微调 {tb:.0%}（下限 真人 {floor_human:.0%} / 原模型 {floor_base:.0%}）"
                              f"，原模型 {bb:.0%}，真人 {rr:.0%}"))
            gates.append(("自我重复不超过 1/3", tuned["repeat"] <= 0.33,
                          f"{tuned['repeat']:.0%}"))

    # ---- 样本对照 ----
    outdir = resolve(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    S = ["# 第 11 步：生成对照", "",
         f"生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}　样本 {len(samples)} 条　temperature {args.temp}", ""]
    for i, s in enumerate(samples, 1):
        conv = s["conversations"]
        ctx = [c for c in conv[:-1] if c["from"] != "system"]
        S += [f"## {i}. 上下文 {s['meta']['n_ctx_turns']} 轮　真实 {len(real[i - 1])} 字", "", "```"]
        for c in ctx[-4:]:
            who = "我" if c["from"] == "gpt" else "对方"
            S.append(f"{who}：{c['value']}")
        S += ["```", "", f"**真实回复**（{len(real[i - 1])} 字）", "", f"> {real[i - 1]}", ""]
        for k in ("base", "tuned"):
            if k in outs:
                txt = outs[k][i - 1]
                S += [f"**{'原模型' if k == 'base' else '微调后'}**（{len(txt)} 字）", "", f"> {txt}", ""]
    (outdir / "samples.md").write_text("\n".join(S), encoding="utf-8")

    # ---- 报告 ----
    L = ["# 第 11 步：评测", "",
         f"生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}", "",
         "## 结论", "",
         f"- 样本：{len(samples)} 条（验证集短/中/长各三分之一）",
         f"- 生成：temperature {args.temp}，max_new_tokens {args.max_new}",
         f"- 口头禅库：{len(phrases)} 条高频短语（取自训练集）",
         f"- 闸门：**{'✅ 通过' if all(g[1] for g in gates) else '❌ 未通过'}**" if gates else "",
         "", "| 闸门 | 结果 | 实测 |", "|---|---|---|"]
    for name, ok, detail in gates:
        L.append(f"| {name} | {'✅' if ok else '❌'} | {detail} |")

    L += ["", "## 指标", "", "| 指标 | " + " | ".join(r["name"] for r in rows) + " |",
          "|---" * (len(rows) + 1) + "|",
          "| 平均字数 | " + " | ".join(f"{r['len_mean']:.0f}" for r in rows) + " |",
          "| 字数中位 | " + " | ".join(f"{r['len_p50']}" for r in rows) + " |",
          "| 平均句数 | " + " | ".join(f"{r['sent_mean']:.1f}" for r in rows) + " |",
          "| 口头禅密度 | " + " | ".join(f"{r['catch_hit']:.1f}/百字" for r in rows) + " |",
          "| 字母占比 | " + " | ".join(f"{r['latin']:.2%}" for r in rows) + " |",
          "| 空回复率 | " + " | ".join(f"{r['empty']:.0%}" for r in rows) + " |",
          "| 词面重合 | " + " | ".join(f"{r['ground']:.1%}" for r in rows) + " |",
          "| 自我重复 | " + " | ".join(f"{r['repeat']:.0%}" for r in rows) + " |"]

    if judge_res:
        L += ["", "## 贴题判定（基座模型当裁判）", "",
              "| 来源 | 贴题率 |", "|---|---|"]
        for k, label in (("real", "真实（他本人）"), ("base", "原模型"), ("tuned", "微调后")):
            if k in judge_res:
                L.append(f"| {label} | {sum(judge_res[k]) / max(1, len(judge_res[k])):.0%} |")
        L += ["", "判定问题：这条回复有没有引入无关新话题、或说出相反的情况。简短和情绪化不算跑题。", ""]

    if ai_flags or name_leak:
        L += ["", "## 污染检查", "", "| 检查 | " + " | ".join(k for k in outs) + " |",
              "|---" * (len(outs) + 1) + "|",
              "| AI 腔 | " + " | ".join(str(ai_flags.get(k, "-")) for k in outs) + " |",
              "| 冒出对方名字 | " + " | ".join(str(name_leak.get(k, "-")) for k in outs) + " |"]

    L += ["", "## 口头禅库（前 30）", "", "`" + "` `".join(phrases[:30]) + "`", "",
          "## 对照样本", "", f"逐条对比看 `{(outdir / 'samples.md').as_posix()}`", "",
          "## 怎么判断", "",
          "指标只是筛异常，最终看对照样本：读起来像不像目标人物本人在微信上随手打字。", ""]

    rep = resolve(args.reports)
    rep.mkdir(parents=True, exist_ok=True)
    md = rep / "11-eval.md"
    md.write_text("\n".join(L), encoding="utf-8")

    print("\n指标：")
    for r in rows:
        print(f"  {r['name']:<12} 均 {r['len_mean']:>3.0f} 字　句 {r['sent_mean']:.1f}　"
              f"口头禅 {r['catch_hit']:.1f}/百字　接话 {r['ground']:.1%}　空 {r['empty']:.0%}")
    if gates:
        print(f"闸门：{'✅ 通过' if all(g[1] for g in gates) else '❌ 未通过'}")
        for name, ok, detail in gates:
            if not ok:
                print(f"  ❌ {name}：{detail}")
    print(f"报告：{md}")
    print(f"对照：{outdir / 'samples.md'}")
    return 0 if (not gates or all(g[1] for g in gates)) else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
