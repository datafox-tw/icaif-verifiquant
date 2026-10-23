#!/usr/bin/env python3
"""Regenerate the 250Q risk-coverage figure, now with the SC@5 REALIZED frontier.

Curves:
  - CoT single-shot / CoT+oracle: perfect-rejector UPPER BOUNDS (no real threshold reaches these)
  - Self-consistency@5: REAL achievable points (vote-share thresholds) -- the realizable proxy
  - VerifiQuant: REAL operating points K=1->3
Message: VQ's real point beats even the perfect-rejector bound of single-shot, and beats the
best realizable proxy (SC@5) outright.
"""
from __future__ import annotations
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "results/paper_v2_250")


def load_correct(path, key):
    return [bool(json.loads(l)[key].get("final_is_correct"))
            for l in open(os.path.join(BASE, "results", path, "output.jsonl")) if l.strip()]


def ub_curve(correct):
    n = len(correct); nc = sum(correct)
    ks = np.arange(1, n + 1)
    return ks / n * 100, np.minimum(nc, ks) / ks * 100


def vq_points():
    vq = [json.loads(l)["framework_guided"]
          for l in open(os.path.join(BASE, "results/vq_flash/output.jsonl")) if l.strip()]
    n = len(vq)
    def committed(d): return isinstance(d, dict) and ("is_correct" in d)
    def pt(K):
        ans = corr = 0
        for fg in vq:
            first = None
            for h in fg.get("history", []):
                if h["turn"] <= K and committed(h.get("diagnostic", {})):
                    first = h["diagnostic"]; break
            if first is not None:
                ans += 1; corr += 1 if first.get("is_correct") else 0
        return ans / n * 100, (corr / ans * 100 if ans else 0)
    return {K: pt(K) for K in (1, 2, 3)}


def main():
    cot1 = load_correct("cot_single_shot_flash", "cot_self_improve")
    cotO = load_correct("cot_basic_oracle_flash", "cot_self_improve")
    vq = vq_points()
    sc = json.load(open(os.path.join(ROOT,
        "results/sc_baseline/sc5_gemini_25_flash_summary.json")))["risk_coverage_points"]
    sc_xy = [(p["coverage"] * 100, p["selective_acc"] * 100)
             for p in sc if p["selective_acc"] is not None]

    plt.rcParams.update({
        "font.size": 17, "axes.titlesize": 18, "axes.labelsize": 17,
        "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 14.5,
    })
    plt.figure(figsize=(7.4, 5.2))
    c, a = ub_curve(cot1); plt.plot(c, a, "--", color="#d62728",
        label="CoT single-shot — perfect-rejector bound")
    c, a = ub_curve(cotO); plt.plot(c, a, "--", color="#ff9896",
        label="CoT+oracle — perfect-rejector bound")
    xs, ys = zip(*sorted(sc_xy))
    plt.plot(xs, ys, "-s", color="#7f7f7f", markersize=7,
        label="Self-consistency@5 — REAL frontier (vote-share thr.)")
    vx = [vq[K][0] for K in (1, 2, 3)]; vy = [vq[K][1] for K in (1, 2, 3)]
    plt.plot(vx, vy, "-o", color="#1f77b4", markersize=9,
        label="VerifiQuant — REAL policy (K=1→3)")
    for K in (1, 2, 3):
        offsets = {1: (8, -14), 2: (-44, -18), 3: (8, -18)}
        plt.annotate(f"K={K}", (vq[K][0], vq[K][1]),
                     textcoords="offset points", xytext=offsets[K], color="#1f77b4", fontsize=16)
    plt.axvline(vq[1][0], color="gray", ls=":", lw=0.8)
    plt.xlabel("Coverage (% answered)")
    plt.ylabel("Selective accuracy (% of answered)")
    plt.title("Risk–coverage (250Q)")
    plt.ylim(70, 100); plt.xlim(75, 101); plt.grid(alpha=0.3)
    plt.legend(fontsize=14.5, loc="lower left")
    plt.tight_layout()
    out = os.path.join(ROOT, "docs/figures/risk_coverage_250q.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.08)
    print("figure:", out)
    print(f"VQ K=1 real: {vq[1][1]:.1f}% @ {vq[1][0]:.1f}% cov")
    print(f"SC@5 best:   {max(y for _,y in sc_xy):.1f}% sel-acc; at ~84% cov = "
          f"{[y for x,y in sc_xy if abs(x-84)<2]}")


if __name__ == "__main__":
    main()
