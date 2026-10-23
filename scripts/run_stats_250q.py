#!/usr/bin/env python3
"""Wave 0-B/0-C: paired statistics + cost accounting on the existing 250Q runs.

Zero LLM calls. Uses per-case outputs already on disk:
  VQ      results/paper_v2_250/results/vq_flash/output.jsonl
  CoT     .../cot_single_shot_flash/output.jsonl
  CoT+O   .../cot_basic_oracle_flash/output.jsonl

Reports:
  1. Exact McNemar tests (binomial on discordant pairs) for accuracy and for
     silent-wrong emission, on the paired 250 cases:
       VQ-K3 vs CoT+oracle, VQ-K3 vs CoT-single, VQ-K1 vs CoT+oracle.
  2. Percentile-bootstrap 95% CIs (10k resamples) for accuracy and SWR of each
     system (VQ K=1 / K=3, CoT single, CoT+oracle).
  3. Cost accounting: wall-clock and per-case LLM-call estimates from summaries
     and per-case round counts.
"""
from __future__ import annotations
import json, os, random
from scipy.stats import binomtest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "results/paper_v2_250")


def load(path, key):
    out = {}
    for l in open(os.path.join(BASE, "results", path, "output.jsonl")):
        if not l.strip():
            continue
        d = json.loads(l)
        out[d["case_id"]] = d[key]
    return out


def committed(diag):
    return isinstance(diag, dict) and ("is_correct" in diag)


def vq_states(fg, K):
    """(correct, silent_wrong, abstain) booleans at budget K (history truncation; K=3 uses final)."""
    if K >= 3:
        corr = bool(fg.get("final_is_correct"))
        com = fg.get("final_output_value") is not None or fg.get("final_status") == "success"
        return corr, (com and not corr), (not com)
    first = None
    for h in fg.get("history", []):
        if h["turn"] <= K and committed(h.get("diagnostic", {})):
            first = h["diagnostic"]; break
    if first is None:
        return False, False, True
    c = bool(first.get("is_correct"))
    return c, (not c), False


def cot_states(cs):
    corr = bool(cs.get("final_is_correct"))
    com = cs.get("final_answer") is not None
    return corr, (com and not corr), (not com)


def mcnemar(pairs, label):
    b = sum(1 for x, y in pairs if x and not y)
    c = sum(1 for x, y in pairs if y and not x)
    n = b + c
    p = binomtest(min(b, c), n, 0.5).pvalue if n else 1.0
    print(f"  {label:42} b={b:3} c={c:3}  p={p:.4g}")
    return {"label": label, "b": b, "c": c, "p": p}


def boot_ci(flags, n_boot=10000, seed=0):
    rng = random.Random(seed); n = len(flags)
    stats = sorted(sum(rng.choices(flags, k=n)) / n for _ in range(n_boot))
    return stats[int(0.025 * n_boot)], stats[int(0.975 * n_boot)]


def main():
    vq = load("vq_flash", "framework_guided")
    cot = load("cot_single_shot_flash", "cot_self_improve")
    coto = load("cot_basic_oracle_flash", "cot_self_improve")
    cases = sorted(set(vq) & set(cot) & set(coto))
    print(f"paired cases: {len(cases)}")

    S = {}
    S["vq_k1"] = {c: vq_states(vq[c], 1) for c in cases}
    S["vq_k3"] = {c: vq_states(vq[c], 3) for c in cases}
    S["cot"] = {c: cot_states(cot[c]) for c in cases}
    S["coto"] = {c: cot_states(coto[c]) for c in cases}

    print("\n== Exact McNemar (accuracy: correct vs not) ==")
    res = []
    for a, b_, lab in [("vq_k3", "coto", "VQ K=3 vs CoT+oracle"),
                       ("vq_k3", "cot", "VQ K=3 vs CoT single-shot"),
                       ("vq_k1", "coto", "VQ K=1 (autonomous) vs CoT+oracle"),
                       ("vq_k1", "cot", "VQ K=1 (autonomous) vs CoT single-shot")]:
        res.append(mcnemar([(S[a][c][0], S[b_][c][0]) for c in cases], lab))

    print("\n== Exact McNemar (silent-wrong emission) ==")
    for a, b_, lab in [("vq_k3", "coto", "VQ K=3 vs CoT+oracle (SW)"),
                       ("vq_k1", "cot", "VQ K=1 vs CoT single-shot (SW)")]:
        res.append(mcnemar([(S[a][c][1], S[b_][c][1]) for c in cases], lab))

    print("\n== Bootstrap 95% CIs (10k resamples) ==")
    ci = {}
    for name in ("vq_k1", "vq_k3", "cot", "coto"):
        acc = [1 if S[name][c][0] else 0 for c in cases]
        sw = [1 if S[name][c][1] else 0 for c in cases]
        alo, ahi = boot_ci(acc); slo, shi = boot_ci(sw)
        ci[name] = {"acc": (sum(acc) / len(acc), alo, ahi), "swr": (sum(sw) / len(sw), slo, shi)}
        print(f"  {name:7} acc={100*ci[name]['acc'][0]:5.1f}% [{100*alo:.1f}, {100*ahi:.1f}]"
              f"   SWR={100*ci[name]['swr'][0]:4.1f}% [{100*slo:.1f}, {100*shi:.1f}]")

    print("\n== Cost accounting ==")
    summ = json.load(open(os.path.join(BASE, "paper_results_summary.json")))
    rounds = [vq[c].get("rounds", len(vq[c].get("history", [])) or 1) for c in cases]
    avg_r = sum(rounds) / len(rounds)
    cost = {"vq_avg_rounds": avg_r,
            "vq_est_llm_calls_per_case_k1": 3,        # selector + extractor + critic
            "vq_est_llm_calls_per_case_k3": round(3 * avg_r + (avg_r - 1), 2)}
    for k in ("cot_single_shot_flash", "cot_basic_oracle_flash", "vq_flash"):
        el = summ.get(k, {}).get("elapsed_seconds")
        if el:
            cost[f"{k}_elapsed_s"] = el
            cost[f"{k}_s_per_case"] = round(el / 250, 1)
            print(f"  {k:26} wall={el:8.0f}s  per-case={el/250:5.1f}s")
    print(f"  VQ avg rounds={avg_r:.2f}; est calls/case: K=1≈3, K=3≈{cost['vq_est_llm_calls_per_case_k3']}"
          f" (selector+extractor+critic per round + oracle per extra round)")

    out = {"paired_cases": len(cases), "mcnemar": res,
           "bootstrap_ci": {k: {"acc": v["acc"], "swr": v["swr"]} for k, v in ci.items()},
           "cost": cost}
    p = os.path.join(BASE, "results", "stats_250q.json")
    json.dump(out, open(p, "w"), indent=2)
    print("\nwritten:", p)


if __name__ == "__main__":
    main()
