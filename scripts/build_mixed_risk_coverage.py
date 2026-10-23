#!/usr/bin/env python3
"""
build_mixed_risk_coverage.py
----------------------------
Adapter for the v2 mixed-distribution risk-coverage analysis (#2).

Reuses the AUTHORITATIVE labelling from scripts/analyze_governance_eval.py
(normalize_vq / normalize_rpal / score_traps -> unsafe_numeric ; clean_labels_*)
so every number matches Panel A / Panel B exactly. No model calls.

Distribution = clean 250Q (answerable) + 80 F/E/I/N traps (no valid numeric answer).
On a clean item : correct / silent_wrong / abstain.
On a trap item  : any un-intercepted number = false acceptance (unsafe_numeric).

Metrics on a mixed set of N = |clean| + |traps| items, per system:
  answered   = clean_correct + clean_silent_wrong + trap_unsafe   (silent numeric outputs)
  coverage   = answered / N
  sel_acc    = clean_correct / answered            (trap answers are never correct)
  sel_risk   = 1 - sel_acc
  swr        = (clean_silent_wrong + trap_unsafe) / N

Because the clean:trap ratio is a modelling choice, we emit several variants and
also a ratio-free "safety vs utility" view (clean coverage/sel-acc vs trap FNAR).

``--vq july`` (default) uses the July VQ trap run and writes mixed_risk_coverage.json
(the submitted Figure 3). ``--vq top4x3`` uses the mean of the three 2026-10-14
top-4 reruns and writes mixed_risk_coverage_top4.json (camera-ready Figure 3).
The safety-vs-utility R-PAL+ point reads the 250Q prompted runs, as the paper states;
the JSON had been patched to those values by hand while this script still read the
50Q pilot runs.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from analyze_governance_eval import (  # noqa: E402
    load_jsonl, normalize_vq, normalize_rpal, score_traps,
    clean_labels_from_vq, clean_labels_from_rpal,
)

BASE = ROOT / "results"
G = BASE / "governance_confirmatory"
_ap = argparse.ArgumentParser()
_ap.add_argument("out_dir", nargs="?", type=Path, default=ROOT / "results/mixed_risk_coverage")
_ap.add_argument("--vq", choices=("july", "top4x3"), default="july")
ARGS = _ap.parse_args()
OUT_DIR = ARGS.out_dir
OUT_DIR.mkdir(parents=True, exist_ok=True)
G1014 = BASE / "governance_confirmatory_1014"
VQ_TRAP_FILES = ([G / "vq_k1_traps.jsonl"] if ARGS.vq == "july"
                 else [G1014 / f"vq_k1_top4_run{i}.jsonl" for i in (1, 2, 3)])
VQ_FNAR_EXPECTED = 0.1125 if ARGS.vq == "july" else 25 / 240  # 10 + 8 + 7 unsafe over 3 x 80
OUT_NAME = "mixed_risk_coverage.json" if ARGS.vq == "july" else "mixed_risk_coverage_top4.json"


def approx(a: float, b: float, tol: float = 5e-4) -> bool:
    return abs(a - b) <= tol


# ---------------------------------------------------------------- trap unsafe flags
traps = load_jsonl(G / "trap_set.jsonl")
trap_ids = [str(t["trap_id"]) for t in traps]


def unsafe_flags(rows, normalizer):
    scored = score_traps(traps, rows, normalizer)
    return {d["case_id"]: int(d["unsafe_numeric"]) for d in scored["details"]}


_vq_runs = [unsafe_flags(load_jsonl(p), normalize_vq) for p in VQ_TRAP_FILES]
vq_unsafe = {tid: statistics.mean(r[tid] for r in _vq_runs) for tid in trap_ids}
plain_unsafe = unsafe_flags(load_jsonl(G / "conf_plain_run1.jsonl"), normalize_rpal)
prompted_runs_unsafe = [
    unsafe_flags(load_jsonl(G / f"conf_prompted_run{i}.jsonl"), normalize_rpal) for i in (1, 2, 3)
]
prompted_unsafe_mean = {tid: statistics.mean(r[tid] for r in prompted_runs_unsafe) for tid in trap_ids}


def fnar(flags):
    return sum(flags[t] for t in trap_ids) / len(trap_ids)


# validate against confirmatory_summary.json
assert approx(fnar(vq_unsafe), VQ_FNAR_EXPECTED), fnar(vq_unsafe)
assert approx(fnar(plain_unsafe), 0.4875), fnar(plain_unsafe)
assert approx(fnar(prompted_unsafe_mean), 0.275), fnar(prompted_unsafe_mean)

# ---------------------------------------------------------------- clean labels
clean_rows = load_jsonl(BASE / "paper_v2_250/questions_250.jsonl")
questions = {str(r["question_id"]): r for r in clean_rows}
clean_ids = set(questions)

vq_clean = clean_labels_from_vq(load_jsonl(BASE / "paper_v2_250/results/vq_flash/output.jsonl"), clean_ids)
plain_clean = clean_labels_from_rpal(
    load_jsonl(BASE / "retrieval_pal_baseline/rpal_top4_gemini_25_flash_audited.jsonl"),
    questions, use_stored_verdict=True,
)
prompted_clean_runs = [
    clean_labels_from_rpal(load_jsonl(BASE / f"governance_eval/gov_run{i}_prompted_clean.jsonl"),
                           questions, use_stored_verdict=False)
    for i in (1, 2, 3)
]
prompted_clean250_runs = [
    clean_labels_from_rpal(load_jsonl(BASE / f"governance_eval/prompted_clean250_run{i}.jsonl"),
                           questions, use_stored_verdict=False)
    for i in (1, 2, 3)
]


def clean_counts(labels):
    from collections import Counter
    c = Counter(labels.values())
    return {"n": len(labels), "correct": c["correct"], "silent_wrong": c["silent_wrong"],
            "abstain": c["abstain"]}


# validate clean against numbers.tex
_vq = clean_counts(vq_clean)
_pl = clean_counts(plain_clean)
assert _vq["correct"] == 203 and _vq["abstain"] == 40, _vq          # VQ K=1: 203/7/40, cov 0.84
assert _pl["correct"] == 242 and _pl["silent_wrong"] == 6, _pl      # R-PAL: 242/6/2, cov 0.992

# ---------------------------------------------------------------- mixed metric
def mixed(clean_labels, trap_flags):
    cc = clean_counts(clean_labels)
    n_clean = cc["n"]
    n_trap = len(trap_ids)
    trap_wrong = sum(trap_flags[t] for t in trap_ids)
    correct = cc["correct"]
    wrong = cc["silent_wrong"] + trap_wrong
    answered = correct + wrong
    N = n_clean + n_trap
    return {
        "n_clean": n_clean, "n_trap": n_trap, "N": N,
        "coverage": answered / N,
        "selective_accuracy": correct / answered if answered else 0.0,
        "selective_risk": (wrong / answered) if answered else 0.0,
        "swr": wrong / N,
        "clean": cc, "trap_false_accept": trap_wrong,
    }


def safety_utility(clean_labels, trap_flags):
    cc = clean_counts(clean_labels)
    ans = cc["correct"] + cc["silent_wrong"]
    return {
        "clean_coverage": ans / cc["n"] if cc["n"] else 0.0,
        "clean_selective_accuracy": cc["correct"] / ans if ans else 0.0,
        "clean_accuracy": cc["correct"] / cc["n"] if cc["n"] else 0.0,
        "trap_fnar": sum(trap_flags[t] for t in trap_ids) / len(trap_ids),
        "n_clean": cc["n"],
    }


def prompted_mixed_mean():
    ms = [mixed(prompted_clean_runs[i], prompted_runs_unsafe[i]) for i in range(3)]
    keys = ("coverage", "selective_accuracy", "selective_risk", "swr")
    return {k: statistics.mean(m[k] for m in ms) for k in keys} | {
        "n_clean": ms[0]["n_clean"], "n_trap": 80, "N": ms[0]["N"], "note": "3-run mean"}


# ---- variant 1: each system on its own full clean + 80 traps (denominators differ) ----
full = {
    "vq_k1":   mixed(vq_clean, vq_unsafe),
    "rpal_plain": mixed(plain_clean, plain_unsafe),
    "rpal_prompted": prompted_mixed_mean(),
}

# ---- variant 2: apples-to-apples on the COMMON clean subset (prompted's 50) + 80 traps ----
common_ids = set(prompted_clean_runs[0]) & set(vq_clean) & set(plain_clean)
def sub(labels):
    return {k: v for k, v in labels.items() if k in common_ids}
common = {
    "n_common_clean": len(common_ids),
    "vq_k1":   mixed(sub(vq_clean), vq_unsafe),
    "rpal_plain": mixed(sub(plain_clean), plain_unsafe),
    "rpal_prompted": {k: statistics.mean(mixed(sub(prompted_clean_runs[i]), prompted_runs_unsafe[i])[k]
                                         for i in range(3))
                      for k in ("coverage", "selective_accuracy", "selective_risk", "swr")}
                     | {"n_clean": len(common_ids), "n_trap": 80, "N": len(common_ids) + 80,
                        "note": "3-run mean"},
}

# ---- variant 3: ratio-free safety vs utility (full native clean) ----
su = {
    "vq_k1": safety_utility(vq_clean, vq_unsafe),
    "rpal_plain": safety_utility(plain_clean, plain_unsafe),
    "rpal_prompted": {k: statistics.mean(safety_utility(prompted_clean250_runs[i], prompted_runs_unsafe[i])[k]
                                         for i in range(3))
                      for k in ("clean_coverage", "clean_selective_accuracy", "clean_accuracy", "trap_fnar")}
                     | {"n_clean": 250,
                        "note": "clean = 3-run mean on the same 250Q set as plain R-PAL and VQ; traps = 3-run mean"},
}

report = {
    "vq_trap_source": [str(p.relative_to(ROOT)) for p in VQ_TRAP_FILES],
    "distribution": "clean 250Q + 80 F/E/I/N traps (no valid numeric answer)",
    "trap_fnar_validated": {"vq_k1": fnar(vq_unsafe), "rpal_plain": fnar(plain_unsafe),
                            "rpal_prompted_mean": fnar(prompted_unsafe_mean)},
    "clean_counts": {"vq_k1": _vq, "rpal_plain": _pl,
                     "rpal_prompted_run1": clean_counts(prompted_clean_runs[0])},
    "variant1_full_clean_plus_traps": full,
    "variant2_common_subset": common,
    "variant3_safety_vs_utility": su,
}
(OUT_DIR / OUT_NAME).write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

# ---------------------------------------------------------------- console
def line(name, m):
    print(f"  {name:16s} cov={m['coverage']:.3f}  sel_acc={m['selective_accuracy']:.3f}  "
          f"risk={m['selective_risk']:.3f}  swr={m['swr']:.3f}  (N={m['N']})")

print("== validated FNAR ==", {k: round(v, 4) for k, v in report['trap_fnar_validated'].items()})
print("\n[V1] full clean (250 / 250 / 50) + 80 traps  -- denominators differ:")
for k in ("vq_k1", "rpal_plain", "rpal_prompted"): line(k, full[k])
print(f"\n[V2] apples-to-apples: common {common['n_common_clean']} clean + 80 traps:")
for k in ("vq_k1", "rpal_plain", "rpal_prompted"): line(k, common[k])
print("\n[V3] ratio-free safety vs utility:")
for k in ("vq_k1", "rpal_plain", "rpal_prompted"):
    s = su[k]
    print(f"  {k:16s} clean_cov={s['clean_coverage']:.3f}  clean_selacc={s['clean_selective_accuracy']:.3f}  "
          f"trap_FNAR={s['trap_fnar']:.3f}")
print(f"\nwrote {OUT_DIR / OUT_NAME}")
