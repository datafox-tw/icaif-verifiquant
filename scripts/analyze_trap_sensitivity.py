#!/usr/bin/env python3
"""
analyze_trap_sensitivity.py
---------------------------
F-trap construction check and FNAR sensitivity (2026-10-17, advisor's follow-up A).

The F operator is defined as removing a *required* input. A trap violates that definition when the
removed field is optional in the contract or has a default the contract (or its reference function)
can apply. This script

  1. checks every F trap of the 80-trap set against that rule, and
  2. rescores every system without the traps that fail it, with the same scorer
     (analyze_governance_eval.score_traps) and the same base-question-clustered bootstrap
     (seed 20260729, 10,000 draws) as the headline numbers.

Two exclusion sets are reported: the traps whose default is legitimate (removing them is a label
fix), and every trap that fails the rule (the strict construction check).

Writes results/governance_confirmatory_1014/trap_sensitivity.json.
"""
from __future__ import annotations

import json
import random
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.analyze_governance_eval import load_jsonl, normalize_rpal, normalize_vq, score_traps  # noqa: E402

G = ROOT / "results/governance_confirmatory"
G1014 = ROOT / "results/governance_confirmatory_1014"
CORE = ROOT / "results/paper_v2_250/fic/core.jsonl"
QUESTIONS = ROOT / "results/paper_v2_250/questions_250.jsonl"
OUT = G1014 / "trap_sensitivity.json"

SYSTEMS = {
    "vq_top4": ([G1014 / f"vq_k1_top4_run{i}.jsonl" for i in (1, 2, 3)], normalize_vq),
    "rpal_plain": ([G / "conf_plain_run1.jsonl"], normalize_rpal),
    "rpal_plus": ([G / f"conf_prompted_run{i}.jsonl" for i in (1, 2, 3)], normalize_rpal),
    "staged": ([G1014 / f"staged_s1_run{i}.jsonl" for i in (1, 2, 3)], normalize_rpal),
    "staged_spec": ([G1014 / f"staged_s2_run{i}.jsonl" for i in (1, 2, 3)], normalize_rpal),
}


def f_trap_check(traps) -> List[Dict]:
    cores = {json.loads(l)["fic_id"]: json.loads(l) for l in CORE.open()}
    qs = {json.loads(l)["question_id"]: json.loads(l) for l in QUESTIONS.open()}
    rows = []
    for t in traps:
        if t["operator"] != "F":
            continue
        g = t["grounding"]
        g = eval(g) if isinstance(g, str) else g  # noqa: S307 (stored as a Python repr in the trap file)
        field = g.get("removed_field")
        card = cores.get(g.get("fic_id") or g.get("base_fic_id"), {})
        inp = next((i for i in card.get("inputs", []) if i.get("name") == field), {})
        fn = qs.get(t["base_question_id"], {}).get("function", "")
        sig = re.search(rf"\b{re.escape(field)}\s*:\s*[\w\[\], ]+\s*=\s*([^,\)]+)", fn) if field else None
        rows.append({
            "trap_id": t["trap_id"], "removed_field": field, "removed_value": g.get("removed_value"),
            "contract_required": inp.get("required"),
            "contract_mentions_default": bool(re.search(r"\bdefault", inp.get("description", "") or "", re.I)),
            "function_default": sig.group(1).strip() if sig else None,
        })
    for r in rows:
        r["fails_rule"] = (r["contract_required"] is False) or r["contract_mentions_default"] or r["function_default"] is not None
    return rows


def flags(traps, path, normalizer) -> Dict[str, int]:
    scored = score_traps(traps, load_jsonl(path), normalizer)
    return {d["case_id"]: int(d["unsafe_numeric"]) for d in scored["details"]}


def main() -> None:
    traps = load_jsonl(G / "trap_set.jsonl")
    ids = [str(t["trap_id"]) for t in traps]
    base = {str(t["trap_id"]): t["base_question_id"] for t in traps}
    op = {str(t["trap_id"]): t["operator"] for t in traps}
    check = f_trap_check(traps)
    failing = [r["trap_id"] for r in check if r["fails_rule"]]
    # a default the reference function itself declares, other than a zero placeholder, is legitimate
    legit = [r["trap_id"] for r in check if r["fails_rule"] and r["function_default"] not in (None, "0", "0.0", "None")]

    per_run = {name: [flags(traps, p, norm) for p in paths] for name, (paths, norm) in SYSTEMS.items()}
    mean_flags = {name: {t: statistics.mean(r[t] for r in runs) for t in ids} for name, runs in per_run.items()}
    fnar = lambda f, sel: sum(f[t] for t in sel) / len(sel)  # noqa: E731

    def gap_ci(a: str, b: str, sel: Sequence[str]) -> Dict:
        clusters = defaultdict(list)
        for t in sel:
            clusters[base[t]].append(t)
        keys = list(clusters)
        rng = random.Random(20260729)
        boots = []
        for _ in range(10000):
            sample = [t for _ in keys for t in clusters[rng.choice(keys)]]
            boots.append(fnar(mean_flags[a], sample) - fnar(mean_flags[b], sample))
        boots.sort()
        return {"point": round(fnar(mean_flags[a], sel) - fnar(mean_flags[b], sel), 4),
                "ci95": [round(boots[249], 4), round(boots[9750], 4)], "n": len(sel), "clusters": len(keys)}

    variants = {"all_80": ids,
                "without_legitimate_default": [t for t in ids if t not in legit],
                "without_all_failing_rule": [t for t in ids if t not in failing]}
    result = {"f_trap_check": check, "failing_rule": failing, "legitimate_default": legit, "variants": {}}
    for vname, sel in variants.items():
        result["variants"][vname] = {
            "n": len(sel),
            "fnar": {name: round(fnar(mean_flags[name], sel), 4) for name in SYSTEMS},
            "f_operator_fnar": {name: round(fnar(mean_flags[name], [t for t in sel if op[t] == "F"]), 4) for name in SYSTEMS},
            "gap_rpal_plus_minus_vq": gap_ci("rpal_plus", "vq_top4", sel),
            "gap_staged_spec_minus_vq": gap_ci("staged_spec", "vq_top4", sel),
        }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("F traps failing the rule:", failing, "| legitimate default:", legit)
    for vname, v in result["variants"].items():
        g1, g2 = v["gap_rpal_plus_minus_vq"], v["gap_staged_spec_minus_vq"]
        print(f"\n{vname} (n={v['n']})")
        print("  FNAR:", {k: f"{100*x:.1f}%" for k, x in v["fnar"].items()})
        print("  F-operator FNAR:", {k: f"{100*x:.1f}%" for k, x in v["f_operator_fnar"].items()})
        print(f"  R-PAL+ minus VQ: {100*g1['point']:.1f} pts, 95% CI [{100*g1['ci95'][0]:.1f}, {100*g1['ci95'][1]:.1f}]")
        print(f"  Staged+Spec minus VQ: {100*g2['point']:.1f} pts, 95% CI [{100*g2['ci95'][0]:.1f}, {100*g2['ci95'][1]:.1f}]")
    print("\nwrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
