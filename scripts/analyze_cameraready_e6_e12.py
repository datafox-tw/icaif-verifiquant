#!/usr/bin/env python3
"""Score camera-ready E12 (Gemini 2.5 Pro backbone) and E6 (scripted user).

No model calls.

E12: VerifiQuant and R-PAL+ on the 80 traps with gemini-2.5-pro, three runs each,
scored exactly like Table 2 Panel B (score_traps + normalize_vq / normalize_rpal).
The VQ runs were sharded 4 ways; shards are concatenated per run here.

E6: the scripted-user rows from run_scripted_user_eval.py (26 traps: 20 F + 6 I_HARD,
three runs per system). Per run and pooled:
  number at turn 1   the system committed a number before any reply
  named              turn 1 withheld a number and named the removed field/convention
  recovered (strict) named, and turn 2 is correct
  recovered (lenient) withheld at turn 1, and turn 2 is correct
  wrong after reply  withheld at turn 1, and turn 2 is a wrong number
"""
from __future__ import annotations

import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from analyze_governance_eval import load_jsonl, normalize_rpal, normalize_vq, score_traps  # noqa: E402

RUNS = ROOT / "results"
G = RUNS / "governance_confirmatory"
PRO = RUNS / "governance_confirmatory_1015_pro"
SU = RUNS / "scripted_user_1015"
OUT = RUNS / "cameraready_e6_e12_summary.json"


def mean(xs: Sequence[float]) -> float:
    return statistics.mean(xs) if xs else float("nan")


def cluster_ci(a: Dict[str, float], b: Dict[str, float], ids: List[str], base_of: Dict[str, str]) -> Dict[str, Any]:
    clusters = defaultdict(list)
    for i in ids:
        clusters[base_of[i]].append(i)
    keys = list(clusters)
    rng = random.Random(20260729)
    boots = []
    for _ in range(10000):
        sample: List[str] = []
        for _ in keys:
            sample.extend(clusters[rng.choice(keys)])
        boots.append(mean([a[i] for i in sample]) - mean([b[i] for i in sample]))
    boots.sort()
    point = mean([a[i] for i in ids]) - mean([b[i] for i in ids])
    return {"point": round(point, 4), "ci95": [round(boots[249], 4), round(boots[9750], 4)], "clusters": len(keys)}


def e12(traps, base_of) -> Dict[str, Any]:
    ids = [str(t["trap_id"]) for t in traps]
    systems = {
        "vq_pro": ([sorted(PRO.glob(f"vq_pro_top4_run{r}_shard*.jsonl")) for r in (1, 2, 3)], normalize_vq),
        "rpal_plus_pro": ([[PRO / f"rpal_plus_pro_run{r}.jsonl"] for r in (1, 2, 3)], normalize_rpal),
    }
    out: Dict[str, Any] = {}
    flags_mean: Dict[str, Dict[str, float]] = {}
    for name, (run_files, norm) in systems.items():
        scored = []
        api_errors = []
        for files in run_files:
            rows = [r for f in files if f.exists() for r in load_jsonl(f)]
            api_errors.append(sum(1 for r in rows if str(r.get("execution_reason", "")).startswith("api")))
            scored.append(score_traps(traps, rows, norm))
        flags = [{d["case_id"]: float(d["unsafe_numeric"]) for d in s["details"]} for s in scored]
        flags_mean[name] = {i: mean([f[i] for f in flags]) for i in ids}
        out[name] = {
            "rows_per_run": [s["overall"]["n"] - s["overall"]["missing"] for s in scored],
            "api_error_rows_per_run": api_errors,
            "fnar_per_run": [round(s["overall"]["fnar"], 4) for s in scored],
            "fnar_mean": round(mean([s["overall"]["fnar"] for s in scored]), 4),
            "diag_acc_mean": round(mean([s["overall"]["diagnostic_accuracy"] for s in scored]), 4),
            "action_acc_mean": round(mean([s["overall"]["action_accuracy"] for s in scored]), 4),
            "per_operator_fnar": {op: round(mean([s["per_operator"][op]["fnar"] for s in scored]), 4)
                                  for op in ("F", "E", "I", "N")},
            "i_split_fnar": {lvl: round(mean([sum(d["unsafe_numeric"] for d in s["details"]
                                                   if d["expected_diagnostic"] == lvl)
                                               / sum(1 for d in s["details"] if d["expected_diagnostic"] == lvl)
                                               for s in scored]), 4)
                             for lvl in ("I_HARD", "I_SOFT")},
        }
    out["gap_rpal_plus_pro_minus_vq_pro"] = cluster_ci(flags_mean["rpal_plus_pro"], flags_mean["vq_pro"], ids, base_of)
    flash = json.loads((RUNS / "governance_confirmatory_1014/confirmatory_summary_top4.json").read_text())
    out["flash_reference"] = {
        "vq_top4_fnar": flash["overall"]["vq_k1"]["fnar"],
        "rpal_plus_fnar": flash["overall"]["rpal_prompted_mean"]["fnar"],
        "gap": flash["prompted_minus_vq_fnar_gap"],
    }
    return out


def e6(traps, base_of) -> Dict[str, Any]:
    trap_ids = [str(t["trap_id"]) for t in traps if t["operator"] == "F" or t["expected_diagnostic"] == "I_HARD"]
    op_of = {str(t["trap_id"]): t["operator"] for t in traps}
    out: Dict[str, Any] = {}
    per_trap: Dict[str, Dict[str, Dict[str, float]]] = {}
    for system in ("vq", "rpal_plus"):
        runs = []
        for r in (1, 2, 3):
            p = SU / f"{system}_run{r}.jsonl"
            if p.exists():
                runs.append({row["trap_id"]: row for row in load_jsonl(p)})
        if not runs:
            continue

        def outcome(row: Dict[str, Any]) -> Dict[str, bool]:
            t1, t2 = row["turn1"], row.get("turn2") or {}
            withheld = not t1["numeric"]
            got = withheld and t2.get("output") is not None
            return {
                "number_at_turn1": t1["numeric"],
                "withheld": withheld,
                "named": withheld and t1["named_removed_item"],
                "recovered_lenient": got and bool(t2.get("correct")),
                "recovered_strict": got and bool(t2.get("correct")) and t1["named_removed_item"],
                "wrong_after_reply": got and not t2.get("correct"),
                "still_withheld": withheld and t2.get("output") is None,
            }

        keys = ("number_at_turn1", "withheld", "named", "recovered_strict", "recovered_lenient",
                "wrong_after_reply", "still_withheld")
        block: Dict[str, Any] = {"runs": len(runs), "missing_rows": [len(set(trap_ids) - set(r)) for r in runs]}
        for subset_name, subset in (("all_26", trap_ids), ("F_20", [i for i in trap_ids if op_of[i] == "F"]),
                                    ("I_HARD_6", [i for i in trap_ids if op_of[i] == "I"])):
            counts = {k: [sum(outcome(r[i])[k] for i in subset if i in r) for r in runs] for k in keys}
            block[subset_name] = {k: {"per_run": v, "mean_rate": round(mean(v) / len(subset), 4)}
                                  for k, v in counts.items()}
        out[system] = block
        per_trap[system] = {k: {i: mean([float(outcome(r[i])[k]) for r in runs if i in r]) for i in trap_ids}
                            for k in ("recovered_strict", "recovered_lenient")}
    if len(per_trap) == 2:
        out["gap_vq_minus_rpal_plus"] = {
            k: cluster_ci(per_trap["vq"][k], per_trap["rpal_plus"][k], trap_ids, base_of)
            for k in ("recovered_strict", "recovered_lenient")
        }
    return out


def main() -> None:
    traps = load_jsonl(G / "trap_set.jsonl")
    base_of = {str(t["trap_id"]): t["base_question_id"] for t in traps}
    report = {"e12_pro_backbone": e12(traps, base_of), "e6_scripted_user": e6(traps, base_of)}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    r = report["e12_pro_backbone"]
    print("E12 (gemini-2.5-pro, 80 traps, 3 runs each)")
    for name in ("rpal_plus_pro", "vq_pro"):
        s = r[name]
        print(f"  {name:14s} FNAR {s['fnar_mean']:.3f} {s['fnar_per_run']}  per-op {s['per_operator_fnar']}  "
              f"I split {s['i_split_fnar']}  diag {s['diag_acc_mean']:.3f} act {s['action_acc_mean']:.3f}  "
              f"rows {s['rows_per_run']} api-errors {s['api_error_rows_per_run']}")
    g = r["gap_rpal_plus_pro_minus_vq_pro"]
    print(f"  gap R-PAL+ - VQ: {g['point']:+.4f} CI [{g['ci95'][0]:+.3f}, {g['ci95'][1]:+.3f}]  "
          f"(Flash: {r['flash_reference']['gap']})")

    e = report["e6_scripted_user"]
    print("\nE6 scripted user (26 traps = 20 F + 6 I_HARD; counts per run, rate = mean / n)")
    for system in ("vq", "rpal_plus"):
        if system not in e:
            continue
        print(f"  {system}  missing={e[system]['missing_rows']}")
        for subset in ("all_26", "F_20", "I_HARD_6"):
            b = e[system][subset]
            print(f"    {subset:9s} " + "  ".join(f"{k}={v['per_run']}" for k, v in b.items()))
    for k, v in e.get("gap_vq_minus_rpal_plus", {}).items():
        print(f"  VQ - R-PAL+ {k}: {v['point']:+.4f} CI [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] ({v['clusters']} clusters)")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
