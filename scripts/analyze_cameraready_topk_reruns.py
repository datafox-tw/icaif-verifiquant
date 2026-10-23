#!/usr/bin/env python3
"""Compare the July top-3 runs with their 2026-10-15 top-4 reruns (camera-ready E13-E15).

No model calls. Each block uses the same arithmetic the paper already uses:

E13 out-of-library split (Table oodsel macros): strict 1% tolerance, as in
    run_ood_family_eval.classify; OOD abstain / correct / silent-wrong over 199,
    ID abstain over 51, selectivity gap = OOD abstain - ID abstain, OOD selective
    accuracy = correct / (correct + silent wrong).
E14 gate ablation (Table gateladder): unit-normalized, near misses count as correct.
E15 library sweep (Figure libsweep): plot_libsweep.point_stats on each sweep dir.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import plot_libsweep  # noqa: E402

RUNS = ROOT / "results"
OUT = RUNS / "cameraready_topk_reruns_summary.json"


def rows(p: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in open(p) if l.strip()]


def ood_block(path: Path) -> Dict[str, Any]:
    r = rows(path)
    out: Dict[str, Any] = {"file": str(path.relative_to(ROOT))}
    for split in ("OOD", "ID"):
        sub = [x for x in r if x["split"] == split]
        n = max(1, len(sub)) if sub else 1
        c = {v: sum(x["verdict"] == v for x in sub) for v in ("abstain", "correct", "silent_wrong")}
        out[split] = {"n": len(sub), **c, "abstain_rate": c["abstain"] / n, "correct_rate": c["correct"] / n,
                      "silent_wrong_rate": c["silent_wrong"] / n,
                      "selective_accuracy": c["correct"] / max(1, c["correct"] + c["silent_wrong"])}
    out["selectivity_gap"] = out["OOD"]["abstain_rate"] - out["ID"]["abstain_rate"]
    return out


def ladder_block(d: Path) -> Dict[str, Any]:
    out = {}
    for tag in ("r1_compute_only", "r2_fe", "r3_fei", "r4_full"):
        p = d / f"{tag}_summary.json"
        if not p.exists():
            continue
        s = json.loads(p.read_text())
        out[tag] = {"n": s["n"], "top_k": s.get("top_k", 3),
                    "acc": (s["correct"] + s["near_miss"]) / s["n"],
                    "swr": (s["silent_wrong"] - s["near_miss"]) / s["n"],
                    "abstain": s["abstain"]}
    return out


def sweep_block(d: Path) -> List[Dict[str, Any]]:
    return [p for p in (plot_libsweep.point_stats(f, str(d)) for f in plot_libsweep.FRACS) if p]


def pct(x: float) -> str:
    return f"{int(100 * x * 10 + 0.5 + 1e-9) / 10:.1f}"


def main() -> None:
    old_ood = sorted((RUNS / "ood_eval").glob("ood_results_20260713_005129.jsonl"))[0]
    new_ood = RUNS / "ood_eval_top4/ood_results_e13_top4.jsonl"
    report = {
        "e13_ood_split": {"top3_july": ood_block(old_ood),
                          "top4_rerun": ood_block(new_ood) if new_ood.exists() else None},
        "e14_gate_ladder": {"top3_july": ladder_block(RUNS / "gate_ladder"),
                            "top4_rerun": ladder_block(RUNS / "gate_ladder_top4")},
        "e15_library_sweep": {"top3_july": sweep_block(RUNS / "libsweep"),
                              "top4_rerun": sweep_block(RUNS / "libsweep_top4")},
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("E13 out-of-library split (VQ, 199 OOD + 51 ID)")
    for k in ("top3_july", "top4_rerun"):
        b = report["e13_ood_split"][k]
        if not b:
            continue
        o, i = b["OOD"], b["ID"]
        print(f"  {k:11s} n={o['n']}+{i['n']}  OOD abstain {pct(o['abstain_rate'])}  correct {pct(o['correct_rate'])}  "
              f"SW {pct(o['silent_wrong_rate'])}  SelAcc {pct(o['selective_accuracy'])} | ID abstain {pct(i['abstain_rate'])} "
              f"acc {pct(i['correct_rate'])} | gap {pct(b['selectivity_gap'])}")

    print("\nE14 gate ladder (250 clean, unit-normalized)")
    for tag in ("r1_compute_only", "r2_fe", "r3_fei", "r4_full"):
        line = f"  {tag:16s}"
        for k in ("top3_july", "top4_rerun"):
            b = report["e14_gate_ladder"][k].get(tag)
            line += (f" | {k}: acc {pct(b['acc'])} SWR {pct(b['swr'])} abst {b['abstain']} (n={b['n']})"
                     if b else f" | {k}: -")
        print(line)

    print("\nE15 library sweep (unit-normalized)")
    for k in ("top3_july", "top4_rerun"):
        print(f"  {k}")
        for p in report["e15_library_sweep"][k]:
            f = lambda x: "  n/a" if x is None else pct(x)
            print(f"    lib_{p['frac']:>3}: ID acc {f(p['id_acc'])} ID abst {f(p['id_abstain'])} ID SW {f(p['id_sw'])} | "
                  f"OOD abst {f(p['ood_abstain'])} OOD SW {f(p['ood_sw'])} | gap {f(p['selectivity_gap'])} "
                  f"n={p['n_id']}+{p['n_ood']}{'' if p['complete'] else ' PARTIAL'}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
