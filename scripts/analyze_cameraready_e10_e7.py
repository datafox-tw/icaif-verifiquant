#!/usr/bin/env python3
"""Score the camera-ready reruns (E10) and the staged-prompt control (E7).

No model calls. Every number comes from the raw jsonl files through the same
``score_traps`` + ``normalize_vq`` / ``normalize_rpal`` used for the paper's
Table 2 (analyze_governance_confirmatory.py), so the definitions match.

Systems
  vq_july_top3     governance_confirmatory/vq_k1_traps.jsonl (the paper's 11.3%)
  vq_top3_rerun    1014 rerun with the July settings (top-3)
  vq_top4_run1..3  1014 reruns at the top-4 depth the paper states
  rpal_plus_july   conf_prompted_run1..3 (the paper's 27.5% mean)
  rpal_plus_run4   1014 rerun, the drift control
  s1_run1..3       staged R-PAL+ (staging only)
  s2_run1..3       staged R-PAL+ with contract text (not executed)

Secondary metric, labelled heuristic: on I_SOFT traps the scorer accepts any
non-empty warning. ``targeted_warning`` additionally asks that the warning or
clarification mention the removed convention (keyword map below).
"""
from __future__ import annotations

import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from scripts.analyze_governance_eval import (  # noqa: E402
    clean_labels_from_rpal, clean_labels_from_vq, clean_summary,
    load_jsonl, normalize_rpal, normalize_vq, score_traps,
)

G = ROOT / "results/governance_confirmatory"
R = ROOT / "results/governance_confirmatory_1014"
CLEAN = ROOT / "results/paper_v2_250/questions_250.jsonl"

FAMILY_KEYWORDS = {
    "scale": ("percent", "%", "decimal", "scale", "fraction"),
    "time_unit": ("month", "year", "time unit", "period"),
    "option_type": ("call", "put", "option type", "per share", "per-share", "total premium", "premium"),
    "inventory": ("fifo", "lifo", "inventory", "costing", "valuation method"),
}


def systems() -> Dict[str, List[Path]]:
    out: Dict[str, List[Path]] = {
        "vq_july_top3": [G / "vq_k1_traps.jsonl"],
        "vq_top3_rerun": [R / "vq_k1_top3_run2.jsonl"],
        "vq_top4": [R / f"vq_k1_top4_run{i}.jsonl" for i in (1, 2, 3)],
        "rpal_plain_july": [G / "conf_plain_run1.jsonl"],
        "rpal_plus_july": [G / f"conf_prompted_run{i}.jsonl" for i in (1, 2, 3)],
        "rpal_plus_run4": [R / "conf_prompted_run4.jsonl"],
        "s1": [R / f"staged_s1_run{i}.jsonl" for i in (1, 2, 3)],
        "s2": [R / f"staged_s2_run{i}.jsonl" for i in (1, 2, 3)],
    }
    return {k: [p for p in v if p.exists()] for k, v in out.items()}


def normalizer_for(name: str):
    return normalize_vq if name.startswith("vq") else normalize_rpal


def run_flags(traps, path: Path, normalizer) -> Dict[str, Dict[str, Any]]:
    rows = load_jsonl(path)
    scored = score_traps(traps, rows, normalizer)
    by_id = {str(r.get("case_id")): normalizer(r) for r in rows}
    trap_by_id = {str(t["trap_id"]): t for t in traps}
    out = {}
    for d in scored["details"]:
        cid = d["case_id"]
        res = by_id.get(cid)
        trap = trap_by_id[cid]
        targeted = None
        if trap["expected_diagnostic"] == "I_SOFT":
            fam = (trap.get("grounding") or {}).get("ambiguity_family", "")
            text = " ".join([str((res or {}).get("warning") or ""),
                             str((res or {}).get("clarification_question") or "")]).lower()
            targeted = bool(res) and d["got_diagnostic"] == "I_SOFT" and any(
                k in text for k in FAMILY_KEYWORDS.get(fam, ()))
        out[cid] = {**d, "targeted_warning": targeted,
                    "missing": res is None,
                    "stopped_at": ((res or {}).get("raw", {}).get("assessment") or {}).get("stopped_at")
                    if isinstance((res or {}).get("raw"), dict) else None}
    return out


def mean(xs: Sequence[float]) -> float:
    return statistics.mean(xs) if xs else float("nan")


def main() -> None:
    traps = load_jsonl(G / "trap_set.jsonl")
    ids = [str(t["trap_id"]) for t in traps]
    op_of = {str(t["trap_id"]): t["operator"] for t in traps}
    exp_of = {str(t["trap_id"]): t["expected_diagnostic"] for t in traps}
    base_of = {str(t["trap_id"]): t["base_question_id"] for t in traps}

    report: Dict[str, Any] = {"n_traps": len(ids), "systems": {}}
    per_trap_mean: Dict[str, Dict[str, float]] = {}
    for name, paths in systems().items():
        if not paths:
            continue
        runs = [run_flags(traps, p, normalizer_for(name)) for p in paths]

        def rate(key: str, sel: Sequence[str], run) -> float:
            return sum(int(bool(run[i][key])) for i in sel) / len(sel)

        def block(sel: Sequence[str]) -> Dict[str, Any]:
            per_run = [rate("unsafe_numeric", sel, r) for r in runs]
            return {"n": len(sel), "fnar_mean": round(mean(per_run), 4),
                    "fnar_per_run": [round(x, 4) for x in per_run]}

        sysrep = {
            "files": [str(p.relative_to(ROOT)) for p in paths],
            "missing_rows": [sum(r[i]["missing"] for i in ids) for r in runs],
            "overall": block(ids),
            "diag_acc_mean": round(mean([rate("diagnostic_correct", ids, r) for r in runs]), 4),
            "action_acc_mean": round(mean([rate("action_correct", ids, r) for r in runs]), 4),
            "per_operator": {op: block([i for i in ids if op_of[i] == op]) for op in ("F", "E", "I", "N")},
            "i_split": {lvl: block([i for i in ids if exp_of[i] == lvl]) for lvl in ("I_HARD", "I_SOFT")},
        }
        soft = [i for i in ids if exp_of[i] == "I_SOFT"]
        sysrep["i_soft_targeted_warning_mean"] = round(mean(
            [sum(int(bool(r[i]["targeted_warning"])) for i in soft) / len(soft) for r in runs]), 4)
        # where staged systems stop, pooled over runs
        stops = defaultdict(int)
        for r in runs:
            for i in ids:
                stops[str(r[i]["stopped_at"])] += 1
        if any(k != "None" for k in stops):
            sysrep["stopped_at_pooled"] = dict(sorted(stops.items()))
        # got-diagnostic confusion on I traps, pooled
        conf = defaultdict(int)
        for r in runs:
            for i in ids:
                if op_of[i] == "I":
                    conf[f"{exp_of[i]}->{r[i]['got_diagnostic']}{'(num)' if r[i]['unsafe_numeric'] else ''}"] += 1
        sysrep["i_confusion_pooled"] = dict(sorted(conf.items()))
        report["systems"][name] = sysrep
        per_trap_mean[name] = {i: mean([float(r[i]["unsafe_numeric"]) for r in runs]) for i in ids}

    # base-clustered bootstrap on FNAR gaps (same scheme and seed as the paper)
    clusters = defaultdict(list)
    for i in ids:
        clusters[base_of[i]].append(i)
    keys = list(clusters)

    def gap_ci(a: str, b: str) -> Dict[str, Any]:
        if a not in per_trap_mean or b not in per_trap_mean:
            return {}
        fa, fb = per_trap_mean[a], per_trap_mean[b]
        point = mean([fa[i] for i in ids]) - mean([fb[i] for i in ids])
        rng = random.Random(20260729)
        boots = []
        for _ in range(10000):
            sample: List[str] = []
            for _ in keys:
                sample.extend(clusters[rng.choice(keys)])
            boots.append(mean([fa[i] for i in sample]) - mean([fb[i] for i in sample]))
        boots.sort()
        return {"point": round(point, 4), "ci95": [round(boots[249], 4), round(boots[9750], 4)]}

    report["gaps"] = {
        "rpal_plus_july - vq_july_top3": gap_ci("rpal_plus_july", "vq_july_top3"),
        "rpal_plus_july - vq_top4": gap_ci("rpal_plus_july", "vq_top4"),
        "rpal_plus_july - s1": gap_ci("rpal_plus_july", "s1"),
        "s1 - vq_top4": gap_ci("s1", "vq_top4"),
        "s1 - s2": gap_ci("s1", "s2"),
        "s2 - vq_top4": gap_ci("s2", "vq_top4"),
        "vq_top4 - vq_july_top3": gap_ci("vq_top4", "vq_july_top3"),
    }
    # ---- clean 250: does a low trap FNAR come with blanket abstention? ----
    questions = {str(q["question_id"]): q for q in load_jsonl(CLEAN)}
    clean: Dict[str, Any] = {}
    vq_rows = load_jsonl(ROOT / "results/paper_v2_250/results/vq_flash/output.jsonl")
    clean["vq_k1 (first turn of canonical K=3 run)"] = clean_summary(clean_labels_from_vq(vq_rows, set(questions)))
    for i in (1, 2, 3):
        p = ROOT / f"results/governance_eval/prompted_clean250_run{i}.jsonl"
        clean[f"rpal_plus_run{i}"] = clean_summary(
            clean_labels_from_rpal(load_jsonl(p), questions, use_stored_verdict=False))
    vq_labels = clean_labels_from_vq(vq_rows, set(questions))
    from scipy.stats import binomtest

    def mcnemar(a_lab, b_lab, key):
        f = (lambda L, i: L.get(i, "abstain") != "abstain") if key == "coverage" else (lambda L, i: L.get(i) == "correct")
        x = sum(1 for i in questions if f(a_lab, i) and not f(b_lab, i))
        y = sum(1 for i in questions if f(b_lab, i) and not f(a_lab, i))
        return {"vq_only": x, "other_only": y, "p": binomtest(x, x + y, 0.5).pvalue if x + y else 1.0}

    for variant in ("s1", "s2"):
        # run 1 was written without a run tag; runs 2 and 3 (E16) carry one.
        run_globs = [f"staged_{variant}_clean250_shard*.jsonl"] + [
            f"staged_{variant}_clean250_run{r}_shard*.jsonl" for r in (2, 3)]
        per_run = []
        for r, pattern in enumerate(run_globs, 1):
            shards = sorted(R.glob(pattern))
            if not shards:
                continue
            v_rows = [row for p in shards for row in load_jsonl(p)]
            v_labels = clean_labels_from_rpal(v_rows, questions, use_stored_verdict=False)
            summ = {**clean_summary(v_labels), "rows": len(v_rows)}
            diag_counts: Dict[str, int] = defaultdict(int)
            exec_fail = 0
            for row in v_rows:
                a = row.get("assessment") or {}
                diag_counts[str(a.get("diagnostic", "FAIL"))] += 1
                exec_fail += a.get("stopped_at") == "program" and row.get("output") is None
            summ["diagnostic_counts"] = dict(sorted(diag_counts.items()))
            summ["program_failures"] = exec_fail
            summ["mcnemar_vs_vq_coverage"] = mcnemar(vq_labels, v_labels, "coverage")
            summ["mcnemar_vs_vq_accuracy"] = mcnemar(vq_labels, v_labels, "accuracy")
            clean[f"{variant}_run{r}"] = summ
            per_run.append(summ)
        if per_run:
            clean[f"{variant}_mean"] = {
                k: statistics.mean(x[k] for x in per_run)
                for k in ("accuracy", "swr", "coverage", "selective_accuracy", "abstain", "program_failures")
            } | {"runs": len(per_run), "complete_rows": [x["rows"] for x in per_run]}
    report["clean250"] = {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                          for k, v in clean.items()}

    out = R / "cameraready_e10_e7_summary.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"{'system':18s} {'FNAR':>6s}  {'per run':24s} {'F':>5s} {'E':>5s} {'I':>5s} {'N':>5s} "
          f"{'I_H':>5s} {'I_S':>5s} {'diag':>5s} {'act':>5s} {'tgtW':>5s}")
    for name, s in report["systems"].items():
        po, sp = s["per_operator"], s["i_split"]
        print(f"{name:18s} {s['overall']['fnar_mean']:6.3f}  {str(s['overall']['fnar_per_run']):24s} "
              f"{po['F']['fnar_mean']:5.2f} {po['E']['fnar_mean']:5.2f} {po['I']['fnar_mean']:5.2f} "
              f"{po['N']['fnar_mean']:5.2f} {sp['I_HARD']['fnar_mean']:5.2f} {sp['I_SOFT']['fnar_mean']:5.2f} "
              f"{s['diag_acc_mean']:5.2f} {s['action_acc_mean']:5.2f} {s['i_soft_targeted_warning_mean']:5.2f}"
              + (f"  missing={s['missing_rows']}" if any(s["missing_rows"]) else ""))
    print("\nFNAR gaps (base-clustered bootstrap, 52 clusters):")
    for k, v in report["gaps"].items():
        if v:
            print(f"  {k:32s} {v['point']:+.4f}  CI [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")
    print("\nClean 250 (acc / SWR / coverage / selective acc):")
    for k, v in report["clean250"].items():
        extra = ""
        if "mcnemar_vs_vq_coverage" in v:
            mc, ma = v["mcnemar_vs_vq_coverage"], v["mcnemar_vs_vq_accuracy"]
            extra = (f"  prog_fail={v['program_failures']}  vs VQ cov {mc['vq_only']}/{mc['other_only']} p={mc['p']:.4f}"
                     f"  acc {ma['vq_only']}/{ma['other_only']} p={ma['p']:.4f}")
        elif "runs" in v:
            extra = f"  (mean of {v['runs']} runs, rows {v['complete_rows']}, prog_fail {v['program_failures']:.1f})"
        print(f"  {k:42s} {v['accuracy']:.3f} / {v['swr']:.3f} / {v['coverage']:.3f} / {v['selective_accuracy']:.3f}{extra}")
    for name in ("s1", "s2"):
        if name in report["systems"]:
            print(f"\n{name} stopped_at (pooled): {report['systems'][name].get('stopped_at_pooled')}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
