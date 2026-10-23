"""
score_trap_set.py
-----------------
Score a system's run over the contract-grounded trap set (build_trap_set.py).

Unlike the clean set, a trap has NO valid numeric answer: the correct behavior is
to *intercept* (refuse / clarify / abstain / proceed-with-warning), grounded in the
trap's `expected_behavior`. We therefore score interception, not arithmetic:

  Caught       — the system intercepted the trap at an appropriate layer
                 (refusal / alert / needs_clarification / error / diagnostic flag),
                 or, for I_SOFT, proceeded *with an explicit warning*.
  Silent-Wrong — the system returned a confident numeric answer with NO flag at all.
  Error        — the system crashed / produced no answer and no diagnostic.

  Trap SWR = Silent-Wrong / total_traps      (lower is better; the headline metric)

A vanilla CoT pipeline has no interception mechanism, so every trap it answers is
Silent-Wrong by construction — that asymmetry is the point of the comparison.

Usage:
    python3 verifiquant/pipeline/score_trap_set.py \
        --trap-set results/paper_v1/trap/trap_set.jsonl \
        --output   results/paper_v1/trap/results/vq_flash_v3/output.jsonl \
        --label    VQ_Flash_V3 \
        --report   results/paper_v1/trap/results/vq_flash_v3/trap_score.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

# diagnostic_type values that mean "the funnel flagged something"
_FLAGGED = lambda d: str(d) not in ("", "None", "Unknown", "none", "null")
# statuses that mean the system intercepted rather than silently answered
_INTERCEPT_STATUS = {"refusal", "alert", "needs_clarification", "error"}


def _load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _extract(result_row, use_round1=False):
    """Normalize a system output row into (status, diagnostic, answer, soft_flag, structured).

    `structured` marks systems with a machine-readable interception layer (VQ).
    A vanilla CoT has none: its only signals are a parsed numeric answer or its
    absence (an *informal*, prose-only refusal that no downstream system can route).

    `soft_flag` is True when VQ attached an I_SOFT warning (proceed-with-warning):
    a confident answer accompanied by a machine-readable ambiguity flag — NOT silent.

    `use_round1` reads the initial-gate diagnostic instead of the post-O-ITL final
    state. On TRAPS this matters: the O-ITL oracle reads the stored ground_truth and
    can re-inject a redacted/illegal value, masking the funnel's interception. The
    fair "did the funnel catch it" measurement is the funnel-only round-1 gate.

    Handles both wrapper schemas:
      VQ : framework_guided.{final_*, history[].diagnostic}
      CoT: cot_self_improve.{final_answer}
    """
    if "framework_guided" in result_row:
        fg = result_row["framework_guided"]
        if use_round1 and fg.get("history"):
            d = fg["history"][0].get("diagnostic", {})
            status = d.get("status")
            diag = d.get("diagnostic_type")
            ans = d.get("output_value")
            soft = bool(d.get("has_i_soft") or d.get("soft_warnings"))
        else:
            status = fg.get("final_status")
            diag = fg.get("final_diagnostic_type")
            ans = fg.get("final_output_value")
            # final-level soft flag: derive from last round if present
            soft = False
            if fg.get("history"):
                d = fg["history"][-1].get("diagnostic", {})
                soft = bool(d.get("has_i_soft") or d.get("soft_warnings"))
        return (status, diag, ans, soft, True)
    if "cot_self_improve" in result_row:
        c = result_row["cot_self_improve"]
        return (None, None, c.get("final_answer"), False, False)
    return (result_row.get("final_status"),
            result_row.get("final_diagnostic_type"),
            result_row.get("final_output_value"), False, True)


def classify(expected_behavior, status, diagnostic, answer, soft_flag, structured):
    """Expected-behavior-aware verdict. Returns one of:
      'caught'           — the system intercepted in a way appropriate to the trap.
      'abstain_informal' — no confident answer, but no structured diagnostic
                           (e.g. CoT refusing in prose). Not silent-wrong, no audit trail.
      'silent_wrong'     — a confident numeric answer where the trap demanded interception.

    Key rule: a soft (I_SOFT) warning only counts as a legitimate catch when the trap
    *expects* proceed-with-warning (I_SOFT). For F/E/N/M/I_HARD the system must HARD-
    intercept (refusal / alert / needs_clarification / error, or raise the matching
    hard diagnostic). A stray soft warning while computing on missing / illegal /
    out-of-scope input is NOT a catch — it is a soft-flagged wrong answer.
    """
    if not structured:                      # CoT: no interception layer at all
        return "abstain_informal" if answer is None else "silent_wrong"

    hard_intercept = (status in _INTERCEPT_STATUS) or (_FLAGGED(diagnostic)
                                                       and str(diagnostic) != "I_SOFT")
    if expected_behavior == "proceed_with_warning":      # I_SOFT trap
        if hard_intercept or soft_flag:
            return "caught"
        return "abstain_informal" if answer is None else "silent_wrong"
    # F / E / N / M / I_HARD : must hard-intercept
    if hard_intercept:
        return "caught"
    if answer is None:
        return "abstain_informal"
    return "silent_wrong"


def score(trap_path, output_path, label, use_round1=False):
    traps = {t.get("trap_id", t.get("question_id")): t for t in _load_jsonl(trap_path)}
    rows = {}
    for r in _load_jsonl(output_path):
        cid = r.get("case_id") or r.get("question_id") or r.get("source_sample_id")
        rows[cid] = r

    _zero = lambda: {"caught": 0, "abstain_informal": 0, "silent_wrong": 0, "total": 0}
    per_op = defaultdict(_zero)
    confusion = defaultdict(lambda: defaultdict(int))   # expected_op -> diagnostic -> n
    details = []
    overall = {"caught": 0, "abstain_informal": 0, "silent_wrong": 0, "total": 0, "missing": 0}

    for tid, t in traps.items():
        op = t["operator"]
        per_op[op]["total"] += 1
        overall["total"] += 1
        r = rows.get(tid)
        if r is None:
            overall["missing"] += 1
            details.append({"trap_id": tid, "operator": op, "verdict": "missing"})
            continue
        status, diag, ans, soft, structured = _extract(r, use_round1=use_round1)
        verdict = classify(t["expected_behavior"], status, diag, ans, soft, structured)
        per_op[op][verdict] += 1
        overall[verdict] += 1
        _key = (str(diag) if _FLAGGED(diag)
                else ("I_SOFT_warn" if soft
                      else (status or ("answered" if ans is not None else "no_answer"))))
        confusion[op][_key] += 1
        details.append({
            "trap_id": tid, "operator": op,
            "expected_diagnostic": t["expected_diagnostic"],
            "expected_behavior": t["expected_behavior"],
            "got_status": status, "got_diagnostic": diag, "got_answer": ans,
            "verdict": verdict,
        })

    n = overall["total"]
    report = {
        "label": label,
        "total_traps": n,
        "caught": overall["caught"],
        "abstain_informal": overall["abstain_informal"],
        "silent_wrong": overall["silent_wrong"],
        "missing": overall["missing"],
        "trap_swr": round(overall["silent_wrong"] / n, 4) if n else None,
        # structured catch only (VQ funnel); CoT prose refusals land in abstain_informal
        "structured_catch_rate": round(overall["caught"] / n, 4) if n else None,
        # not-silent-wrong = caught + informal abstain (avoided a confident wrong answer)
        "intercept_rate": round((overall["caught"] + overall["abstain_informal"]) / n, 4) if n else None,
        "per_operator": {},
        "confusion_matrix": {op: dict(cols) for op, cols in confusion.items()},
        "details": details,
    }
    for op, c in per_op.items():
        tot = c["total"]
        report["per_operator"][op] = {
            **c,
            "trap_swr": round(c["silent_wrong"] / tot, 4) if tot else None,
        }
    return report


def _print(report):
    print(f"\n{'='*64}\n  Trap scoring: {report['label']}\n{'='*64}")
    print(f"  total={report['total_traps']}  caught(structured)={report['caught']}  "
          f"abstain_informal={report['abstain_informal']}  "
          f"silent_wrong={report['silent_wrong']}  missing={report['missing']}")
    print(f"  Trap SWR = {report['trap_swr']:.1%}   "
          f"structured-catch = {report['structured_catch_rate']:.1%}   "
          f"intercept (not silent-wrong) = {report['intercept_rate']:.1%}")
    print(f"\n  {'Operator':<10}{'caught':>8}{'informal':>10}{'silent':>8}{'TrapSWR':>10}")
    for op in ("F", "E", "I", "N", "M"):
        if op in report["per_operator"]:
            c = report["per_operator"][op]
            print(f"  {op:<10}{c['caught']:>8}{c['abstain_informal']:>10}{c['silent_wrong']:>8}"
                  f"{c['trap_swr']:>9.0%}")


def main():
    ap = argparse.ArgumentParser(description="Score a run over the trap set.")
    ap.add_argument("--trap-set", required=True)
    ap.add_argument("--output", required=True, help="system output.jsonl")
    ap.add_argument("--label", required=True)
    ap.add_argument("--report", default=None)
    ap.add_argument("--round1", action="store_true",
                    help="score the initial-gate (funnel-only) diagnostic instead of the "
                         "post-O-ITL final state; required for fair TRAP scoring of VQ "
                         "(the oracle leaks stored ground_truth on traps).")
    args = ap.parse_args()
    report = score(args.trap_set, args.output, args.label, use_round1=args.round1)
    _print(report)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n  report -> {args.report}")


if __name__ == "__main__":
    main()
