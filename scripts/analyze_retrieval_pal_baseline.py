#!/usr/bin/env python3
"""Audit and compare the matched retrieval+PAL baseline without new LLM calls.

The raw run is immutable.  This script deterministically replays cases that the
legacy regex denylist rejected, applies the corrected AST policy, and rescores
all returned values with boolean-aware numeric parsing.  It then reports paired
comparisons against VQ K=1, VQ K=3, and the legacy plain-PAL run.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_pal_baseline import deny_reason, run_code  # noqa: E402
from run_retrieval_pal_baseline import _score_output  # noqa: E402


DEFAULT_RPAL = (
    ROOT / "results/retrieval_pal_baseline/rpal_top4_gemini_25_flash.jsonl"
)
DEFAULT_VQ = ROOT / "results/paper_v2_250/results/vq_flash/output.jsonl"
DEFAULT_PAL = ROOT / "results/pal_baseline/pal_gemini_25_flash.jsonl"
DEFAULT_QUESTIONS = ROOT / "results/paper_v2_250/questions_250.jsonl"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def rates(labels: Sequence[str]) -> Dict[str, Any]:
    counts = Counter(labels)
    n = len(labels)
    answered = counts["correct"] + counts["silent_wrong"]
    return {
        "n": n,
        "correct": counts["correct"],
        "silent_wrong": counts["silent_wrong"],
        "abstain": counts["abstain"],
        "accuracy": counts["correct"] / n if n else 0.0,
        "swr": counts["silent_wrong"] / n if n else 0.0,
        "coverage": answered / n if n else 0.0,
        "selective_accuracy": counts["correct"] / answered if answered else 0.0,
    }


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> List[float]:
    if n == 0:
        return [0.0, 0.0]
    p = successes / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    radius = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def with_intervals(labels: Sequence[str]) -> Dict[str, Any]:
    result = rates(labels)
    result["accuracy_wilson_95"] = wilson(result["correct"], result["n"])
    result["swr_wilson_95"] = wilson(result["silent_wrong"], result["n"])
    return result


def audit_rpal(
    rows: Sequence[Mapping[str, Any]],
    questions_by_id: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    audited: List[Dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        case_id = str(row["case_id"])
        question = str(questions_by_id[case_id].get("question", ""))
        action = "rescored_unchanged"
        detail = ""

        if row.get("reason") == "denylist":
            code = str(row.get("generated_code") or "")
            detail = deny_reason(code)
            if detail:
                score = {
                    "verdict": "abstain",
                    "verdict_unit_normalized": "abstain",
                    "near_miss": False,
                    "numeric_output": None,
                }
                action = "confirmed_ast_denial"
            else:
                replayed = run_code(code)
                if replayed is None:
                    row["reason"] = "audit_exec_fail"
                    score = {
                        "verdict": "abstain",
                        "verdict_unit_normalized": "abstain",
                        "near_miss": False,
                        "numeric_output": None,
                    }
                    action = "legacy_regex_false_positive_replay_failed"
                elif "CANNOT SOLVE" in str(replayed).upper():
                    row["reason"] = "declined"
                    score = {
                        "verdict": "abstain",
                        "verdict_unit_normalized": "abstain",
                        "near_miss": False,
                        "numeric_output": None,
                    }
                    action = "legacy_regex_false_positive_replayed_decline"
                else:
                    row["output"] = replayed
                    row["reason"] = ""
                    score = _score_output(replayed, row.get("gold"), question)
                    action = "legacy_regex_false_positive_replayed"
        elif row.get("reason") or row.get("output") is None:
            score = {
                "verdict": "abstain",
                "verdict_unit_normalized": "abstain",
                "near_miss": False,
                "numeric_output": None,
            }
            action = "abstention_unchanged"
        else:
            score = _score_output(row.get("output"), row.get("gold"), question)
            if score["verdict"] != row.get("verdict"):
                if isinstance(row.get("gold"), bool):
                    action = "boolean_scorer_correction"
                else:
                    action = "unified_vq_scorer_correction"

        row.update(score)
        row["audit_action"] = action
        row["deny_detail"] = detail
        audited.append(row)
    return audited


def vq_labels(rows: Sequence[Mapping[str, Any]], k: int) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    for row in rows:
        framework = row["framework_guided"]
        if k == 3:
            committed = (
                framework.get("final_output_value") is not None
                or framework.get("final_status") == "success"
            )
            is_correct = framework.get("final_is_correct")
        elif k == 1:
            diagnostic = framework["history"][0]["diagnostic"]
            committed = "is_correct" in diagnostic
            is_correct = diagnostic.get("is_correct")
        else:
            raise ValueError("Only K=1 and K=3 are present in the canonical run")
        if not committed:
            label = "abstain"
        else:
            label = "correct" if bool(is_correct) else "silent_wrong"
        labels[str(row["case_id"])] = label
    return labels


def pal_labels(
    rows: Sequence[Mapping[str, Any]],
    *,
    correct_scorer: bool,
    questions_by_id: Mapping[str, Mapping[str, Any]],
) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    for row in rows:
        label = str(row.get("verdict", "abstain"))
        if correct_scorer and label != "abstain":
            case_id = str(row["case_id"])
            question = str(questions_by_id[case_id].get("question", ""))
            label = _score_output(row.get("output"), row.get("gold"), question)[
                "verdict_unit_normalized"
            ]
        labels[str(row["case_id"])] = label
    return labels


def paired_bootstrap(
    left: Sequence[str],
    right: Sequence[str],
    target: str,
    *,
    iterations: int = 10_000,
    seed: int = 20260729,
) -> List[float]:
    rng = random.Random(seed)
    n = len(left)
    diffs: List[float] = []
    for _ in range(iterations):
        total = 0
        for _ in range(n):
            index = rng.randrange(n)
            total += int(left[index] == target) - int(right[index] == target)
        diffs.append(total / n)
    diffs.sort()
    return [diffs[int(iterations * 0.025)], diffs[int(iterations * 0.975) - 1]]


def paired_comparison(
    primary: Mapping[str, str], comparator: Mapping[str, str], case_order: Sequence[str]
) -> Dict[str, Any]:
    if set(primary) != set(comparator) or set(primary) != set(case_order):
        raise ValueError("Paired comparison requires identical case IDs")
    left = [primary[case_id] for case_id in case_order]
    right = [comparator[case_id] for case_id in case_order]

    def contrast(target: str) -> Dict[str, Any]:
        both = sum(a == target and b == target for a, b in zip(left, right))
        primary_only = sum(a == target and b != target for a, b in zip(left, right))
        comparator_only = sum(a != target and b == target for a, b in zip(left, right))
        neither = len(left) - both - primary_only - comparator_only
        discordant = primary_only + comparator_only
        p_value = (
            float(binomtest(min(primary_only, comparator_only), discordant, 0.5).pvalue)
            if discordant
            else 1.0
        )
        return {
            "both": both,
            "primary_only": primary_only,
            "comparator_only": comparator_only,
            "neither": neither,
            "rate_difference": (primary_only - comparator_only) / len(left),
            "paired_bootstrap_difference_95": paired_bootstrap(left, right, target),
            "mcnemar_exact_p": p_value,
        }

    return {"correct": contrast("correct"), "silent_wrong": contrast("silent_wrong")}


def group_metrics(rows: Sequence[Mapping[str, Any]], key: str) -> Dict[str, Any]:
    groups: Dict[str, List[str]] = defaultdict(list)
    for row in rows:
        value = row.get(key)
        if key == "retrieval_hit_rank":
            value = "hit" if value is not None else "miss"
        groups[str(value)].append(str(row["verdict_unit_normalized"]))
    return {group: with_intervals(labels) for group, labels in sorted(groups.items())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rpal", type=Path, default=DEFAULT_RPAL)
    parser.add_argument("--vq", type=Path, default=DEFAULT_VQ)
    parser.add_argument("--pal", type=Path, default=DEFAULT_PAL)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RPAL.parent)
    args = parser.parse_args()

    raw_rpal = load_jsonl(args.rpal)
    questions_by_id = {
        str(row["question_id"]): row for row in load_jsonl(args.questions)
    }
    audited = audit_rpal(raw_rpal, questions_by_id)
    vq_rows = load_jsonl(args.vq)
    pal_rows = load_jsonl(args.pal)
    case_order = [str(row["case_id"]) for row in audited]
    if len(case_order) != 250 or len(set(case_order)) != len(case_order):
        raise ValueError("Expected 250 unique R-PAL cases")

    rpal = {str(row["case_id"]): str(row["verdict_unit_normalized"]) for row in audited}
    systems = {
        "retrieval_pal_top4_audited_unified_vq_scorer": rpal,
        "vq_k3": vq_labels(vq_rows, 3),
        "vq_k1": vq_labels(vq_rows, 1),
        "plain_pal_legacy_reported": pal_labels(
            pal_rows,
            correct_scorer=False,
            questions_by_id=questions_by_id,
        ),
        "plain_pal_unified_vq_scorer": pal_labels(
            pal_rows,
            correct_scorer=True,
            questions_by_id=questions_by_id,
        ),
    }

    action_counts = Counter(str(row["audit_action"]) for row in audited)
    changed = [
        {
            "case_id": row["case_id"],
            "audit_action": row["audit_action"],
            "raw_verdict": next(
                source["verdict"] for source in raw_rpal if source["case_id"] == row["case_id"]
            ),
            "audited_verdict": row["verdict_unit_normalized"],
            "output": row.get("output"),
            "gold": row.get("gold"),
            "deny_detail": row.get("deny_detail"),
        }
        for row in audited
        if row["audit_action"] not in {"rescored_unchanged", "abstention_unchanged"}
    ]

    audited_summary = {
        "protocol": "matched_retrieval_pal_v1_posthoc_audit_unified_vq_scorer",
        "source_raw_file": str(args.rpal.resolve()),
        "raw_metrics": with_intervals([str(row["verdict_unit_normalized"]) for row in raw_rpal]),
        "audited_metrics": with_intervals([rpal[case_id] for case_id in case_order]),
        "by_tier": group_metrics(audited, "level"),
        "by_retrieval": group_metrics(audited, "retrieval_hit_rank"),
        "retrieval": {
            "hit": sum(row.get("retrieval_hit_rank") is not None for row in audited),
            "miss": sum(row.get("retrieval_hit_rank") is None for row in audited),
            "recall_at_4": sum(row.get("retrieval_hit_rank") is not None for row in audited)
            / len(audited),
            "rank_counts": dict(
                sorted(
                    Counter(
                        str(row["retrieval_hit_rank"])
                        for row in audited
                        if row.get("retrieval_hit_rank") is not None
                    ).items()
                )
            ),
        },
        "audit_action_counts": dict(sorted(action_counts.items())),
        "changed_cases": changed,
        "residual_silent_wrong": [
            {
                "case_id": row["case_id"],
                "level": row.get("level"),
                "output": row.get("output"),
                "gold": row.get("gold"),
                "retrieval_hit_rank": row.get("retrieval_hit_rank"),
                "target_function_id": row.get("target_function_id"),
            }
            for row in audited
            if row["verdict_unit_normalized"] == "silent_wrong"
        ],
        "residual_abstain": [
            {
                "case_id": row["case_id"],
                "reason": row.get("reason"),
                "deny_detail": row.get("deny_detail"),
                "retrieval_hit_rank": row.get("retrieval_hit_rank"),
            }
            for row in audited
            if row["verdict_unit_normalized"] == "abstain"
        ],
    }

    comparison = {
        "primary": "retrieval_pal_top4_audited_unified_vq_scorer",
        "system_metrics": {
            name: with_intervals([labels[case_id] for case_id in case_order])
            for name, labels in systems.items()
        },
        "paired_vs_primary": {
            name: paired_comparison(rpal, labels, case_order)
            for name, labels in systems.items()
            if name != "retrieval_pal_top4_audited_unified_vq_scorer"
        },
        "interpretation_guardrail": (
            "The unified scorer uses VQ's task-aware rounding, tolerances, scale handling, and "
            "boolean-aware parsing. Plain PAL's four legacy denylist abstentions cannot be replay-"
            "audited because that runner did not persist generated code."
        ),
    }

    stem = args.rpal.stem
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / f"{stem}_audited.jsonl", audited)
    write_json(args.out_dir / f"{stem}_audited_summary.json", audited_summary)
    write_json(args.out_dir / f"{stem}_comparison.json", comparison)
    print(json.dumps({"audit": audited_summary, "comparison": comparison}, indent=2))


if __name__ == "__main__":
    main()
