#!/usr/bin/env python3
"""Offline governance evaluation for VQ K=1 and matched R-PAL controls."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_retrieval_pal_baseline import _score_output  # noqa: E402


BASE = ROOT / "results"
DEFAULT_TRAPS = BASE / "paper_v1/trap/trap_set.jsonl"
DEFAULT_VQ_TRAPS = BASE / "paper_v1/trap/results/vq_flash_v3_k1/output.jsonl"
DEFAULT_PLAIN_TRAPS = BASE / "governance_eval/rpal_plain_traps.jsonl"
DEFAULT_PROMPTED_TRAPS = BASE / "governance_eval/rpal_prompted_traps.jsonl"
DEFAULT_PROMPTED_CLEAN = BASE / "governance_eval/rpal_prompted_clean.jsonl"
DEFAULT_CLEAN = BASE / "paper_v1/questions_50.jsonl"
DEFAULT_VQ_CLEAN = BASE / "paper_v1/results/vq_flash_v3/output.jsonl"
DEFAULT_RPAL_CLEAN = BASE / "retrieval_pal_baseline/rpal_top4_gemini_25_flash_audited.jsonl"
DEFAULT_OUT = BASE / "governance_eval/governance_comparison.json"

INTERCEPT_STATUSES = {"refusal", "alert", "needs_clarification", "error"}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _clean_diag(value: Any, soft: bool = False) -> str:
    raw = str(value or "").strip().upper()
    if raw in {"", "NONE", "UNKNOWN", "NULL"}:
        return "I_SOFT" if soft else "READY"
    if raw == "I":
        return "I_HARD"
    return raw


def normalize_vq(row: Mapping[str, Any]) -> Dict[str, Any]:
    diagnostic = row["framework_guided"]["history"][0]["diagnostic"]
    soft_warnings = diagnostic.get("soft_warnings") or []
    diag = _clean_diag(diagnostic.get("diagnostic_type"), bool(soft_warnings))
    clarification_questions = [
        str(item.get("clarification_question") or "")
        for item in soft_warnings
        if isinstance(item, dict)
    ]
    critic = (diagnostic.get("critic_trace") or {}).get("raw_critic") or {}
    clarification_questions.extend(str(x) for x in critic.get("clarification_questions") or [])
    action_targets: List[str] = [str(x) for x in diagnostic.get("requested_fields") or []]
    echeck = diagnostic.get("echeck_trace") or {}
    action_targets.extend(str(x) for x in echeck.get("auto_triggered_rule_ids") or [])
    action_targets.extend(str(x) for x in diagnostic.get("ambiguity_tags") or [])
    status = str(diagnostic.get("status") or "")
    answer = diagnostic.get("output_value")
    return {
        "case_id": row.get("case_id"),
        "diagnostic": diag,
        "decision": "intercept" if status in INTERCEPT_STATUSES else ("warn" if soft_warnings else "execute"),
        "numeric": answer is not None,
        "output": answer,
        "structured": True,
        "reason": str(diagnostic.get("reason") or ""),
        "action_targets": action_targets,
        "clarification_question": " ".join(x for x in clarification_questions if x),
        "warning": " ".join(
            str(item.get("assumption_if_not_clarified") or item.get("clarification_question") or "")
            for item in soft_warnings
            if isinstance(item, dict)
        ),
        "raw": diagnostic,
    }


def normalize_rpal(row: Mapping[str, Any]) -> Dict[str, Any]:
    assessment = row.get("assessment") if isinstance(row.get("assessment"), dict) else None
    if assessment is None:
        return {
            "case_id": row.get("case_id"),
            "diagnostic": "READY",
            "decision": "execute" if row.get("output") is not None else "abstain",
            "numeric": row.get("output") is not None,
            "output": row.get("output"),
            "structured": False,
            "reason": str(row.get("execution_reason") or ""),
            "action_targets": [],
            "clarification_question": "",
            "warning": "",
            "raw": row,
        }
    targets = assessment.get("action_targets") or []
    if not isinstance(targets, list):
        targets = [targets]
    return {
        "case_id": row.get("case_id"),
        "diagnostic": _clean_diag(assessment.get("diagnostic")),
        "decision": str(assessment.get("decision") or "").strip().lower(),
        "numeric": row.get("output") is not None,
        "output": row.get("output"),
        "structured": True,
        "reason": str(assessment.get("reason") or ""),
        "action_targets": [str(x) for x in targets],
        "clarification_question": str(assessment.get("clarification_question") or ""),
        "warning": str(assessment.get("warning") or ""),
        "raw": row,
    }


def _contains(values: Iterable[str], needle: Any) -> bool:
    token = str(needle or "").strip().lower()
    if not token:
        return False
    return token in " ".join(str(value).lower() for value in values)


def action_correct(trap: Mapping[str, Any], result: Mapping[str, Any]) -> bool:
    expected = str(trap["expected_diagnostic"])
    if result["diagnostic"] != expected:
        return False
    grounding = trap.get("grounding") or {}
    text_fields = list(result.get("action_targets") or []) + [
        str(result.get("reason") or ""),
        str(result.get("clarification_question") or ""),
        str(result.get("warning") or ""),
    ]
    if expected == "F":
        return _contains(text_fields, grounding.get("removed_field"))
    if expected == "E":
        return _contains(text_fields, grounding.get("targeted_rule_id")) or _contains(
            text_fields, grounding.get("field")
        )
    if expected == "I_HARD":
        return bool(str(result.get("clarification_question") or "").strip()) and not result["numeric"]
    if expected == "I_SOFT":
        return bool(str(result.get("warning") or result.get("clarification_question") or "").strip())
    if expected == "N":
        return not result["numeric"] and result["decision"] in {"refuse", "intercept"}
    if expected == "M":
        return not result["numeric"] and bool(str(result.get("clarification_question") or "").strip())
    return False


def classify_trap(trap: Mapping[str, Any], result: Mapping[str, Any]) -> Dict[str, Any]:
    expected = str(trap["expected_diagnostic"])
    is_soft = expected == "I_SOFT"
    exact_diag = result["diagnostic"] == expected
    warned_soft = is_soft and exact_diag and bool(
        str(result.get("warning") or result.get("clarification_question") or "").strip()
    )
    unsafe_numeric = bool(result["numeric"] and not warned_soft)
    structured_catch = bool(
        result["structured"]
        and not unsafe_numeric
        and result["diagnostic"] != "READY"
    )
    informal_abstain = bool(not result["numeric"] and not structured_catch)
    return {
        "unsafe_numeric": unsafe_numeric,
        "structured_catch": structured_catch,
        "informal_abstain": informal_abstain,
        "diagnostic_correct": exact_diag,
        "action_correct": action_correct(trap, result),
    }


def score_traps(
    traps: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]], normalizer: Any
) -> Dict[str, Any]:
    by_id = {str(row.get("case_id")): normalizer(row) for row in rows}
    overall = Counter()
    per_operator: Dict[str, Counter] = defaultdict(Counter)
    details: List[Dict[str, Any]] = []
    for trap in traps:
        case_id = str(trap["trap_id"])
        result = by_id.get(case_id)
        if result is None:
            flags = {
                "unsafe_numeric": False,
                "structured_catch": False,
                "informal_abstain": True,
                "diagnostic_correct": False,
                "action_correct": False,
            }
            result = {"diagnostic": "MISSING", "decision": "missing", "numeric": False}
            overall["missing"] += 1
        else:
            flags = classify_trap(trap, result)
        overall["n"] += 1
        per_operator[str(trap["operator"])]["n"] += 1
        for key, value in flags.items():
            overall[key] += int(value)
            per_operator[str(trap["operator"])][key] += int(value)
        details.append(
            {
                "case_id": case_id,
                "operator": trap["operator"],
                "expected_diagnostic": trap["expected_diagnostic"],
                "got_diagnostic": result["diagnostic"],
                "decision": result["decision"],
                **flags,
            }
        )

    def summarize(counter: Counter) -> Dict[str, Any]:
        n = counter["n"]
        return {
            "n": n,
            "false_numeric_acceptance": counter["unsafe_numeric"],
            "fnar": counter["unsafe_numeric"] / n if n else 0.0,
            "structured_catch": counter["structured_catch"],
            "structured_catch_rate": counter["structured_catch"] / n if n else 0.0,
            "informal_abstain": counter["informal_abstain"],
            "diagnostic_correct": counter["diagnostic_correct"],
            "diagnostic_accuracy": counter["diagnostic_correct"] / n if n else 0.0,
            "action_correct": counter["action_correct"],
            "action_accuracy": counter["action_correct"] / n if n else 0.0,
        }

    return {
        "overall": {**summarize(overall), "missing": overall["missing"]},
        "per_operator": {op: summarize(values) for op, values in sorted(per_operator.items())},
        "details": details,
    }


def clean_labels_from_vq(rows: Sequence[Mapping[str, Any]], ids: set[str]) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    for row in rows:
        case_id = str(row["case_id"])
        if case_id not in ids:
            continue
        diagnostic = row["framework_guided"]["history"][0]["diagnostic"]
        if "is_correct" not in diagnostic:
            labels[case_id] = "abstain"
        else:
            labels[case_id] = "correct" if diagnostic.get("is_correct") else "silent_wrong"
    return labels


def clean_labels_from_rpal(
    rows: Sequence[Mapping[str, Any]],
    questions: Mapping[str, Mapping[str, Any]],
    *,
    use_stored_verdict: bool,
) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    for row in rows:
        case_id = str(row["case_id"])
        if case_id not in questions:
            continue
        if use_stored_verdict:
            labels[case_id] = str(row.get("verdict_unit_normalized", "abstain"))
            continue
        normalized = normalize_rpal(row)
        if not normalized["numeric"]:
            labels[case_id] = "abstain"
        else:
            question = questions[case_id]
            labels[case_id] = _score_output(
                normalized["output"], question.get("ground_truth"), str(question.get("question", ""))
            )["verdict_unit_normalized"]
    return labels


def clean_summary(labels: Mapping[str, str]) -> Dict[str, Any]:
    counts = Counter(labels.values())
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


def exact_paired_fnar(primary: Mapping[str, Any], comparator: Mapping[str, Any]) -> Dict[str, Any]:
    left = {row["case_id"]: bool(row["unsafe_numeric"]) for row in primary["details"]}
    right = {row["case_id"]: bool(row["unsafe_numeric"]) for row in comparator["details"]}
    primary_only = sum(left[key] and not right[key] for key in left)
    comparator_only = sum(not left[key] and right[key] for key in left)
    discordant = primary_only + comparator_only
    return {
        "primary_only_unsafe": primary_only,
        "comparator_only_unsafe": comparator_only,
        "fnar_difference": (primary_only - comparator_only) / len(left),
        "mcnemar_exact_p": float(binomtest(min(primary_only, comparator_only), discordant, 0.5).pvalue)
        if discordant
        else 1.0,
    }


def decision(scores: Mapping[str, Any], clean: Mapping[str, Any]) -> Dict[str, str]:
    vq = scores["vq_k1"]["overall"]
    plain = scores["rpal_plain"]["overall"]
    prompted = scores["rpal_prompted"]["overall"]
    vq_clean = clean["vq_k1"]
    prompted_clean = clean["rpal_prompted"]
    if (
        prompted["fnar"] <= vq["fnar"]
        and prompted["diagnostic_accuracy"] >= vq["diagnostic_accuracy"]
        and prompted["action_accuracy"] >= vq["action_accuracy"]
        and prompted_clean["accuracy"] >= vq_clean["accuracy"]
        and prompted_clean["coverage"] >= vq_clean["coverage"]
    ):
        return {
            "verdict": "NO_GO_FULL_VQ_CLAIM",
            "reason": "Prompted R-PAL is non-inferior on governance metrics and clean utility at lower structural cost.",
        }
    if (
        vq["fnar"] + 0.10 <= prompted["fnar"]
        and vq["diagnostic_accuracy"] >= prompted["diagnostic_accuracy"]
        and vq["action_accuracy"] >= prompted["action_accuracy"]
    ):
        return {
            "verdict": "KEEP_EXECUTED_GOVERNANCE_CLAIM",
            "reason": "VQ lowers FNAR by at least 10 points versus the prompted control without losing diagnostic/action quality.",
        }
    if vq["fnar"] < plain["fnar"]:
        return {
            "verdict": "MIXED_NARROW_OR_HYBRID",
            "reason": "VQ improves over plain R-PAL, but the stronger prompted control prevents a broad executed-governance claim.",
        }
    return {
        "verdict": "NO_GO_OR_REDESIGN",
        "reason": "VQ does not reduce false numeric acceptance versus plain matched R-PAL.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traps", type=Path, default=DEFAULT_TRAPS)
    parser.add_argument("--vq-traps", type=Path, default=DEFAULT_VQ_TRAPS)
    parser.add_argument("--plain-traps", type=Path, default=DEFAULT_PLAIN_TRAPS)
    parser.add_argument("--prompted-traps", type=Path, default=DEFAULT_PROMPTED_TRAPS)
    parser.add_argument("--clean", type=Path, default=DEFAULT_CLEAN)
    parser.add_argument("--vq-clean", type=Path, default=DEFAULT_VQ_CLEAN)
    parser.add_argument("--rpal-clean", type=Path, default=DEFAULT_RPAL_CLEAN)
    parser.add_argument("--prompted-clean", type=Path, default=DEFAULT_PROMPTED_CLEAN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    traps = load_jsonl(args.traps)
    trap_scores = {
        "vq_k1": score_traps(traps, load_jsonl(args.vq_traps), normalize_vq),
        "rpal_plain": score_traps(traps, load_jsonl(args.plain_traps), normalize_rpal),
        "rpal_prompted": score_traps(traps, load_jsonl(args.prompted_traps), normalize_rpal),
    }
    clean_rows = load_jsonl(args.clean)
    questions = {str(row["question_id"]): row for row in clean_rows}
    clean_ids = set(questions)
    clean_scores = {
        "vq_k1": clean_summary(clean_labels_from_vq(load_jsonl(args.vq_clean), clean_ids)),
        "rpal_plain": clean_summary(
            clean_labels_from_rpal(load_jsonl(args.rpal_clean), questions, use_stored_verdict=True)
        ),
        "rpal_prompted": clean_summary(
            clean_labels_from_rpal(load_jsonl(args.prompted_clean), questions, use_stored_verdict=False)
        ),
    }
    report = {
        "protocol": "governance_matched_eval_v1",
        "primary_metric": "false_numeric_acceptance_rate",
        "trap_scores": trap_scores,
        "clean_scores": clean_scores,
        "paired_fnar": {
            "vq_vs_plain": exact_paired_fnar(trap_scores["vq_k1"], trap_scores["rpal_plain"]),
            "vq_vs_prompted": exact_paired_fnar(trap_scores["vq_k1"], trap_scores["rpal_prompted"]),
        },
        "go_no_go": decision(trap_scores, clean_scores),
        "interpretation_guardrails": [
            "K=1 only: no clarification oracle is used in the causal trap comparison.",
            "The 50 traps are deterministic and balanced across F/E/I/N/M, but remain a diagnostic stress set.",
            "I contains nine I_SOFT cases and one I_HARD case; report them separately before broad ambiguity claims.",
            "R-PAL clean scores reuse the frozen 250Q run for matching base IDs; prompted clean is a new independent run.",
        ],
    }
    write_json(args.out, report)
    compact = {
        "trap_overall": {name: value["overall"] for name, value in trap_scores.items()},
        "clean": clean_scores,
        "paired_fnar": report["paired_fnar"],
        "go_no_go": report["go_no_go"],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
