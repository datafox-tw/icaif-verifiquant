#!/usr/bin/env python3
"""Aggregate repeated governance runs without treating repeats as independent cases."""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyze_governance_eval import (  # noqa: E402
    clean_labels_from_rpal,
    load_jsonl,
)

BASE = ROOT / "results/governance_eval"
DEFAULT_COMPARISONS = [BASE / f"gov_run{i}_comparison.json" for i in range(1, 4)]
DEFAULT_CLEAN_RUNS = [BASE / f"gov_run{i}_prompted_clean.jsonl" for i in range(1, 4)]
DEFAULT_QUESTIONS = ROOT / "results/paper_v1/questions_50.jsonl"
DEFAULT_TRAPS = ROOT / "results/paper_v1/trap/trap_set.jsonl"
DEFAULT_OUT = BASE / "governance_multirun_summary.json"
SYSTEMS = ("vq_k1", "rpal_plain", "rpal_prompted")
METRICS = ("fnar", "structured_catch_rate", "diagnostic_accuracy", "action_accuracy")
CLEAN_METRICS = ("accuracy", "swr", "coverage", "selective_accuracy")


def load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def distribution(values: Iterable[float]) -> Dict[str, Any]:
    data = [float(value) for value in values]
    return {
        "values": data,
        "mean": statistics.fmean(data),
        "min": min(data),
        "max": max(data),
        "range": max(data) - min(data),
    }


def metric_summary(reports: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for system in SYSTEMS:
        output[system] = {
            metric: distribution(report["trap_scores"][system]["overall"][metric] for report in reports)
            for metric in METRICS
        }
    return output


def clean_metric_summary(reports: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for system in SYSTEMS:
        output[system] = {
            metric: distribution(report["clean_scores"][system][metric] for report in reports)
            for metric in CLEAN_METRICS
        }
    return output


def detail_maps(
    reports: Sequence[Mapping[str, Any]], system: str
) -> List[Dict[str, Mapping[str, Any]]]:
    return [
        {str(row["case_id"]): row for row in report["trap_scores"][system]["details"]}
        for report in reports
    ]


def agreement(reports: Sequence[Mapping[str, Any]], system: str) -> Dict[str, Any]:
    maps = detail_maps(reports, system)
    ids = sorted(maps[0])
    keys = ("unsafe_numeric", "got_diagnostic", "diagnostic_correct", "action_correct")
    result: Dict[str, Any] = {"n_cases": len(ids), "n_runs": len(maps)}
    for key in keys:
        stable = sum(len({mapping[case_id][key] for mapping in maps}) == 1 for case_id in ids)
        result[f"all_run_{key}_agreement"] = stable / len(ids)
        result[f"all_run_{key}_agreement_n"] = stable
    unsafe_votes = Counter(
        sum(bool(mapping[case_id]["unsafe_numeric"]) for mapping in maps) for case_id in ids
    )
    result["unsafe_vote_histogram"] = {str(votes): unsafe_votes[votes] for votes in range(len(maps) + 1)}
    return result


def percentile(sorted_values: Sequence[float], probability: float) -> float:
    return sorted_values[round(probability * (len(sorted_values) - 1))]


def clustered_bootstrap_difference(
    reports: Sequence[Mapping[str, Any]],
    comparator: str,
    *,
    iterations: int,
    seed: int,
    case_to_cluster: Mapping[str, str] | None = None,
    operators: set[str] | None = None,
) -> Dict[str, Any]:
    """Bootstrap clusters; preserve repeats and related perturbations together."""
    vq = detail_maps(reports, "vq_k1")[0]
    comparison_runs = detail_maps(reports, comparator)
    ids = sorted(
        case_id
        for case_id, row in vq.items()
        if operators is None or str(row["operator"]) in operators
    )
    cluster_for = case_to_cluster or {case_id: case_id for case_id in ids}
    by_cluster: Dict[str, List[str]] = {}
    for case_id in ids:
        by_cluster.setdefault(cluster_for.get(case_id, case_id), []).append(case_id)
    observed = statistics.fmean(
        statistics.fmean(bool(run[case_id]["unsafe_numeric"]) for run in comparison_runs)
        - bool(vq[case_id]["unsafe_numeric"])
        for case_id in ids
    )
    rng = random.Random(seed)
    cluster_ids = sorted(by_cluster)
    samples: List[float] = []
    for _ in range(iterations):
        sampled_clusters = [rng.choice(cluster_ids) for _ in cluster_ids]
        sampled_ids = [
            case_id
            for cluster_id in sampled_clusters
            for case_id in by_cluster[cluster_id]
        ]
        samples.append(
            statistics.fmean(
                statistics.fmean(bool(run[case_id]["unsafe_numeric"]) for run in comparison_runs)
                - bool(vq[case_id]["unsafe_numeric"])
                for case_id in sampled_ids
            )
        )
    samples.sort()
    return {
        "estimand": "mean comparator FNAR minus VQ FNAR",
        "difference": observed,
        "bootstrap_95_ci": [percentile(samples, 0.025), percentile(samples, 0.975)],
        "n_cases": len(ids),
        "n_clusters": len(cluster_ids),
        "operators": sorted(operators) if operators is not None else "all",
        "iterations": iterations,
        "seed": seed,
        "note": (
            "The 3 repeats and all selected perturbations from one base question stay together."
            if case_to_cluster is not None
            else "The 3 repeats for a case stay together; case identifiers define clusters."
        ),
    }


def subset_summary(
    reports: Sequence[Mapping[str, Any]], operators: set[str]
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for system in SYSTEMS:
        run_values: Dict[str, List[float]] = {metric: [] for metric in METRICS}
        for report in reports:
            rows = [
                row
                for row in report["trap_scores"][system]["details"]
                if str(row["operator"]) in operators
            ]
            n = len(rows)
            run_values["fnar"].append(sum(row["unsafe_numeric"] for row in rows) / n)
            run_values["structured_catch_rate"].append(sum(row["structured_catch"] for row in rows) / n)
            run_values["diagnostic_accuracy"].append(sum(row["diagnostic_correct"] for row in rows) / n)
            run_values["action_accuracy"].append(sum(row["action_correct"] for row in rows) / n)
        result[system] = {metric: distribution(values) for metric, values in run_values.items()}
    return result


def operator_summary(reports: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    operators = sorted(reports[0]["trap_scores"]["vq_k1"]["per_operator"])
    return {
        operator: {
            system: {
                metric: distribution(
                    report["trap_scores"][system]["per_operator"][operator][metric]
                    for report in reports
                )
                for metric in METRICS
            }
            for system in SYSTEMS
        }
        for operator in operators
    }


def clean_agreement(clean_paths: Sequence[Path], questions_path: Path) -> Dict[str, Any]:
    questions = {str(row["question_id"]): row for row in load_jsonl(questions_path)}
    runs = [
        clean_labels_from_rpal(load_jsonl(path), questions, use_stored_verdict=False)
        for path in clean_paths
    ]
    ids = sorted(questions)
    stable = sum(len({run[case_id] for run in runs}) == 1 for case_id in ids)
    vote_patterns = Counter("/".join(run[case_id] for run in runs) for case_id in ids)
    return {
        "n_cases": len(ids),
        "n_runs": len(runs),
        "all_run_label_agreement": stable / len(ids),
        "all_run_label_agreement_n": stable,
        "label_patterns": dict(sorted(vote_patterns.items())),
    }


def final_verdict(reports: Sequence[Mapping[str, Any]], trap_metrics: Mapping[str, Any]) -> Dict[str, Any]:
    verdicts = [str(report["go_no_go"]["verdict"]) for report in reports]
    vq_fnar = trap_metrics["vq_k1"]["fnar"]["mean"]
    prompted_fnar = trap_metrics["rpal_prompted"]["fnar"]["mean"]
    vq_diag = trap_metrics["vq_k1"]["diagnostic_accuracy"]["mean"]
    prompted_diag = trap_metrics["rpal_prompted"]["diagnostic_accuracy"]["mean"]
    vq_action = trap_metrics["vq_k1"]["action_accuracy"]["mean"]
    prompted_action = trap_metrics["rpal_prompted"]["action_accuracy"]["mean"]
    keep = (
        all(value == "KEEP_EXECUTED_GOVERNANCE_CLAIM" for value in verdicts)
        and vq_fnar + 0.10 <= prompted_fnar
        and vq_diag >= prompted_diag
        and vq_action >= prompted_action
    )
    return {
        "verdict": "KEEP_NARROW_EXECUTED_GOVERNANCE_CLAIM" if keep else "DO_NOT_KEEP_BROAD_CLAIM",
        "all_run_verdicts": verdicts,
        "reason": (
            "All three repeats meet the frozen >=10-point FNAR-margin rule, while VQ retains higher exact diagnostic and action accuracy."
            if keep
            else "The repeated results do not consistently satisfy the frozen governance rule."
        ),
        "scope_limit": "Do not claim that VQ is the stronger clean calculation engine or that it solves open-ended M intent discovery.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparisons", nargs="+", type=Path, default=DEFAULT_COMPARISONS)
    parser.add_argument("--prompted-clean-runs", nargs="+", type=Path, default=DEFAULT_CLEAN_RUNS)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--traps", type=Path, default=DEFAULT_TRAPS)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    reports = [load_json(path) for path in args.comparisons]
    traps = load_jsonl(args.traps)
    case_to_base = {
        str(row["trap_id"]): str(row.get("base_question_id") or row["trap_id"])
        for row in traps
    }
    if len(reports) < 2:
        raise ValueError("At least two repeats are required for a stability analysis")
    if len(args.prompted_clean_runs) != len(reports):
        raise ValueError("Provide one prompted clean run per comparison")

    trap_metrics = metric_summary(reports)
    payload = {
        "protocol": "governance_matched_eval_multirun_v1",
        "n_unique_traps": len(reports[0]["trap_scores"]["vq_k1"]["details"]),
        "n_repeats": len(reports),
        "external_request_accounting": {
            "per_repeat": {"plain_traps": 50, "prompted_traps": 50, "prompted_clean": 50},
            "total": len(reports) * 150,
        },
        "trap_metrics": trap_metrics,
        "clean_metrics": clean_metric_summary(reports),
        "contract_grounded_subset_F_E_I_N": subset_summary(reports, {"F", "E", "I", "N"}),
        "per_operator": operator_summary(reports),
        "repeat_agreement": {
            "rpal_plain": agreement(reports, "rpal_plain"),
            "rpal_prompted": agreement(reports, "rpal_prompted"),
            "rpal_prompted_clean": clean_agreement(args.prompted_clean_runs, args.questions),
        },
        "clustered_bootstrap": {
            comparator: clustered_bootstrap_difference(
                reports, comparator, iterations=args.bootstrap_iterations, seed=args.seed
            )
            for comparator in ("rpal_plain", "rpal_prompted")
        },
        "contract_grounded_clustered_bootstrap_F_E_I_N": {
            comparator: clustered_bootstrap_difference(
                reports,
                comparator,
                iterations=args.bootstrap_iterations,
                seed=args.seed,
                case_to_cluster=case_to_base,
                operators={"F", "E", "I", "N"},
            )
            for comparator in ("rpal_plain", "rpal_prompted")
        },
        "per_run_paired_fnar": [report["paired_fnar"] for report in reports],
        "interpretation_guardrails": [
            "The repeated Gemini calls are summarized by mean/range and case-clustered bootstrap; 150 repeated trap outputs are not treated as 150 independent questions.",
            "VQ K=1 is a frozen existing run, not three newly repeated VQ calls.",
            "Plain R-PAL clean utility is the same frozen baseline in every comparison; only prompted R-PAL clean was repeated.",
            "The deterministic 50-case trap set is balanced by F/E/I/N/M, but I has 9 I_SOFT and 1 I_HARD.",
        ],
    }
    payload["final_go_no_go"] = final_verdict(reports, trap_metrics)
    write_json(args.out, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
