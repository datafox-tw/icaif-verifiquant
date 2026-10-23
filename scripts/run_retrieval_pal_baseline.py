#!/usr/bin/env python3
"""Matched retrieval + PAL/PoT execution baseline for the 250Q paper set.

Protocol
--------
1. Retrieve top-k candidates with the exact SQLAlchemyArtifactStore query used
   by VerifiQuant: ``question + "\\n" + context``.
2. Resolve each candidate's ``function_id`` against FinanceReasoning's public
   3,133-function library (never against the per-question worked solution).
3. Give candidate titles, summaries, and source functions to one Gemini call.
4. Ask the model for one self-contained ``solve()`` function and execute it
   with the same sandbox, timeout, and deny policy as the plain PAL run.
5. Score the output with VerifiQuant's canonical task-aware evaluator.

The baseline receives retrieval plus deterministic execution, but no FIC input
schema, invariants, semantic hints, gates, DiagnosticReport, or repair rules.

Usage:
  PYTHONPATH=. GEMINI_API_KEY=... .venv/bin/python \
      scripts/run_retrieval_pal_baseline.py --dry-run

  PYTHONPATH=. GEMINI_API_KEY=... .venv/bin/python \
      scripts/run_retrieval_pal_baseline.py --model gemini-2.5-flash --top-k 4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_pal_baseline import _num, deny_reason, extract_code, run_code  # noqa: E402
from verifiquant.card_store import SQLAlchemyArtifactStore  # noqa: E402
from verifiquant.pipeline.run_error_classification_pipeline import (  # noqa: E402
    _answer_match,
    create_genai_client_from_env,
)


DEFAULT_QUESTIONS = ROOT / "results/paper_v2_250/questions_250.jsonl"
DEFAULT_DB = ROOT / "results/paper_v2_250/fic/cards.db"
DEFAULT_FUNCTIONS = ROOT / "FinanceReasoning/data/functions/functions-article-all.json"
DEFAULT_OUT_DIR = ROOT / "results/retrieval_pal_baseline"

PROMPT = """You are a financial reasoning assistant that solves problems by writing Python.

You are given top-ranked reference functions retrieved from a financial function library.
The candidates are retrieval results, not guaranteed to be relevant or correct for the question.
Select the useful candidate, if any, and adapt its logic to the exact question and context.

Write a single self-contained Python function solve() that computes the numeric answer.
Rules:
- Only the Python standard library `math` may be imported; no other imports.
- No input(), no file or network access.
- solve() must return a single number (int or float).
- Match the output unit and scale requested by the question.
- If no candidate applies or the problem cannot be solved from the given information,
  return the string "CANNOT SOLVE".
- Do not merely call a reference function: solve() must be self-contained.
- Return ONLY the code, in one ```python block.

Question:
{question}

Context:
{context}

Retrieved reference functions (ranked):
{references}
"""


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _load_function_library(path: Path) -> Dict[str, Dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, list):
        raise ValueError(f"Function library must be a JSON list: {path}")
    out: Dict[str, Dict[str, Any]] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        function_id = str(row.get("function_id", "")).strip()
        code = str(row.get("function", "")).strip()
        if function_id and code:
            out[function_id] = row
    if not out:
        raise ValueError(f"No valid functions loaded from {path}")
    return out


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _safe_model_name(model: str) -> str:
    return model.replace(".", "").replace("-", "_").replace("/", "_")


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def _percentile(values: Sequence[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(int(round((len(ordered) - 1) * fraction)), len(ordered) - 1)
    return ordered[index]


def _candidate_payloads(
    *,
    store: SQLAlchemyArtifactStore,
    core_by_id: Mapping[str, Dict[str, Any]],
    function_by_id: Mapping[str, Dict[str, Any]],
    question: str,
    context: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    rows = store.retrieve_candidates(query=f"{question}\n{context}", top_k=top_k)
    out: List[Dict[str, Any]] = []
    for rank, row in enumerate(rows, 1):
        fic_id = str(row.get("fic_id", "")).strip()
        source_meta = (core_by_id.get(fic_id, {}).get("source_meta") or {})
        function_id = str(source_meta.get("function_id", "")).strip()
        source_function = function_by_id.get(function_id)
        if not function_id or source_function is None:
            raise RuntimeError(
                f"Retrieved candidate {fic_id!r} has no resolvable public function_id "
                f"in the configured function library"
            )
        code = str(source_function.get("function", "")).strip()
        out.append(
            {
                "rank": rank,
                "fic_id": fic_id,
                "function_id": function_id,
                "score": float(row.get("score", 0.0) or 0.0),
                "title": str(row.get("title") or source_function.get("article_title") or ""),
                "summary": str(row.get("summary", "")),
                "function": code,
                "function_sha256": _sha256_text(code),
            }
        )
    return out


def _render_references(candidates: Sequence[Mapping[str, Any]]) -> str:
    blocks: List[str] = []
    for candidate in candidates:
        blocks.append(
            "\n".join(
                [
                    f"### Candidate {candidate['rank']}: {candidate['title']}",
                    f"Summary: {candidate['summary']}",
                    "```python",
                    str(candidate["function"]),
                    "```",
                ]
            )
        )
    return "\n\n".join(blocks) if blocks else "(no candidates retrieved)"


def _prompt_for(row: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> str:
    return PROMPT.format(
        question=str(row.get("question", "")),
        context=str(row.get("context", "") or "(none)"),
        references=_render_references(candidates),
    )


def _score_output(output: Any, gold: Any, question: str) -> Dict[str, Any]:
    """Score an R-PAL output with VQ's canonical task-aware evaluator."""
    answer = _num(output)
    target = _num(gold)
    abs_error, is_correct, scale_note = _answer_match(question, answer, target)
    verdict = "correct" if is_correct is True else "silent_wrong"
    return {
        "verdict": verdict,
        "verdict_unit_normalized": verdict,
        "near_miss": scale_note == "output_scale_x100",
        "numeric_output": answer,
        "abs_error": abs_error,
        "scale_note": scale_note,
    }


def _generate_with_retries(client: Any, *, model: str, prompt: str, max_retries: int) -> tuple[str, str]:
    last_error = ""
    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            return str(response.text or ""), ""
        except Exception as exc:
            last_error = f"api:{type(exc).__name__}:{exc}"[:300]
            if attempt < max_retries:
                time.sleep(min(2 ** attempt, 8))
    return "", last_error


def _public_candidate_log(candidates: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    keys = ("rank", "fic_id", "function_id", "score", "title", "function_sha256")
    return [{key: candidate.get(key) for key in keys} for candidate in candidates]


def _rates(records: Sequence[Mapping[str, Any]], verdict_key: str) -> Dict[str, Any]:
    counts = Counter(str(row.get(verdict_key, "abstain")) for row in records)
    n = len(records)
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


def _summarize(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_n: int,
    model: str,
    top_k: int,
    elapsed_s: float,
) -> Dict[str, Any]:
    tier_rows: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    retrieval_rows: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        tier_rows[str(row.get("level", "unknown"))].append(row)
        retrieval_rows["hit" if row.get("retrieval_hit_rank") else "miss"].append(row)
    retrieval_ranks = Counter(
        int(row["retrieval_hit_rank"])
        for row in records
        if row.get("retrieval_hit_rank") is not None
    )
    return {
        "protocol": "matched_retrieval_pal_v2",
        "model": model,
        "top_k": top_k,
        "expected_n": expected_n,
        "processed_n": len(records),
        "complete": len(records) == expected_n,
        "raw": _rates(records, "verdict"),
        "unit_normalized": _rates(records, "verdict_unit_normalized"),
        "near_miss": sum(bool(row.get("near_miss")) for row in records),
        "code_fail": sum(str(row.get("reason", "")).startswith(("exec_fail", "no_code")) for row in records),
        "denied": sum(row.get("reason") == "denylist" for row in records),
        "api_fail": sum(str(row.get("reason", "")).startswith("api:") for row in records),
        "retrieval": {
            "hit": sum(retrieval_ranks.values()),
            "miss": len(records) - sum(retrieval_ranks.values()),
            "recall_at_k": sum(retrieval_ranks.values()) / len(records) if records else 0.0,
            "rank_counts": {str(rank): retrieval_ranks[rank] for rank in range(1, top_k + 1)},
        },
        "by_tier_unit_normalized": {
            tier: _rates(rows, "verdict_unit_normalized") for tier, rows in sorted(tier_rows.items())
        },
        "by_retrieval_unit_normalized": {
            key: _rates(rows, "verdict_unit_normalized") for key, rows in sorted(retrieval_rows.items())
        },
        "elapsed_s": round(elapsed_s, 1),
        "ran_at": datetime.now().isoformat(timespec="seconds"),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _dry_run_audit(
    *,
    questions: Sequence[Mapping[str, Any]],
    store: SQLAlchemyArtifactStore,
    core_by_id: Mapping[str, Dict[str, Any]],
    function_by_id: Mapping[str, Dict[str, Any]],
    top_k: int,
) -> Dict[str, Any]:
    prompt_lengths: List[int] = []
    rank_counts: Counter[int] = Counter()
    for row in questions:
        candidates = _candidate_payloads(
            store=store,
            core_by_id=core_by_id,
            function_by_id=function_by_id,
            question=str(row.get("question", "")),
            context=str(row.get("context", "") or ""),
            top_k=top_k,
        )
        prompt = _prompt_for(row, candidates)
        prompt_lengths.append(len(prompt))
        function_ids = [candidate["function_id"] for candidate in candidates]
        target_function_id = str(row.get("function_id", ""))
        if target_function_id in function_ids:
            rank_counts[function_ids.index(target_function_id) + 1] += 1
        if "python_solution" in prompt or str(row.get("ground_truth")) in prompt.split("Question:", 1)[0]:
            raise AssertionError("Prompt construction leaked a forbidden per-question artifact")
    hits = sum(rank_counts.values())
    return {
        "protocol": "matched_retrieval_pal_v2",
        "n": len(questions),
        "top_k": top_k,
        "function_library_size": len(function_by_id),
        "retrieval_hit": hits,
        "retrieval_miss": len(questions) - hits,
        "retrieval_recall_at_k": hits / len(questions) if questions else 0.0,
        "rank_counts": {str(rank): rank_counts[rank] for rank in range(1, top_k + 1)},
        "prompt_chars": {
            "min": min(prompt_lengths) if prompt_lengths else 0,
            "median": int(statistics.median(prompt_lengths)) if prompt_lengths else 0,
            "p95": _percentile(prompt_lengths, 0.95),
            "max": max(prompt_lengths) if prompt_lengths else 0,
        },
        "forbidden_inputs": ["python_solution", "ground_truth", "repair_rules", "diagnostic_checks"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--function-library", type=Path, default=DEFAULT_FUNCTIONS)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be >= 1")
    if args.resume and args.force:
        parser.error("--resume and --force are mutually exclusive")

    questions = _load_jsonl(args.questions)
    if args.limit:
        questions = questions[: args.limit]
    function_by_id = _load_function_library(args.function_library)
    store = SQLAlchemyArtifactStore(f"sqlite:///{args.db.resolve()}")
    core_by_id = store.load_core_by_id()

    missing_library_functions = sorted(
        {
            str((core.get("source_meta") or {}).get("function_id", ""))
            for core in core_by_id.values()
        }
        - set(function_by_id)
    )
    missing_library_functions = [item for item in missing_library_functions if item]
    if missing_library_functions:
        raise RuntimeError(
            f"{len(missing_library_functions)} card functions are absent from the public function library; "
            f"first={missing_library_functions[:3]}"
        )

    if args.dry_run:
        print(
            json.dumps(
                _dry_run_audit(
                    questions=questions,
                    store=store,
                    core_by_id=core_by_id,
                    function_by_id=function_by_id,
                    top_k=args.top_k,
                ),
                indent=2,
            )
        )
        return

    run_name = args.run_name or f"rpal_v2_top{args.top_k}_{_safe_model_name(args.model)}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.out_dir / f"{run_name}.jsonl"
    summary_path = args.out_dir / f"{run_name}_summary.json"
    manifest_path = args.out_dir / f"{run_name}_manifest.json"

    if output_path.exists() and not args.resume and not args.force:
        raise FileExistsError(
            f"Refusing to overwrite {output_path}; use --resume or explicit --force"
        )
    if args.force:
        for path in (output_path, summary_path, manifest_path):
            if path.exists():
                path.unlink()

    completed: set[str] = set()
    records: List[Dict[str, Any]] = []
    if args.resume and output_path.exists():
        records = _load_jsonl(output_path)
        completed = {str(row.get("case_id", "")) for row in records}

    manifest = {
        "protocol": "matched_retrieval_pal_v2",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "model": args.model,
        "generation_config": "SDK default, matching scripts/run_pal_baseline.py",
        "top_k": args.top_k,
        "retrieval_query": "question + newline + context",
        "retriever": "SQLAlchemyArtifactStore.retrieve_candidates (identical to canonical VQ)",
        "candidate_content": ["retrieval title", "retrieval summary", "public source function"],
        "excluded_fic_content": [
            "typed input schema",
            "invariants and scale checks",
            "semantic hints",
            "repair rules",
            "gate decisions",
        ],
        "questions": {"path": str(args.questions.resolve()), "sha256": _sha256_file(args.questions)},
        "card_db": {
            "path": str(args.db.resolve()),
            "sha256": _sha256_file(args.db),
            "core_card_count": len(core_by_id),
        },
        "function_library": {
            "path": str(args.function_library.resolve()),
            "sha256": _sha256_file(args.function_library),
            "function_count": len(function_by_id),
        },
        "prompt_template": PROMPT,
        "execution": {
            "runner": "same extract_code/run_code/AST deny policy as scripts/run_pal_baseline.py",
            "timeout_seconds": 10,
            "allowed_imports": ["math"],
        },
        "scoring": {
            "implementation": "verifiquant.pipeline.run_error_classification_pipeline._answer_match",
            "numeric_tolerance": "math.isclose(answer, gold, rel_tol=0.01, abs_tol=0.01)",
            "nearest_integer": "round(answer) == round(gold) when requested by the question",
            "unit_normalization": ["answer * 100"],
            "ternary_labels": ["correct", "silent_wrong", "abstain"],
        },
        "forbidden_per_question_inputs": ["python_solution", "ground_truth in prompt"],
    }
    _write_json(manifest_path, manifest)

    client = create_genai_client_from_env()
    t0 = time.time()
    mode = "a" if args.resume and output_path.exists() else "w"
    with output_path.open(mode, encoding="utf-8") as output_file:
        for index, row in enumerate(questions, 1):
            case_id = str(row.get("question_id") or row.get("case_id") or "")
            if case_id in completed:
                print(f"[{run_name} {index}/{len(questions)}] {case_id} skip(resume)", flush=True)
                continue

            case_t0 = time.time()
            candidates = _candidate_payloads(
                store=store,
                core_by_id=core_by_id,
                function_by_id=function_by_id,
                question=str(row.get("question", "")),
                context=str(row.get("context", "") or ""),
                top_k=args.top_k,
            )
            prompt = _prompt_for(row, candidates)
            candidate_function_ids = [candidate["function_id"] for candidate in candidates]
            target_function_id = str(row.get("function_id", ""))
            retrieval_hit_rank = (
                candidate_function_ids.index(target_function_id) + 1
                if target_function_id in candidate_function_ids
                else None
            )

            response_text, reason = _generate_with_retries(
                client,
                model=args.model,
                prompt=prompt,
                max_retries=args.max_retries,
            )
            code = extract_code(response_text)
            output_value: Any = None
            denied = ""
            if reason:
                score = {
                    "verdict": "abstain",
                    "verdict_unit_normalized": "abstain",
                    "near_miss": False,
                    "numeric_output": None,
                }
            elif code is None:
                reason = "no_code"
                score = {
                    "verdict": "abstain",
                    "verdict_unit_normalized": "abstain",
                    "near_miss": False,
                    "numeric_output": None,
                }
            else:
                denied = deny_reason(code)
                if denied:
                    reason = "denylist"
                    score = {
                        "verdict": "abstain",
                        "verdict_unit_normalized": "abstain",
                        "near_miss": False,
                        "numeric_output": None,
                    }
                else:
                    output_value = run_code(code)
                    if output_value is None:
                        reason = "exec_fail"
                        score = {
                            "verdict": "abstain",
                            "verdict_unit_normalized": "abstain",
                            "near_miss": False,
                            "numeric_output": None,
                        }
                    elif "CANNOT SOLVE" in str(output_value).upper():
                        reason = "declined"
                        score = {
                            "verdict": "abstain",
                            "verdict_unit_normalized": "abstain",
                            "near_miss": False,
                            "numeric_output": None,
                        }
                    else:
                        score = _score_output(
                            output_value,
                            row.get("ground_truth"),
                            str(row.get("question", "")),
                        )

            record = {
                "case_id": case_id,
                "level": row.get("level"),
                "gold": row.get("ground_truth"),
                "target_function_id": target_function_id,
                "retrieval_hit_rank": retrieval_hit_rank,
                "candidates": _public_candidate_log(candidates),
                "prompt_sha256": _sha256_text(prompt),
                "prompt_chars": len(prompt),
                "response_text": response_text,
                "generated_code": code,
                "output": output_value,
                "reason": reason,
                "deny_detail": denied,
                "elapsed_s": round(time.time() - case_t0, 3),
                **score,
            }
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_file.flush()
            records.append(record)
            print(
                f"[{run_name} {index}/{len(questions)}] {case_id} "
                f"hit={retrieval_hit_rank or '-'} {record['verdict_unit_normalized']} {reason}",
                flush=True,
            )

    summary = _summarize(
        records,
        expected_n=len(questions),
        model=args.model,
        top_k=args.top_k,
        elapsed_s=time.time() - t0,
    )
    _write_json(summary_path, summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
