#!/usr/bin/env python3
"""Run matched R-PAL controls on clean or governance-critical inputs.

This runner deliberately saves raw decisions and execution artifacts only.
Final clean/trap labels are assigned offline by analyze_governance_eval.py.

Modes
-----
plain:
    The existing retrieval+PAL prompt. It may execute or informally abstain.
prompted:
    The same retrieval+PAL system, but explicitly prompted to classify
    M/N/F/E/I_HARD/I_SOFT before deciding whether to execute. No deterministic
    FIC gate is run; this is the strong "governance described in a prompt"
    control for the governance-by-execution claim.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_pal_baseline import deny_reason, extract_code, run_code  # noqa: E402
from run_retrieval_pal_baseline import (  # noqa: E402
    DEFAULT_FUNCTIONS,
    PROMPT,
    _candidate_payloads,
    _generate_with_retries,
    _git_commit,
    _load_function_library,
    _load_jsonl,
    _prompt_for,
    _public_candidate_log,
    _sha256_file,
    _sha256_text,
)
from verifiquant.card_store import SQLAlchemyArtifactStore  # noqa: E402
from verifiquant.pipeline.run_error_classification_pipeline import (  # noqa: E402
    create_genai_client_from_env,
)


DEFAULT_TRAPS = ROOT / "results/paper_v1/trap/trap_set.jsonl"
DEFAULT_CLEAN = ROOT / "results/paper_v1/questions_50.jsonl"
DEFAULT_DB = ROOT / "results/paper_v1/fic/cards_v3.db"
DEFAULT_OUT = ROOT / "results/governance_eval"

PROMPTED_TEMPLATE = """You are a financial reasoning assistant using retrieved reference functions.

Before calculating, decide whether the request is safe and sufficiently specified.
Use this taxonomy and priority order:
- M: the user's intent or requested metric is unclear; ask what they want to evaluate.
- N: the requested calculation is unsupported or the references are only superficially related; refuse.
- F: a required numeric/input field is absent or redacted; ask for the exact missing field.
- E: a provided value violates a financial, mathematical, or domain boundary; identify it and do not calculate.
- I_HARD: an unresolved unit, direction, timing, basis, or convention can change the answer; clarify before calculating.
- I_SOFT: the calculation can proceed, but an interpretation assumption must be disclosed as a warning.
- READY: the request has one clear intent, supported logic, complete inputs, admissible values, and no material ambiguity.

The retrieved candidates are not guaranteed to be applicable or safe. Do not force-fit one.
For READY or I_SOFT, write a self-contained Python solve() function. Only `math` may be imported.
For M/N/F/E/I_HARD, do not provide executable code or a numeric answer.

Return ONLY one valid JSON object with exactly these fields:
{{
  "decision": "execute|warn|clarify|refuse",
  "diagnostic": "READY|M|N|F|E|I_HARD|I_SOFT",
  "reason": "concise evidence-grounded explanation",
  "action_targets": ["exact missing field, invalid field/rule, or ambiguity target"],
  "clarification_question": "one actionable question or empty string",
  "warning": "explicit assumption/warning for I_SOFT or empty string",
  "code": "self-contained Python code as a JSON string, or empty string"
}}

Question:
{question}

Context:
{context}

Retrieved reference functions (ranked):
{references}
"""


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def _prompted_prompt(row: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> str:
    return PROMPTED_TEMPLATE.format(
        question=str(row.get("question", "")),
        context=str(row.get("context", "") or "(none)"),
        references=_render_references(candidates),
    )


def _extract_json_object(text: str) -> Dict[str, Any] | None:
    raw = (text or "").strip()
    if raw.startswith("```"):
        first_newline = raw.find("\n")
        last_fence = raw.rfind("```")
        if first_newline >= 0 and last_fence > first_newline:
            raw = raw[first_newline + 1 : last_fence].strip()
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        start = raw.find("{")
        if start < 0:
            return None
        try:
            value, _ = json.JSONDecoder().raw_decode(raw[start:])
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None


def _abstain_fields(reason: str) -> Dict[str, Any]:
    return {
        "generated_code": None,
        "output": None,
        "execution_reason": reason,
        "deny_detail": "",
    }


def _execute(code: str | None) -> Dict[str, Any]:
    if not code:
        return _abstain_fields("no_code")
    denied = deny_reason(code)
    if denied:
        result = _abstain_fields("denylist")
        result["generated_code"] = code
        result["deny_detail"] = denied
        return result
    output = run_code(code)
    if output is None:
        result = _abstain_fields("exec_fail")
        result["generated_code"] = code
        return result
    if "CANNOT SOLVE" in str(output).upper():
        result = _abstain_fields("declined")
        result["generated_code"] = code
        return result
    return {
        "generated_code": code,
        "output": output,
        "execution_reason": "",
        "deny_detail": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plain", "prompted"), required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_TRAPS)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--function-library", type=Path, default=DEFAULT_FUNCTIONS)
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--per-operator", type=int, default=0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = _load_jsonl(args.dataset)
    if args.per_operator:
        kept: List[Dict[str, Any]] = []
        operator_counts: Dict[str, int] = {}
        for row in rows:
            operator = str(row.get("operator") or "clean")
            if operator_counts.get(operator, 0) >= args.per_operator:
                continue
            kept.append(row)
            operator_counts[operator] = operator_counts.get(operator, 0) + 1
        rows = kept
    if args.limit:
        rows = rows[: args.limit]
    function_by_id = _load_function_library(args.function_library)
    store = SQLAlchemyArtifactStore(f"sqlite:///{args.db.resolve()}")
    core_by_id = store.load_core_by_id()

    if args.dry_run:
        hits = 0
        prompt_chars: List[int] = []
        for row in rows:
            candidates = _candidate_payloads(
                store=store,
                core_by_id=core_by_id,
                function_by_id=function_by_id,
                question=str(row.get("question", "")),
                context=str(row.get("context", "") or ""),
                top_k=args.top_k,
            )
            hits += str(row.get("function_id") or "") in {
                str(candidate["function_id"]) for candidate in candidates
            }
            prompt = _prompt_for(row, candidates) if args.mode == "plain" else _prompted_prompt(row, candidates)
            prompt_chars.append(len(prompt))
        print(
            json.dumps(
                {
                    "protocol": "governance_matched_rpal_v1",
                    "mode": args.mode,
                    "n": len(rows),
                    "top_k": args.top_k,
                    "target_retrieval_hit": hits,
                    "target_retrieval_recall": hits / len(rows) if rows else 0.0,
                    "prompt_chars_min": min(prompt_chars) if prompt_chars else 0,
                    "prompt_chars_max": max(prompt_chars) if prompt_chars else 0,
                    "api_calls": 0,
                },
                indent=2,
            )
        )
        return

    dataset_tag = "traps" if "trap" in args.dataset.name else "clean"
    run_name = args.run_name or f"rpal_{args.mode}_{dataset_tag}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.out_dir / f"{run_name}.jsonl"
    manifest_path = args.out_dir / f"{run_name}_manifest.json"
    summary_path = args.out_dir / f"{run_name}_summary.json"
    if output_path.exists() and not args.resume and not args.force:
        raise FileExistsError(f"Refusing to overwrite {output_path}; use --resume or --force")
    if args.force:
        for path in (output_path, manifest_path, summary_path):
            if path.exists():
                path.unlink()

    records: List[Dict[str, Any]] = []
    completed: set[str] = set()
    if args.resume and output_path.exists():
        records = _load_jsonl(output_path)
        completed = {str(row["case_id"]) for row in records}

    prompt_template = PROMPT if args.mode == "plain" else PROMPTED_TEMPLATE
    manifest = {
        "protocol": "governance_matched_rpal_v1",
        "mode": args.mode,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "model": args.model,
        "top_k": args.top_k,
        "dataset": {"path": str(args.dataset.resolve()), "sha256": _sha256_file(args.dataset)},
        "card_db": {"path": str(args.db.resolve()), "sha256": _sha256_file(args.db)},
        "function_library": {
            "path": str(args.function_library.resolve()),
            "sha256": _sha256_file(args.function_library),
            "size": len(function_by_id),
        },
        "retrieval_query": "question + newline + context",
        "prompt_template": prompt_template,
        "scoring": "raw artifacts only; offline scorer required",
        "forbidden_prompt_inputs": ["ground_truth", "python_solution", "operator", "expected_diagnostic"],
    }
    _write_json(manifest_path, manifest)

    client = create_genai_client_from_env()
    started = time.time()
    mode = "a" if args.resume and output_path.exists() else "w"
    with output_path.open(mode, encoding="utf-8") as fh:
        for index, row in enumerate(rows, 1):
            case_id = str(row.get("question_id") or row.get("trap_id") or row.get("case_id") or row.get("id"))
            if case_id in completed:
                print(f"[{run_name} {index}/{len(rows)}] {case_id} skip", flush=True)
                continue
            case_started = time.time()
            candidates = _candidate_payloads(
                store=store,
                core_by_id=core_by_id,
                function_by_id=function_by_id,
                question=str(row.get("question", "")),
                context=str(row.get("context", "") or ""),
                top_k=args.top_k,
            )
            prompt = _prompt_for(row, candidates) if args.mode == "plain" else _prompted_prompt(row, candidates)
            response_text, api_reason = _generate_with_retries(
                client,
                model=args.model,
                prompt=prompt,
                max_retries=args.max_retries,
            )

            assessment: Dict[str, Any] | None = None
            if api_reason:
                execution = _abstain_fields(api_reason)
            elif args.mode == "plain":
                execution = _execute(extract_code(response_text))
            else:
                assessment = _extract_json_object(response_text)
                if assessment is None:
                    execution = _abstain_fields("assessment_parse_fail")
                else:
                    decision = str(assessment.get("decision", "")).strip().lower()
                    if decision in {"execute", "warn"}:
                        code = str(assessment.get("code") or "").strip() or extract_code(response_text)
                        execution = _execute(code)
                    else:
                        execution = _abstain_fields("governance_intercept")

            record = {
                "case_id": case_id,
                "base_question_id": row.get("base_question_id") or case_id,
                "dataset_operator": row.get("operator"),
                "mode": args.mode,
                "candidates": _public_candidate_log(candidates),
                "prompt_sha256": _sha256_text(prompt),
                "prompt_chars": len(prompt),
                "response_text": response_text,
                "assessment": assessment,
                "elapsed_s": round(time.time() - case_started, 3),
                **execution,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            records.append(record)
            decision = (assessment or {}).get("diagnostic", "-")
            answered = "numeric" if record.get("output") is not None else record["execution_reason"]
            print(f"[{run_name} {index}/{len(rows)}] {case_id} diag={decision} {answered}", flush=True)

    counts: Dict[str, int] = {}
    for row in records:
        key = "numeric" if row.get("output") is not None else str(row.get("execution_reason") or "abstain")
        counts[key] = counts.get(key, 0) + 1
    summary = {
        "protocol": "governance_matched_rpal_v1",
        "run_name": run_name,
        "mode": args.mode,
        "n": len(records),
        "complete": len(records) == len(rows),
        "raw_outcomes": counts,
        "elapsed_s": round(time.time() - started, 1),
        "note": "Not a scored result; use scripts/analyze_governance_eval.py",
    }
    _write_json(summary_path, summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
