from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover
    genai = None
    genai_types = None

from verifiquant.pipeline.run_error_classification_pipeline import (
    ErrorClassificationAPI,
    _dump_records,
    _load_records,
)


def _schema_oracle_rewrite() -> Any:
    return genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "updated_question": genai_types.Schema(type=genai_types.Type.STRING),
            "updated_context": genai_types.Schema(type=genai_types.Type.STRING),
            "notes": genai_types.Schema(type=genai_types.Type.STRING),
        },
        required=["updated_question", "updated_context", "notes"],
    )


def _llm_json(client: Any, *, model: str, prompt: str, schema: Any) -> Dict[str, Any]:
    config = genai_types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=schema,
    )
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    return json.loads(resp.text)


def _gold_value(row: Dict[str, Any]) -> Any:
    return row.get("gold_answer", row.get("answer", row.get("ground_truth")))


def _oracle_rewrite_for_framework(
    *,
    client: Any,
    model: str,
    row: Dict[str, Any],
    current_question: str,
    current_context: str,
    diagnostic: Dict[str, Any],
) -> Dict[str, str]:
    prompt = f"""
You are an Oracle-in-the-loop rewriter for VerifiQuant framework-guided iteration.
You can ONLY use the logic within the ground-truth code to clarify assumptions and revise user question/context. You must NOT rely on the final numeric ground truth result.

Goal:
- Given framework diagnostic output, rewrite question/context so the next run can pass gates.
- Keep semantics faithful to original task and the logic implied by the provided code.
- Do NOT output the final numeric answer directly in question/context.

Current question:
{current_question}

Current context:
{current_context}

Framework diagnostic:
{json.dumps(diagnostic, ensure_ascii=False, indent=2)}

Ground-truth code:
{row.get("code", row.get("python_solution", ""))}

Return JSON only.
"""
    out = _llm_json(client, model=model, prompt=prompt, schema=_schema_oracle_rewrite())
    return {
        "updated_question": str(out.get("updated_question", "") or current_question).strip() or current_question,
        "updated_context": str(out.get("updated_context", "") or current_context).strip() or current_context,
        "notes": str(out.get("notes", "")).strip(),
    }


def _framework_oracle_loop(
    *,
    api: ErrorClassificationAPI,
    client: Any,
    oracle_model: str,
    row: Dict[str, Any],
    max_turns: int,
) -> Dict[str, Any]:
    question = str(row.get("question", "") or "")
    context = str(row.get("context", "") or "")
    history: List[Dict[str, Any]] = []
    final_diag: Dict[str, Any] = {}

    for turn in range(1, max_turns + 1):
        diag = api.diagnose_row(
            {
                **row,
                "question": question,
                "context": context,
                "case_id": row.get("case_id", row.get("question_id", f"case_{turn}")),
            }
        )
        final_diag = diag

        diag_type = str(diag.get("diagnostic_type", "")).strip().upper()
        status = diag.get("status")
        is_correct = diag.get("is_correct")
        has_i_soft = diag.get("has_i_soft", False)

        intercepted = False
        if status == "success":
            if has_i_soft and is_correct is False:
                # Intercept for iteration: computed but failed to match ground truth under I_soft
                diag_type = "I_SOFT_MISMATCH"
                status = "needs_iteration"
                diag["diagnostic_type"] = diag_type
                diag["status"] = status
                diag["oracle_hint"] = "System completed execution but final logic/result mismatch ground truth. Soft missing information hints were triggered. Please use ground-truth code logic to resolve the soft_warnings."
                intercepted = True

        recorded_error = diag_type if diag_type else status

        history.append(
            {
                "turn": turn,
                "question": question,
                "context": context,
                "diagnostic": diag,
                "recorded_error": recorded_error,
            }
        )

        print(f"  \u21b3 Turn {turn}: {recorded_error}")

        if status == "success" and not intercepted:
            break
        if turn >= max_turns:
            break
        if recorded_error not in {"M", "N", "F", "E", "I", "I_HARD", "I_SOFT", "I_SOFT_MISMATCH"}:
            break

        rewrite = _oracle_rewrite_for_framework(
            client=client,
            model=oracle_model,
            row=row,
            current_question=question,
            current_context=context,
            diagnostic=diag,
        )
        new_q = rewrite["updated_question"]
        new_c = rewrite["updated_context"]
        if new_q == question and new_c == context:
            break
        question, context = new_q, new_c

    return {
        "rounds": len(history),
        "final_status": final_diag.get("status"),
        "final_diagnostic_type": final_diag.get("diagnostic_type"),
        "final_output_value": final_diag.get("output_value"),
        "final_is_correct": final_diag.get("is_correct"),
        "final_abs_error": final_diag.get("abs_error"),
        "history": history,
    }


def _aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    success_count = sum(1 for r in rows if r["framework_guided"].get("final_status") == "success")
    correct_count = sum(1 for r in rows if r["framework_guided"].get("final_is_correct") is True)
    recovered = 0
    for r in rows:
        h = r["framework_guided"].get("history", [])
        if len(h) >= 2 and h[0].get("diagnostic", {}).get("status") != "success" and r["framework_guided"].get("final_status") == "success":
            recovered += 1
    return {
        "total_cases": total,
        "success_count": success_count,
        "success_rate": (success_count / total) if total else 0.0,
        "correct_count": correct_count,
        "accuracy": (correct_count / total) if total else 0.0,
        "recovery_count": recovered,
        "recovery_rate": (recovered / total) if total else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Framework-guided self-improvement pipeline (VerifiQuant output -> Oracle rewrite -> rerun)."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input JSON/JSONL with question/context/code/result")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON/JSONL records")
    parser.add_argument("--summary-output", type=Path, help="Optional summary JSON output")
    parser.add_argument("--db-url", help="If provided, load cards from SQLAlchemy store")
    parser.add_argument("--core", type=Path, help="fic_core JSON/JSONL when --db-url is not set")
    parser.add_argument("--retrieval", type=Path, help="fic_retrieval JSON/JSONL when --db-url is not set")
    parser.add_argument("--repair", type=Path, help="repair_rule JSON/JSONL when --db-url is not set")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--m-min-top-score", type=float, default=0.05)
    parser.add_argument("--selector-model", default="gemini-2.5-flash")
    parser.add_argument("--extractor-model", default="gemini-2.5-flash")
    parser.add_argument("--judge-model", default="gemini-2.5-flash")
    parser.add_argument("--oracle-model", default="gemini-2.5-flash")
    args = parser.parse_args()

    if genai is None or genai_types is None:
        print("Missing dependency: google.genai import failed.", file=sys.stderr)
        sys.exit(1)
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Missing GEMINI_API_KEY in environment.", file=sys.stderr)
        sys.exit(1)

    rows = _load_records(args.input)
    if args.max_records > 0:
        rows = rows[: args.max_records]
    client = genai.Client(api_key=api_key)

    if args.db_url:
        api = ErrorClassificationAPI.from_db(
            db_url=args.db_url,
            client=client,
            selector_model=args.selector_model,
            extractor_model=args.extractor_model,
            judge_model=args.judge_model,
            top_k=args.top_k,
            m_min_top_score=args.m_min_top_score,
        )
    else:
        if not (args.core and args.retrieval and args.repair):
            raise ValueError("When --db-url is not provided, --core --retrieval --repair are required.")
        api = ErrorClassificationAPI.from_files(
            core_path=args.core,
            retrieval_path=args.retrieval,
            repair_path=args.repair,
            client=client,
            selector_model=args.selector_model,
            extractor_model=args.extractor_model,
            judge_model=args.judge_model,
            top_k=args.top_k,
            m_min_top_score=args.m_min_top_score,
        )

    out_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows, 1):
        cid = row.get("case_id", row.get("question_id", idx))
        print(f"[framework-guided] processing {cid}")
        fw = _framework_oracle_loop(
            api=api,
            client=client,
            oracle_model=args.oracle_model,
            row=row,
            max_turns=max(1, args.max_turns),
        )
        out_rows.append(
            {
                "case_id": cid,
                "source_sample_id": row.get("source_sample_id", row.get("question_id")),
                "framework_guided": fw,
            }
        )

    _dump_records(args.output, out_rows)
    summary = _aggregate(out_rows)
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(out_rows)} framework-guided records to {args.output}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
