from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover
    genai = None
    genai_types = None


def _load_records(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    raise ValueError(f"Unsupported input format: {path}")


def _dump_records(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".jsonl":
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().replace(",", "").replace("_", "")
    if not s:
        return None
    is_pct = "%" in s
    s = s.replace("%", "")
    m = re.search(r"-?\d+(\.\d+)?", s)
    if not m:
        return None
    n = float(m.group())
    if is_pct:
        return n / 100.0
    return n


def _answer_match(question: str, output_value: Optional[float], gold_num: Optional[float]) -> Tuple[Optional[float], Optional[bool]]:
    if output_value is None or gold_num is None:
        return None, None
    q = question.lower()
    round_mode = any(x in q for x in ["nearest integer", "nearest whole", "round to nearest", "四捨五入", "整數"])
    if round_mode:
        lhs = float(round(output_value))
        rhs = float(round(gold_num))
        return abs(lhs - rhs), lhs == rhs
    abs_err = abs(output_value - gold_num)
    return abs_err, math.isclose(output_value, gold_num, rel_tol=1e-6, abs_tol=1e-2)


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


def _schema_cot_step() -> Any:
    return genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "answer": genai_types.Schema(type=genai_types.Type.STRING),
            "needs_more_info": genai_types.Schema(type=genai_types.Type.BOOLEAN),
            "missing_information": genai_types.Schema(
                type=genai_types.Type.ARRAY,
                items=genai_types.Schema(type=genai_types.Type.STRING),
            ),
            "confidence": genai_types.Schema(type=genai_types.Type.NUMBER),
            "revised_question": genai_types.Schema(type=genai_types.Type.STRING),
            "revised_context": genai_types.Schema(type=genai_types.Type.STRING),
            "reasoning_note": genai_types.Schema(type=genai_types.Type.STRING),
        },
        required=[
            "answer",
            "needs_more_info",
            "missing_information",
            "confidence",
            "revised_question",
            "revised_context",
            "reasoning_note",
        ],
    )


def _llm_json(
    client: Any,
    *,
    model: str,
    prompt: str,
    schema: Any,
    fallback: Optional[Dict[str, Any]] = None,
    max_attempts: int = 3,
) -> Dict[str, Any]:
    config = genai_types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=schema,
    )
    last_err: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            # Gemini can return an empty candidate (resp.text is None) on
            # safety blocks or transient server issues; treat as retryable.
            if resp.text is None:
                raise ValueError("empty response text from model")
            return json.loads(resp.text)
        except Exception as exc:  # noqa: BLE001 — retry any transient failure
            last_err = exc
            print(f"  ⚠ _llm_json attempt {attempt}/{max_attempts} failed: {exc}")
            if attempt < max_attempts:
                time.sleep(2 * attempt)
    if fallback is not None:
        print(f"  ⚠ _llm_json exhausted retries; using fallback ({last_err})")
        return dict(fallback)
    raise RuntimeError(f"_llm_json failed after {max_attempts} attempts: {last_err}")


_ORACLE_REWRITE_FALLBACK: Dict[str, Any] = {
    # No-op rewrite: callers use `rewrite["updated_question"] or <previous>`,
    # so empty strings degrade to "leave question/context unchanged".
    "updated_question": "",
    "updated_context": "",
    "notes": "llm_empty_response_fallback",
}

_COT_STEP_FALLBACK: Dict[str, Any] = {
    # Recorded as a no-answer turn rather than crashing the whole run.
    "answer": "",
    "needs_more_info": True,
    "missing_information": ["llm_empty_response_fallback"],
    "confidence": 0.0,
    "revised_question": "",
    "revised_context": "",
    "reasoning_note": "llm_empty_response_fallback",
}


def _gold_value(row: Dict[str, Any]) -> Any:
    return row.get("gold_answer", row.get("answer", row.get("ground_truth")))


def _cot_step(
    *,
    client: Any,
    model: str,
    question: str,
    context: str,
) -> Dict[str, Any]:
    prompt = _COT_PROMPT_TEMPLATE.format(question=question, context=context)
    return _llm_json(
        client, model=model, prompt=prompt,
        schema=_schema_cot_step(), fallback=_COT_STEP_FALLBACK,
    )


def _oracle_rewrite_for_cot(
    *,
    client: Any,
    model: str,
    row: Dict[str, Any],
    current_question: str,
    current_context: str,
    cot_step_output: Dict[str, Any],
) -> Dict[str, str]:
    prompt = _ORACLE_PROMPT_TEMPLATE.format(
        current_question=current_question,
        current_context=current_context,
        cot_step_output_json=json.dumps(cot_step_output, ensure_ascii=False, indent=2),
        ground_truth_code=row.get("code", row.get("python_solution", "")),
    )
    out = _llm_json(
        client, model=model, prompt=prompt,
        schema=_schema_oracle_rewrite(), fallback=_ORACLE_REWRITE_FALLBACK,
    )
    return {
        "updated_question": str(out.get("updated_question", "") or current_question).strip() or current_question,
        "updated_context": str(out.get("updated_context", "") or current_context).strip() or current_context,
        "notes": str(out.get("notes", "")).strip(),
    }


def _oracle_rewrite_for_cot_funnel_guided(
    *,
    client: Any,
    model: str,
    row: Dict[str, Any],
    current_question: str,
    current_context: str,
    cot_step_output: Dict[str, Any],
) -> Dict[str, str]:
    prompt = _ORACLE_PROMPT_TEMPLATE_FUNNEL_GUIDED.format(
        current_question=current_question,
        current_context=current_context,
        cot_step_output_json=json.dumps(cot_step_output, ensure_ascii=False, indent=2),
        ground_truth_code=row.get("code", row.get("python_solution", "")),
    )
    out = _llm_json(
        client, model=model, prompt=prompt,
        schema=_schema_oracle_rewrite(), fallback=_ORACLE_REWRITE_FALLBACK,
    )
    return {
        "updated_question": str(out.get("updated_question", "") or current_question).strip() or current_question,
        "updated_context": str(out.get("updated_context", "") or current_context).strip() or current_context,
        "notes": str(out.get("notes", "")).strip(),
    }


def _cot_oracle_loop(
    *,
    client: Any,
    cot_model: str,
    oracle_model: str,
    row: Dict[str, Any],
    max_turns: int,
    oracle_mode: str,
) -> Dict[str, Any]:
    question = str(row.get("question", "") or "")
    context = str(row.get("context", "") or "")
    gold = _gold_value(row)
    gold_num = _parse_number(gold)
    history: List[Dict[str, Any]] = []
    final_answer: Optional[float] = None
    final_correct: Optional[bool] = None
    final_abs_error: Optional[float] = None

    for turn in range(1, max_turns + 1):
        step = _cot_step(
            client=client,
            model=cot_model,
            question=question,
            context=context,
        )
        ans = _parse_number(step.get("answer"))
        abs_err, is_correct = _answer_match(question, ans, gold_num)
        final_answer = ans
        final_correct = is_correct
        final_abs_error = abs_err

        history.append(
            {
                "turn": turn,
                "question": question,
                "context": context,
                "cot_step": step,
                "parsed_answer": ans,
                "is_correct": is_correct,
                "abs_error": abs_err,
            }
        )

        print(f"  \u21b3 CoT Turn {turn}: is_correct={is_correct}")

        # Option A (blind review): the oracle reviews EVERY turn regardless of
        # correctness. We intentionally do NOT gate iteration on is_correct, because
        # gating on ground truth would (a) leak "this answer is wrong" through the
        # oracle's entry itself, and (b) only ever rescue wrong answers while never
        # risking a correct one, biasing accuracy upward. Natural termination is:
        # max_turns reached, or the oracle returns question/context unchanged (below).
        if turn >= max_turns:
            break

        if oracle_mode == "funnel-guided":
            rewrite = _oracle_rewrite_for_cot_funnel_guided(
                client=client,
                model=oracle_model,
                row=row,
                current_question=question,
                current_context=context,
                cot_step_output=step,
            )
        else:
            rewrite = _oracle_rewrite_for_cot(
                client=client,
                model=oracle_model,
                row=row,
                current_question=question,
                current_context=context,
                cot_step_output=step,
            )
        history[-1]["oracle_rewrite"] = rewrite
        new_q = rewrite["updated_question"] or str(step.get("revised_question", "")).strip() or question
        new_c = rewrite["updated_context"] or str(step.get("revised_context", "")).strip() or context
        if new_q == question and new_c == context:
            break
        question, context = new_q, new_c

    return {
        "rounds": len(history),
        "final_answer": final_answer,
        "final_is_correct": final_correct,
        "final_abs_error": final_abs_error,
        "history": history,
    }


_COT_PROMPT_TEMPLATE = """\
You are a financial CoT solver with self-improvement hints.
Solve using the current question/context and return a tentative answer.
If information is missing/ambiguous, set needs_more_info=true and list missing items.
IMPORTANT: `answer` must be a numeric string only (for example: "10185.19"), or empty string if you cannot provide one.

Question:
{question}

Context:
{context}

Return JSON only."""

_ORACLE_PROMPT_TEMPLATE = """\
You are Oracle-in-the-loop support for a pure CoT baseline.
There is no VerifiQuant framework in this loop.
You can ONLY use the logic within the ground-truth code to clarify assumptions and revise user question/context. You must NOT rely on the final numeric ground truth result.

Current question:
{current_question}

Current context:
{current_context}

Current CoT step output:
{cot_step_output_json}

Ground-truth code:
{ground_truth_code}

Return JSON only."""


_ORACLE_PROMPT_TEMPLATE_FUNNEL_GUIDED = """\
You are Oracle-in-the-loop support for a CoT baseline with structured diagnostic guidance.
There is no VerifiQuant execution framework in this loop; you are only providing an oracle rewrite for the CoT baseline.
You can ONLY use the logic within the ground-truth code to clarify assumptions and revise user question/context. You must NOT rely on the final numeric ground truth result.

When reviewing the CoT step output for potential errors or ambiguities, systematically check the following diagnostic layers (M/N/F/E/I/C):

1. M (Intent/Meaning): Does the user's question map unambiguously to a single calculation or formula? Are there multiple valid interpretations of what is being asked?
   - Example: "profit on total contract" could mean profit as a percentage of incurred costs or a fixed fee on total contract value.
   - If ambiguous, explicitly list the alternative interpretations.

2. N (Scope/Domain): Does the question fall within a well-defined financial domain, such as valuation, cash flow, or bond pricing? Or does it require expertise outside standard finance?
   - If out-of-scope, note which aspect is unclear or non-standard.

3. F (Fields/Schema): Are all required input fields explicitly provided or inferable from context?
   - Check the ground-truth code for required parameters. If any are missing, list them.
   - Example: CAGR requires start_value, end_value, and years. Are all given?

4. E (Execution/Boundary): Do the provided input values satisfy reasonable boundary constraints, such as interest rates > -1 and prices > 0? Would the computation hit numerical errors such as division by zero?
   - If boundary violations exist, flag them explicitly.

5. I (Interpretation/Ambiguity): Are there hidden semantic conventions or scale conventions that the CoT might have misinterpreted?
   - I_HARD examples that change computation: period-start vs period-end, decimal (0.05) vs percentage (5%), annualized vs monthly.
   - I_SOFT examples about representation: returning 0.25 vs 25% for a ratio, or rounding precision.
   - If CoT chose a convention without declaring it, note the alternative.

6. C (Code Logic): Is the Python code logic consistent with standard financial practice? Does it correctly implement the stated formula?
   - Use the provided ground-truth code as the reference.

Current question:
{current_question}

Current context:
{current_context}

Current CoT step output:
{cot_step_output_json}

Ground-truth code (for reference; do NOT use final numeric result):
{ground_truth_code}

Instructions:
1. Work through the six diagnostic layers in priority order: M > N > F > E > I > C.
2. Identify the highest-priority issue that should be clarified or corrected.
3. Revise the question/context only enough to address that issue while preserving the original task.
4. In notes, clearly mark the relevant layer, such as [M], [F], [I_HARD], [I_SOFT], or [C].
5. If no issue is found, leave the question/context unchanged and explain the checked layer(s) briefly.

Return JSON only."""


def _aggregate(
    rows: List[Dict[str, Any]],
    *,
    cot_model: str,
    oracle_model: str,
    max_turns: int,
    oracle_mode: str,
) -> Dict[str, Any]:
    total = len(rows)
    correct = sum(1 for r in rows if r["cot_self_improve"].get("final_is_correct") is True)
    improved = 0
    mode = "single_shot"
    if max_turns > 1:
        mode = "cot_basic_oracle" if oracle_mode == "plain" else "cot_funnel_guided_oracle"
    broken = 0
    for r in rows:
        h = r["cot_self_improve"].get("history", [])
        final_ok = r["cot_self_improve"].get("final_is_correct") is True
        if len(h) >= 2 and h[0].get("is_correct") is False and final_ok:
            improved += 1
        # Blind review can also turn an initially-correct answer wrong.
        if len(h) >= 2 and h[0].get("is_correct") is True and not final_ok:
            broken += 1
    return {
        "total_cases": total,
        "correct_count": correct,
        "accuracy": (correct / total) if total else 0.0,
        "improved_count": improved,
        "improved_rate": (improved / total) if total else 0.0,
        "broken_count": broken,
        "broken_rate": (broken / total) if total else 0.0,
        # ── run_config: prompt templates + model metadata for comparison ──
        "run_config": {
            "cot_model": cot_model,
            "oracle_model": oracle_model,
            "max_turns": max_turns,
            "mode": mode,
            "oracle_mode": oracle_mode,
            "oracle_entry_policy": "blind_review_every_turn",
            "cot_prompt_template": _COT_PROMPT_TEMPLATE,
            "oracle_prompt_template": (
                _ORACLE_PROMPT_TEMPLATE_FUNNEL_GUIDED
                if max_turns > 1 and oracle_mode == "funnel-guided"
                else _ORACLE_PROMPT_TEMPLATE
                if max_turns > 1
                else None
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pure CoT self-improvement pipeline (CoT + Oracle rewrite, no VerifiQuant framework feedback)."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input JSON/JSONL with question/context/code/result")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON/JSONL records")
    parser.add_argument("--summary-output", type=Path, help="Optional summary JSON output")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--cot-model", default="gemini-2.5-flash")
    parser.add_argument("--oracle-model", default="gemini-2.5-flash")
    parser.add_argument(
        "--oracle-mode",
        choices=["plain", "funnel-guided"],
        default="plain",
        help="Oracle guidance mode for iterative CoT correction.",
    )
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

    out_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows, 1):
        cid = row.get("case_id") or row.get("question_id") or row.get("id") or idx
        print(f"[cot-self-improve] processing {cid}")
        cot = _cot_oracle_loop(
            client=client,
            cot_model=args.cot_model,
            oracle_model=args.oracle_model,
            row=row,
            max_turns=max(1, args.max_turns),
            oracle_mode=args.oracle_mode,
        )
        out_rows.append(
            {
                "case_id": cid,
                "source_sample_id": row.get("source_sample_id") or row.get("question_id") or row.get("id"),
                "cot_self_improve": cot,
            }
        )

    _dump_records(args.output, out_rows)
    summary = _aggregate(
        out_rows,
        cot_model=args.cot_model,
        oracle_model=args.oracle_model,
        max_turns=max(1, args.max_turns),
        oracle_mode=args.oracle_mode,
    )
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(out_rows)} cot self-improve records to {args.output}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
