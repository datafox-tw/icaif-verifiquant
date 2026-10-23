#!/usr/bin/env python3
"""Staged R-PAL+ control for the 80-trap set (camera-ready E7).

R1 and R2 point out that VerifiQuant differs from R-PAL+ in two coupled ways:
it *executes* contract-bound checks, and it reaches them through *separate*
LLM calls where R-PAL+ makes one. This runner separates the two.

Variants
--------
s1  Staged R-PAL+. Exactly R-PAL+'s inputs: the same top-4 retrieval, the same
    reference rendering (title, summary, public source function) and the same
    taxonomy text, verbatim. The one R-PAL+ call is split into VerifiQuant's
    stage order, one LLM call per stage:

        1 scope     M / N      sees all top-4 references, picks one or stops
        2 inputs    F          sees the chosen reference, binds values
        3 bounds    E          judges the bound values
        4 ambiguity I_HARD/SOFT judges conventions; I_SOFT carries a warning
        5 program   solve()    writes code, executed in R-PAL+'s sandbox

    Every check is an LLM judgement. No contract predicate is executed.
    S1 vs R-PAL+ therefore differs only in staging.

s2  S1 plus the contract's declared content as text, still not executed:
    retrieval metadata (selection hints, applicability, scope) in stage 1, the
    input schema in stage 2, the E-check rules in stage 3, the semantic hints
    in stage 4. S2 vs S1 isolates contract content; VerifiQuant vs S2 isolates
    deterministic execution of F/E (VerifiQuant's I gate is itself an LLM call
    over the same hints, so S2 and VerifiQuant are close on I by design).

The first stage that does not pass ends the case. Records have R-PAL+'s shape
(``assessment`` + execution fields) plus a ``stages`` trace, so
``analyze_governance_eval.normalize_rpal`` scores them unchanged. The gold,
python_solution, operator and expected_diagnostic never enter a prompt.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_pal_baseline import extract_code  # noqa: E402
from run_retrieval_pal_baseline import (  # noqa: E402
    DEFAULT_FUNCTIONS,
    _candidate_payloads,
    _generate_with_retries,
    _git_commit,
    _load_function_library,
    _load_jsonl,
    _public_candidate_log,
    _sha256_file,
    _sha256_text,
)
from run_retrieval_pal_governance_eval import (  # noqa: E402
    PROMPTED_TEMPLATE,
    _abstain_fields,
    _execute,
    _extract_json_object,
)
from verifiquant.card_store import SQLAlchemyArtifactStore  # noqa: E402
from verifiquant.pipeline.run_error_classification_pipeline import (  # noqa: E402
    create_genai_client_from_env,
)

G = ROOT / "results/governance_confirmatory"
DEFAULT_DB = ROOT / "results/paper_v2_250/fic/cards.db"

# The taxonomy block of PROMPTED_TEMPLATE, verbatim (header through the
# "Do not force-fit one." line). Asserted below so a later edit to R-PAL+'s
# prompt cannot silently drift away from this control.
_TAXONOMY_END = "Do not force-fit one.\n"
TAXONOMY = PROMPTED_TEMPLATE[: PROMPTED_TEMPLATE.index(_TAXONOMY_END) + len(_TAXONOMY_END)]
assert TAXONOMY.startswith("You are a financial reasoning assistant using retrieved reference functions.")
assert "- I_SOFT: the calculation can proceed" in TAXONOMY

STAGE_FRAME = """
The decision is made in five separate stages, in this order:
(1) scope: M and N; (2) inputs: F; (3) boundaries: E; (4) ambiguity: I_HARD and I_SOFT; (5) program.
Any earlier stages have passed. This is stage {k}, {name}. {task}
Later stages handle the other categories, so do not judge them here.
"""

STAGES: Dict[str, Dict[str, str]] = {
    "scope": {
        "k": "1",
        "name": "scope",
        "task": (
            "Decide only whether the retrieved candidates support the requested calculation. "
            "If at least one does, choose the most applicable candidate. Return M if the user's "
            "intent or requested metric is unclear, or N if the requested calculation is "
            "unsupported or the references are only superficially related."
        ),
        "schema": """{
  "diagnostic": "PASS|M|N",
  "chosen_candidate": <candidate number, or 0 for M/N>,
  "reason": "concise evidence-grounded explanation",
  "clarification_question": "one actionable question for M, or empty string"
}""",
    },
    "inputs": {
        "k": "2",
        "name": "inputs",
        "task": (
            "Decide only whether every input the calculation needs is present. Use the chosen "
            "reference function's parameters as the list of inputs; a parameter with a default "
            "value is not missing. Bind each input to its value in the question or context. "
            "Return F if a required numeric/input field is absent or redacted."
        ),
        "schema": """{
  "diagnostic": "PASS|F",
  "inputs": [{"name": "parameter name", "status": "provided|missing", "value": "value as stated, or empty string"}],
  "missing_fields": ["exact missing field"],
  "reason": "concise evidence-grounded explanation",
  "clarification_question": "one actionable question for F, or empty string"
}""",
    },
    "bounds": {
        "k": "3",
        "name": "boundaries",
        "task": (
            "Decide only whether a provided value violates a financial, mathematical, or domain "
            "boundary. Return E if one does."
        ),
        "schema": """{
  "diagnostic": "PASS|E",
  "invalid_fields": ["invalid field or violated rule"],
  "reason": "concise evidence-grounded explanation",
  "clarification_question": "one actionable question for E, or empty string"
}""",
    },
    "ambiguity": {
        "k": "4",
        "name": "ambiguity",
        "task": (
            "Decide only whether an interpretation question remains. Return I_HARD if an "
            "unresolved unit, direction, timing, basis, or convention can change the answer, "
            "I_SOFT if the calculation can proceed but an interpretation assumption must be "
            "disclosed as a warning, or PASS if neither."
        ),
        "schema": """{
  "diagnostic": "PASS|I_HARD|I_SOFT",
  "ambiguity_targets": ["ambiguity target"],
  "reason": "concise evidence-grounded explanation",
  "clarification_question": "one actionable question for I_HARD, or empty string",
  "warning": "explicit assumption/warning for I_SOFT, or empty string"
}""",
    },
    "program": {
        "k": "5",
        "name": "program",
        "task": (
            "Write a self-contained Python solve() function that computes the answer from the "
            "bound inputs. Only `math` may be imported. If an I_SOFT assumption is listed below, "
            "follow it."
        ),
        "schema": """{
  "code": "self-contained Python code as a JSON string"
}""",
    },
}


def _render_reference(candidate: Mapping[str, Any]) -> str:
    """Identical block format to run_retrieval_pal_governance_eval._render_references."""
    return "\n".join(
        [
            f"### Candidate {candidate['rank']}: {candidate['title']}",
            f"Summary: {candidate['summary']}",
            "```python",
            str(candidate["function"]),
            "```",
        ]
    )


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            try:
                parsed = json.loads(text.replace("'", '"'))
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
        return [text] if text else []
    return []


def _contract_retrieval_text(row: Mapping[str, Any]) -> str:
    keys = ("selection_hints", "applicable_when", "not_applicable_when", "scope_boundaries")
    return "\n".join(f"{key}: {json.dumps(_as_list(row.get(key)), ensure_ascii=False)}" for key in keys)


def _contract_inputs_text(core: Mapping[str, Any]) -> str:
    rows = [
        {k: item.get(k) for k in ("name", "type", "required", "unit", "description")}
        for item in core.get("inputs") or []
        if isinstance(item, dict)
    ]
    return json.dumps(rows, ensure_ascii=False, indent=2)


def _contract_echecks_text(core: Mapping[str, Any]) -> str:
    rows = [
        {k: item.get(k) for k in ("rule_id", "name", "description", "expression", "applies_to")}
        for item in core.get("diagnostic_checks") or []
        if isinstance(item, dict) and str(item.get("diagnostic_type", "E")).upper() == "E"
    ]
    return json.dumps(rows, ensure_ascii=False, indent=2)


def _contract_hints_text(core: Mapping[str, Any]) -> str:
    rows = []
    for hint in core.get("semantic_hints") or []:
        if not isinstance(hint, dict):
            continue
        options = [
            {"label": o.get("label"), "is_default": o.get("is_default")} if isinstance(o, dict) else {"label": o}
            for o in hint.get("options") or []
        ]
        rows.append(
            {
                "id": hint.get("id"),
                "ambiguity_type": hint.get("ambiguity_type"),
                "trigger_signal": hint.get("trigger_signal"),
                "clarification_question": hint.get("clarification_question"),
                "options": options,
                "i_level": hint.get("i_level"),
                "assumption_if_not_clarified": hint.get("assumption_if_not_clarified"),
            }
        )
    return json.dumps(rows, ensure_ascii=False, indent=2)


def _stage_prompt(stage: str, row: Mapping[str, Any], body: str) -> str:
    spec = STAGES[stage]
    return (
        TAXONOMY
        + STAGE_FRAME.format(k=spec["k"], name=spec["name"], task=spec["task"])
        + "\nReturn ONLY one valid JSON object with exactly these fields:\n"
        + spec["schema"]
        + "\n\nQuestion:\n"
        + str(row.get("question", ""))
        + "\n\nContext:\n"
        + (str(row.get("context", "") or "") or "(none)")
        + "\n\n"
        + body
    )


def _call(client: Any, model: str, prompt: str, max_retries: int) -> Tuple[Dict[str, Any] | None, str, str, int]:
    """One stage call. A parse failure is re-asked once; the count is recorded."""
    parse_retries = 0
    text, api_reason = "", ""
    for _ in range(2):
        text, api_reason = _generate_with_retries(client, model=model, prompt=prompt, max_retries=max_retries)
        if api_reason:
            return None, text, api_reason, parse_retries
        parsed = _extract_json_object(text)
        if parsed is not None:
            return parsed, text, "", parse_retries
        parse_retries += 1
    return None, text, "stage_parse_fail", parse_retries


def _diag(parsed: Mapping[str, Any], allowed: Sequence[str]) -> str:
    value = str(parsed.get("diagnostic", "")).strip().upper()
    return value if value in allowed else "PASS"


DECISION_FOR = {"M": "clarify", "N": "refuse", "F": "clarify", "E": "refuse", "I_HARD": "clarify"}


def run_case(
    *,
    row: Mapping[str, Any],
    variant: str,
    candidates: Sequence[Mapping[str, Any]],
    retrieval_by_fic: Mapping[str, Mapping[str, Any]],
    core_by_id: Mapping[str, Mapping[str, Any]],
    client: Any,
    model: str,
    max_retries: int,
) -> Dict[str, Any]:
    stages: List[Dict[str, Any]] = []
    with_contract = variant == "s2"

    def record_stage(name: str, prompt: str, parsed: Any, text: str, reason: str, retries: int, t0: float) -> None:
        stages.append(
            {
                "stage": name,
                "prompt_sha256": _sha256_text(prompt),
                "prompt_chars": len(prompt),
                "response_text": text,
                "parsed": parsed,
                "error": reason,
                "parse_retries": retries,
                "elapsed_s": round(time.time() - t0, 3),
            }
        )

    def stop(diagnostic: str, parsed: Mapping[str, Any], targets: List[Any], **extra: str) -> Dict[str, Any]:
        assessment = {
            "decision": DECISION_FOR.get(diagnostic, "refuse"),
            "diagnostic": diagnostic,
            "reason": str(parsed.get("reason") or ""),
            "action_targets": [str(x) for x in targets if str(x).strip()],
            "clarification_question": str(parsed.get("clarification_question") or ""),
            "warning": extra.get("warning", ""),
            "code": "",
            "stopped_at": stages[-1]["stage"],
        }
        return {"assessment": assessment, "stages": stages, **_abstain_fields("governance_intercept")}

    def failed(reason: str) -> Dict[str, Any]:
        # API or parse failure: no assessment, scored like R-PAL+'s parse failures.
        return {"assessment": None, "stages": stages, **_abstain_fields(reason)}

    # ---- stage 1: scope (M/N) over all top-k references -------------------
    refs = []
    for cand in candidates:
        block = _render_reference(cand)
        if with_contract:
            block += "\nContract metadata:\n" + _contract_retrieval_text(retrieval_by_fic.get(cand["fic_id"], {}))
        refs.append(block)
    body = "Retrieved reference functions (ranked):\n" + ("\n\n".join(refs) if refs else "(no candidates retrieved)")
    prompt = _stage_prompt("scope", row, body)
    t0 = time.time()
    parsed, text, reason, retries = _call(client, model, prompt, max_retries)
    record_stage("scope", prompt, parsed, text, reason, retries, t0)
    if parsed is None:
        return failed(reason)
    diagnostic = _diag(parsed, ("PASS", "M", "N"))
    if not candidates and diagnostic == "PASS":
        diagnostic = "N"
    if diagnostic != "PASS":
        return stop(diagnostic, parsed, [])
    try:
        chosen_rank = int(parsed.get("chosen_candidate") or 0)
    except (TypeError, ValueError):
        chosen_rank = 0
    chosen = next((c for c in candidates if c["rank"] == chosen_rank), None)
    if chosen is None:
        # PASS without a valid pick: take rank 1 and record it, rather than invent an exit.
        chosen = candidates[0]
        stages[-1]["chosen_fallback_rank1"] = True
    core = core_by_id.get(chosen["fic_id"], {})
    chosen_block = "Chosen reference function:\n" + _render_reference(chosen)

    # ---- stage 2: inputs (F) ----------------------------------------------
    body = chosen_block
    if with_contract:
        body += "\n\nContract input schema:\n" + _contract_inputs_text(core)
    prompt = _stage_prompt("inputs", row, body)
    t0 = time.time()
    parsed, text, reason, retries = _call(client, model, prompt, max_retries)
    record_stage("inputs", prompt, parsed, text, reason, retries, t0)
    if parsed is None:
        return failed(reason)
    if _diag(parsed, ("PASS", "F")) == "F":
        return stop("F", parsed, _as_list(parsed.get("missing_fields")))
    bound = [i for i in _as_list(parsed.get("inputs")) if isinstance(i, dict)]
    bound_text = "Bound inputs (from stage 2):\n" + json.dumps(bound, ensure_ascii=False, indent=2)

    # ---- stage 3: boundaries (E) ------------------------------------------
    body = chosen_block + "\n\n" + bound_text
    if with_contract:
        body += "\n\nContract boundary checks (declared rules; judge them, they are not run):\n" + _contract_echecks_text(core)
    prompt = _stage_prompt("bounds", row, body)
    t0 = time.time()
    parsed, text, reason, retries = _call(client, model, prompt, max_retries)
    record_stage("bounds", prompt, parsed, text, reason, retries, t0)
    if parsed is None:
        return failed(reason)
    if _diag(parsed, ("PASS", "E")) == "E":
        return stop("E", parsed, _as_list(parsed.get("invalid_fields")))

    # ---- stage 4: ambiguity (I_HARD / I_SOFT) -----------------------------
    body = chosen_block + "\n\n" + bound_text
    if with_contract:
        body += "\n\nContract semantic hints (declared ambiguities):\n" + _contract_hints_text(core)
    prompt = _stage_prompt("ambiguity", row, body)
    t0 = time.time()
    parsed, text, reason, retries = _call(client, model, prompt, max_retries)
    record_stage("ambiguity", prompt, parsed, text, reason, retries, t0)
    if parsed is None:
        return failed(reason)
    i_diag = _diag(parsed, ("PASS", "I_HARD", "I_SOFT"))
    if i_diag == "I_HARD":
        return stop("I_HARD", parsed, _as_list(parsed.get("ambiguity_targets")))
    i_parsed = parsed
    warning = str(parsed.get("warning") or "") if i_diag == "I_SOFT" else ""

    # ---- stage 5: program ---------------------------------------------------
    body = chosen_block + "\n\n" + bound_text
    if warning:
        body += "\n\nI_SOFT assumption (from stage 4):\n" + warning
    prompt = _stage_prompt("program", row, body)
    t0 = time.time()
    parsed, text, reason, retries = _call(client, model, prompt, max_retries)
    record_stage("program", prompt, parsed, text, reason, retries, t0)
    if parsed is None and reason != "stage_parse_fail":
        return failed(reason)
    code = str((parsed or {}).get("code") or "").strip() or extract_code(text)
    execution = _execute(code)
    assessment = {
        "decision": "warn" if i_diag == "I_SOFT" else "execute",
        "diagnostic": i_diag if i_diag == "I_SOFT" else "READY",
        "reason": str(i_parsed.get("reason") or ""),
        "action_targets": [str(x) for x in _as_list(i_parsed.get("ambiguity_targets")) if str(x).strip()]
        if i_diag == "I_SOFT" else [],
        "clarification_question": "",
        "warning": warning,
        "code": code or "",
        "stopped_at": "program",
    }
    return {"assessment": assessment, "stages": stages, **execution}


def _client_with_timeout(timeout_s: int) -> Any:
    """create_genai_client_from_env plus an HTTP timeout.

    Without one, a request on a connection that died (for example after a
    network switch) blocks forever; with one it raises and the caller retries.
    """
    import os
    from google import genai
    from google.genai import types as genai_types

    create_genai_client_from_env()  # same key check and error as every other runner
    return genai.Client(
        api_key=os.environ["GEMINI_API_KEY"],
        http_options=genai_types.HttpOptions(timeout=timeout_s * 1000),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=("s1", "s2"), required=True)
    parser.add_argument("--dataset", type=Path, default=G / "trap_set.jsonl")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--function-library", type=Path, default=DEFAULT_FUNCTIONS)
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard", default="", help="i/n: keep rows whose index mod n == i")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--http-timeout-s", type=int, default=120)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    rows = _load_jsonl(args.dataset)
    if args.limit:
        rows = rows[: args.limit]
    if args.shard:
        shard_i, shard_n = (int(x) for x in args.shard.split("/"))
        rows = [r for idx, r in enumerate(rows) if idx % shard_n == shard_i]
    function_by_id = _load_function_library(args.function_library)
    store = SQLAlchemyArtifactStore(f"sqlite:///{args.db.resolve()}")
    core_by_id = store.load_core_by_id()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.out_dir / f"{args.run_name}.jsonl"
    manifest_path = args.out_dir / f"{args.run_name}_manifest.json"
    if output_path.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {output_path}; use --resume")
    completed: set[str] = set()
    if args.resume and output_path.exists():
        completed = {str(r["case_id"]) for r in _load_jsonl(output_path)}

    manifest = {
        "protocol": "staged_rpal_governance_v1",
        "experiment": "camera-ready E7",
        "variant": args.variant,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "model": args.model,
        "http_timeout_s": args.http_timeout_s,
        "top_k": args.top_k,
        "shard": args.shard,
        "dataset": {"path": str(args.dataset.resolve()), "sha256": _sha256_file(args.dataset)},
        "card_db": {"path": str(args.db.resolve()), "sha256": _sha256_file(args.db)},
        "function_library": {
            "path": str(args.function_library.resolve()),
            "sha256": _sha256_file(args.function_library),
            "size": len(function_by_id),
        },
        "retrieval_query": "question + newline + context",
        "taxonomy_text_sha256": _sha256_text(TAXONOMY),
        "stage_frame": STAGE_FRAME,
        "stages": STAGES,
        "forbidden_prompt_inputs": ["ground_truth", "python_solution", "operator", "expected_diagnostic"],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    client = _client_with_timeout(args.http_timeout_s)
    started = time.time()
    with output_path.open("a" if args.resume else "w", encoding="utf-8") as fh:
        for index, row in enumerate(rows, 1):
            case_id = str(row.get("question_id") or row.get("trap_id") or row.get("case_id") or row.get("id"))
            if case_id in completed:
                continue
            case_started = time.time()
            question = str(row.get("question", ""))
            context = str(row.get("context", "") or "")
            candidates = _candidate_payloads(
                store=store, core_by_id=core_by_id, function_by_id=function_by_id,
                question=question, context=context, top_k=args.top_k,
            )
            retrieval_by_fic: Dict[str, Mapping[str, Any]] = {}
            if args.variant == "s2":
                for r in store.retrieve_candidates(query=f"{question}\n{context}", top_k=args.top_k):
                    retrieval_by_fic[str(r.get("fic_id", "")).strip()] = r
            result = run_case(
                row=row, variant=args.variant, candidates=candidates, retrieval_by_fic=retrieval_by_fic,
                core_by_id=core_by_id, client=client, model=args.model, max_retries=args.max_retries,
            )
            record = {
                "case_id": case_id,
                "base_question_id": row.get("base_question_id") or case_id,
                "dataset_operator": row.get("operator"),
                "mode": f"staged_{args.variant}",
                "candidates": _public_candidate_log(candidates),
                "elapsed_s": round(time.time() - case_started, 3),
                **result,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            a = record.get("assessment") or {}
            answered = "numeric" if record.get("output") is not None else record["execution_reason"]
            print(f"[{args.run_name} {index}/{len(rows)}] {case_id} diag={a.get('diagnostic', '-')} "
                  f"stop={a.get('stopped_at', '-')} {answered}", flush=True)

    manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["elapsed_s"] = round(time.time() - started, 1)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
