#!/usr/bin/env python3
"""Scripted-user control on the F and I_HARD traps (camera-ready E6).

The traps run K=1 everywhere else, so no table shows what a clarification is
worth when the information really is missing. This runner adds one scripted
user turn to the existing turn-1 outputs and re-runs the same system.

Scripted user (no LLM, never sees the gold):
- It replies only when turn 1 withheld a number.
- F trap: "The <field> is <value>.", where <value> is the exact span the trap
  replaced with [REDACTED] (taken from the base question), plus any unit glued to
  it (%, .00, "trillion", ...).
- I_HARD trap: "It is a call option." / "Use the FIFO method.", one fixed template
  per ambiguity family, filled with the trap's removed phrase.
- The reply is appended to the context and the system runs again (K=1).

Two user policies are scored offline from the same rows:
- lenient: replies to every turn-1 abstention.
- strict:  replies only if turn 1 named the removed field (F) or the removed
  convention (I_HARD) in its question, targets, or reason. Underscores and spaces
  are treated alike, and conventions are matched as whole words.
The gap between the two is what asking the right question is worth.

Turn-1 sources: VerifiQuant = the three 2026-10-14 top-4 Flash reruns;
R-PAL+ = the three July prompted Flash runs. Turn 2 uses the same backbone,
top-4 retrieval and code path as turn 1 for each system.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from analyze_governance_eval import load_jsonl, normalize_rpal, normalize_vq  # noqa: E402
from run_pal_baseline import extract_code  # noqa: E402
from run_retrieval_pal_baseline import (  # noqa: E402
    DEFAULT_FUNCTIONS,
    _candidate_payloads,
    _generate_with_retries,
    _git_commit,
    _load_function_library,
    _score_output,
)
from run_retrieval_pal_governance_eval import (  # noqa: E402
    _abstain_fields,
    _execute,
    _extract_json_object,
    _prompted_prompt,
)
from run_vq_trap_rerun import _client_with_timeout  # noqa: E402
from verifiquant.card_store import SQLAlchemyArtifactStore  # noqa: E402
from verifiquant.pipeline.run_error_classification_pipeline import ErrorClassificationAPI  # noqa: E402
from verifiquant.pipeline.run_framework_guided_self_improve_pipeline import _framework_oracle_loop  # noqa: E402

RUNS = ROOT / "results"
G = RUNS / "governance_confirmatory"
DB = RUNS / "paper_v2_250/fic/cards.db"
TURN1 = {
    "vq": [RUNS / f"governance_confirmatory_1014/vq_k1_top4_run{i}.jsonl" for i in (1, 2, 3)],
    "rpal_plus": [G / f"conf_prompted_run{i}.jsonl" for i in (1, 2, 3)],
}

_UNIT_AFTER = re.compile(r"^(%|\.\d+| (?:trillion|billion|million|thousand|dollars))")
CONVENTION_PATTERNS = {
    "option_type": re.compile(r"\b(call|put)\b|option type", re.I),
    "inventory": re.compile(r"\b(fifo|lifo)\b|inventory (?:valuation|costing) method|costing method|valuation method", re.I),
}
REPLY_TEMPLATES = {
    "option_type": "It is a {phrase} option.",
    "inventory": "Use the {phrase} method.",
}


def redacted_value(trap: Mapping[str, Any], base: Mapping[str, Any]) -> str:
    """The exact text the trap replaced with [REDACTED], plus a glued unit."""
    field = "context" if "[REDACTED]" in trap["context"] else "question"
    prefix, suffix = trap[field].split("[REDACTED]", 1)
    original = base[field]
    assert original.startswith(prefix) and original.endswith(suffix), trap["trap_id"]
    span = original[len(prefix): len(original) - len(suffix)]
    unit = _UNIT_AFTER.match(suffix)
    return span + (unit.group(1) if unit else "")


def user_reply(trap: Mapping[str, Any], base: Mapping[str, Any]) -> str:
    g = trap["grounding"]
    if trap["operator"] == "F":
        return f"The {g['removed_field'].replace('_', ' ')} is {redacted_value(trap, base)}."
    return REPLY_TEMPLATES[g["ambiguity_family"]].format(phrase=g["removed_phrase"])


def turn1_text(system: str, raw: Mapping[str, Any], norm: Mapping[str, Any]) -> str:
    """Everything the turn-1 exit said to the user, lower-cased."""
    parts = [str(x) for x in norm.get("action_targets") or []]
    parts += [str(norm.get("reason") or ""), str(norm.get("clarification_question") or "")]
    if system == "vq":
        d = raw["framework_guided"]["history"][0]["diagnostic"]
        parts += [str(x) for x in d.get("requested_fields") or []]
        req = d.get("clarification_request") or {}
        parts += [str(x) for x in req.get("questions") or []]
        for hint in d.get("repair_hints") or []:
            parts.append(str(hint.get("user_message") or ""))
    return " ".join(parts).lower()


def named_removed_item(trap: Mapping[str, Any], text: str) -> bool:
    g = trap["grounding"]
    if trap["operator"] == "F":
        field = g["removed_field"].lower()
        return field in text or field.replace("_", " ") in text
    return bool(CONVENTION_PATTERNS[g["ambiguity_family"]].search(text))


def vq_turn2(api: ErrorClassificationAPI, client: Any, model: str, row: Dict[str, Any]) -> Dict[str, Any]:
    fw = _framework_oracle_loop(api=api, client=client, oracle_model=model, row=row, max_turns=1)
    d = fw["history"][0]["diagnostic"]
    out = d.get("output_value")
    return {"diagnostic": d.get("diagnostic_type"), "status": d.get("status"), "output": out,
            "correct": bool(d.get("is_correct")) if out is not None else None, "raw": fw}


def rpal_turn2(client: Any, model: str, row: Dict[str, Any], store, core_by_id, function_by_id) -> Dict[str, Any]:
    candidates = _candidate_payloads(store=store, core_by_id=core_by_id, function_by_id=function_by_id,
                                     question=row["question"], context=row["context"], top_k=4)
    prompt = _prompted_prompt(row, candidates)
    text, api_reason = _generate_with_retries(client, model=model, prompt=prompt, max_retries=5)
    assessment = None
    if api_reason:
        execution = _abstain_fields(api_reason)
    else:
        assessment = _extract_json_object(text)
        if assessment is None:
            execution = _abstain_fields("assessment_parse_fail")
        elif str(assessment.get("decision", "")).strip().lower() in {"execute", "warn"}:
            execution = _execute(str(assessment.get("code") or "").strip() or extract_code(text))
        else:
            execution = _abstain_fields("governance_intercept")
    out = execution.get("output")
    correct = None
    if out is not None:
        correct = _score_output(out, row["ground_truth"], row["question"])["verdict_unit_normalized"] == "correct"
    return {"diagnostic": (assessment or {}).get("diagnostic"), "status": execution.get("execution_reason") or "numeric",
            "output": out, "correct": correct, "raw": {"response_text": text, "assessment": assessment, **execution}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", choices=sorted(TURN1), required=True)
    ap.add_argument("--run", type=int, choices=(1, 2, 3), required=True)
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--out-dir", type=Path, default=RUNS / "scripted_user_1015")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    traps = [t for t in load_jsonl(G / "trap_set.jsonl")
             if t["operator"] == "F" or t["expected_diagnostic"] == "I_HARD"]
    base = {r["question_id"]: r for r in load_jsonl(RUNS / "paper_v2_250/questions_250.jsonl")}
    turn1_path = TURN1[args.system][args.run - 1]
    normalizer = normalize_vq if args.system == "vq" else normalize_rpal
    turn1 = {str(r["case_id"]): r for r in load_jsonl(turn1_path)}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{args.system}_run{args.run}.jsonl"
    if out_path.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {out_path}; use --resume")
    done = {str(r["trap_id"]) for r in load_jsonl(out_path)} if args.resume and out_path.exists() else set()
    manifest = {
        "protocol": "scripted_user_v1", "experiment": "camera-ready E6",
        "created_at": datetime.now().isoformat(timespec="seconds"), "git_commit": _git_commit(),
        "system": args.system, "run": args.run, "model": args.model, "top_k": 4,
        "turn1_source": str(turn1_path.relative_to(ROOT)),
        "reply_templates": {"F": "The {removed_field with spaces} is {redacted span + unit}.", **REPLY_TEMPLATES},
        "reply_channel": "appended to context as 'User reply to the clarification request: <reply>'",
    }
    (args.out_dir / f"{args.system}_run{args.run}_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    client = _client_with_timeout(120)
    api = store = core_by_id = function_by_id = None
    if args.system == "vq":
        api = ErrorClassificationAPI.from_db(db_url=f"sqlite:///{DB}", client=client, selector_model=args.model,
                                             extractor_model=args.model, judge_model=args.model, top_k=4,
                                             m_min_top_score=0.05)
    else:
        store = SQLAlchemyArtifactStore(f"sqlite:///{DB}")
        core_by_id = store.load_core_by_id()
        function_by_id = _load_function_library(DEFAULT_FUNCTIONS)

    with out_path.open("a" if args.resume else "w", encoding="utf-8") as fh:
        for trap in traps:
            tid = str(trap["trap_id"])
            if tid in done:
                continue
            raw1 = turn1[tid]
            norm1 = normalizer(raw1)
            text1 = turn1_text(args.system, raw1, norm1)
            record: Dict[str, Any] = {
                "trap_id": tid, "operator": trap["operator"], "expected": trap["expected_diagnostic"],
                "system": args.system, "run": args.run,
                "turn1": {"diagnostic": norm1["diagnostic"], "numeric": norm1["numeric"],
                          "named_removed_item": named_removed_item(trap, text1)},
                "reply": None, "turn2": None,
            }
            if not norm1["numeric"]:
                reply = user_reply(trap, base[trap["base_question_id"]])
                row = {**trap, "case_id": tid,
                       "context": f"{trap['context']}\n\nUser reply to the clarification request: {reply}"}
                record["reply"] = reply
                t2: Optional[Dict[str, Any]] = None
                for attempt in range(4):
                    try:
                        t2 = (vq_turn2(api, client, args.model, row) if args.system == "vq"
                              else rpal_turn2(client, args.model, row, store, core_by_id, function_by_id))
                        break
                    except Exception as exc:  # API errors inside the VQ engine
                        record.setdefault("retry_errors", []).append(f"{type(exc).__name__}: {str(exc)[:160]}")
                        time.sleep(5 * (attempt + 1))
                record["turn2"] = t2
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            t2 = record["turn2"] or {}
            print(f"[{args.system} run{args.run}] {tid} t1={norm1['diagnostic']}"
                  f"{' NUM' if norm1['numeric'] else ''} named={record['turn1']['named_removed_item']} "
                  f"t2={t2.get('diagnostic')} out={t2.get('output')} correct={t2.get('correct')}", flush=True)


if __name__ == "__main__":
    main()
