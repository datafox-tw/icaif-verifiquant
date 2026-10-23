from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
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

from verifiquant.preprocessing.validate_relations import validate_artifact_relations
from verifiquant.preprocessing.lint_echecks import ECHECK_SAFE_BUILTINS
from verifiquant.preprocessing.common import safe_fic_import
from verifiquant.preprocessing.stage_repair import GLOBAL_N_NOT_SUPPORTED_RULE, GLOBAL_SCOPE_FIC_ID
from verifiquant.card_store import SQLAlchemyArtifactStore


@dataclass
class InputCoercionResult:
    """Container for the result of coercing one FIC input value.

    Attributes
    ----------
    value:
        The coerced Python value.  ``None`` means the raw string could not be
        parsed to the expected type (treat as missing).
    confidence:
        * ``"explicit"``   — source format was unambiguous (e.g. the source
          text had a ``%`` sign or a clear decimal, and the FIC unit agrees).
        * ``"inferred"``   — LLM made a reasonable assumption; result is
          likely correct but not 100 % certain.
        * ``"ambiguous"``  — source format or FIC unit field is contradictory;
          downstream should ask the user for confirmation or offer a card
          unit correction.
    note:
        Human-readable coercion log entry to be appended to
        ``normalization_note``.
    """
    value: Optional[Any]
    confidence: str  # "explicit" | "inferred" | "ambiguous"
    note: str


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_]+", str(text or "").lower())


def _parse_number(value: Any, *, percent_as_decimal: bool = True) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().replace(",", "").replace("_", "")
    if not s:
        return None

    multiplier = 1.0
    s_lower = s.lower()
    if re.search(r"\btrillion\b|\bt\b", s_lower):
        multiplier = 1e12
    elif re.search(r"\bbillion\b|\bbn\b|\bb\b", s_lower):
        multiplier = 1e9
    elif re.search(r"\bmillion\b|\bmm\b|\bmn\b|\bm\b", s_lower):
        multiplier = 1e6
    elif re.search(r"\bthousand\b|\bk\b", s_lower):
        multiplier = 1e3

    is_pct = "%" in s
    s = s.replace("%", "")
    m = re.search(r"-?\d+(\.\d+)?", s)
    if not m:
        return None
    n = float(m.group())
    if is_pct:
        return n / 100.0 if percent_as_decimal else n
    return n * multiplier


def _extract_numeric_list(value: Any, *, percent_as_decimal: bool = True) -> List[float]:
    if value is None:
        return []
    if isinstance(value, list):
        out = []
        for v in value:
            n = _parse_number(v, percent_as_decimal=percent_as_decimal)
            if n is not None:
                out.append(n)
        return out

    s = str(value).strip()
    if not s:
        return []

    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            out = []
            for v in parsed:
                n = _parse_number(v, percent_as_decimal=percent_as_decimal)
                if n is not None:
                    out.append(n)
            if out:
                return out
    except Exception:
        pass

    tokens = re.findall(r"-?\d[\d,]*(?:\.\d+)?%?", s)
    out = []
    for tok in tokens:
        n = _parse_number(tok, percent_as_decimal=percent_as_decimal)
        if n is not None:
            out.append(n)
    return out


def _resolve_schema_scale(unit: Any, description: Any) -> str:
    """Determine the expected numeric scale from the FIC input schema.

    This is the single source of truth for the FIC side of the
    percent-vs-decimal decision, replacing the old
    ``_infer_percent_as_decimal`` / ``_is_percent_point_unit`` pair that
    expressed the same logic from two opposite angles.

    Returns
    -------
    ``"decimal"``
        FIC code expects values in [0, 1] decimal form (e.g. 8 % → 0.08).
    ``"percent_point"``
        FIC code expects values as percentage points (e.g. 8 % → 8).
    ``"unknown"``
        No specific scale convention declared; treat as plain numeric.
    """
    u = str(unit or "").strip().lower()
    d = str(description or "").strip().lower()
    if "decimal" in u or "as a decimal" in d or "expressed as a decimal" in d:
        return "decimal"
    if (
        u in {"%", "percent", "percentage", "percentage_point", "percentage_points"}
        or ("percentage" in u and "decimal" not in u)
    ):
        return "percent_point"
    return "unknown"


def _parse_nested_list(raw_value: Any, *, percent_as_decimal: bool = True) -> Optional[List[List[float]]]:
    """Try to parse raw_value as a list-of-lists of floats.

    Handles three source shapes:
    - Already a Python list-of-lists / list-of-tuples  →  validate and coerce.
    - A JSON string encoding a list-of-lists           →  parse then coerce.
    - A flat list/string of numbers whose length is divisible by the inferred row
      width (taken from the first valid sub-list)      →  chunk and return.

    Returns ``None`` when the value cannot be reliably interpreted as nested.
    """
    if raw_value is None:
        return None

    # --- normalise to a Python object ---
    raw: Any = raw_value
    if isinstance(raw_value, str):
        s = raw_value.strip()
        if not s:
            return None
        try:
            raw = json.loads(s)
        except Exception:
            return None

    if not isinstance(raw, list) or not raw:
        return None

    # --- already a list-of-lists / list-of-tuples ---
    if isinstance(raw[0], (list, tuple)):
        out: List[List[float]] = []
        row_len: Optional[int] = None
        for item in raw:
            if not isinstance(item, (list, tuple)):
                return None
            if row_len is None:
                row_len = len(item)
            elif len(item) != row_len:
                return None  # inconsistent row widths
            row: List[float] = []
            for v in item:
                n = _parse_number(v, percent_as_decimal=percent_as_decimal)
                if n is None:
                    return None
                row.append(n)
            out.append(row)
        return out if out else None

    return None  # flat list — caller decides


def _parse_json_object(raw_value: Any) -> Optional[Any]:
    if raw_value is None:
        return None
    if isinstance(raw_value, (dict, list)):
        return raw_value
    if isinstance(raw_value, str):
        s = raw_value.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except Exception:
            return None
    return None


def _coerce_numeric_dict(raw_value: Any, *, percent_as_decimal: bool = True) -> Optional[Dict[str, float]]:
    parsed = _parse_json_object(raw_value)
    if not isinstance(parsed, dict):
        return None
    out: Dict[str, float] = {}
    for key, value in parsed.items():
        n = _parse_number(value, percent_as_decimal=percent_as_decimal)
        if n is None:
            return None
        out[str(key)] = n
    return out


def _coerce_object_list(raw_value: Any) -> Optional[List[Dict[str, Any]]]:
    parsed = _parse_json_object(raw_value)
    if not isinstance(parsed, list):
        return None
    if not all(isinstance(item, dict) for item in parsed):
        return None
    return [dict(item) for item in parsed]


def _parse_typed_value(
    raw_value: Any,
    declared_type: str,
    *,
    unit: Any = None,
    description: Any = None,
    source_scale: str = "inferred",
) -> Optional[Any]:
    """Thin wrapper around :func:`coerce_input` that returns just the value.

    Kept for call-sites that don't need the full :class:`InputCoercionResult`.
    """
    return coerce_input(
        raw_value,
        declared_type,
        unit=unit,
        description=description,
        source_scale=source_scale,
    ).value


def coerce_input(
    raw_value: Any,
    declared_type: str,
    *,
    unit: Any = None,
    description: Any = None,
    source_scale: str = "inferred",
) -> InputCoercionResult:
    """Single entry point for all FIC input value coercion.

    Parameters
    ----------
    raw_value:
        The raw string (or already-parsed value) returned by the LLM.
    declared_type:
        The ``type`` field from the FIC card's input schema
        (e.g. ``"number"``, ``"integer"``, ``"array[number]"``).
    unit:
        The ``unit`` field from the FIC card's input schema.
    description:
        The ``description`` field from the FIC card's input schema.
    source_scale:
        Signal from the LLM about the source format:
        * ``"explicit_percent"``  — source text had a ``%`` sign.
        * ``"explicit_decimal"``  — source text was already a decimal (0.08).
        * ``"explicit_integer"``  — source text was an unambiguous integer
          with no ``%`` sign.
        * ``"inferred"``          — LLM had to make a scale assumption.

    Returns
    -------
    :class:`InputCoercionResult`
    """
    dt = str(declared_type or "").lower()
    schema_scale = _resolve_schema_scale(unit, description)

    # ------------------------------------------------------------------
    # Decide the `percent_as_decimal` flag for _parse_number.
    # When schema says decimal: % signs should divide by 100 (True).
    # When schema says percent_point: % signs keep their face value (False).
    # ------------------------------------------------------------------
    percent_as_decimal: bool = schema_scale != "percent_point"

    # ------------------------------------------------------------------
    # Compute confidence based on source_scale × schema_scale.
    # ------------------------------------------------------------------
    schema_unit_contradiction = False
    if source_scale == "explicit_percent":
        # Source had %, schema explicitly sets a convention → explicit.
        confidence = "explicit"
    elif source_scale in ("explicit_decimal", "explicit_integer"):
        if schema_scale == "percent_point" and source_scale == "explicit_decimal":
            # Source is 0.15 but FIC schema says percent_point — likely a
            # FIC card unit-field error (AI wrote 'percent' but code uses 0-1).
            confidence = "ambiguous"
            schema_unit_contradiction = True
        else:
            confidence = "explicit"
    else:  # "inferred"
        confidence = "inferred"

    note_parts: List[str] = []
    if schema_unit_contradiction:
        note_parts.append(
            f"Warning: FIC card declares unit='{unit}' (percent_point convention) "
            "but the extracted value appears to be in decimal form. "
            "The FIC card's unit field may be incorrect (should be 'decimal')."
        )

    # ------------------------------------------------------------------
    # Parse by declared_type.
    # ------------------------------------------------------------------
    if "dict" in dt or "object" == dt:
        if "number" in dt:
            obj = _coerce_numeric_dict(raw_value, percent_as_decimal=percent_as_decimal)
        else:
            obj = _parse_json_object(raw_value)
        return InputCoercionResult(
            value=obj,
            confidence=confidence,
            note=" ".join(note_parts) if note_parts else f"Parsed as JSON object ({declared_type}).",
        )

    if "array" in dt or "list" in dt:
        if "object" in dt:
            obj_list = _coerce_object_list(raw_value)
            return InputCoercionResult(
                value=obj_list,
                confidence=confidence,
                note=" ".join(note_parts) if note_parts else f"Parsed as list of objects ({declared_type}).",
            )
        is_nested = (
            ("array" in dt and any(tok in dt for tok in ("array[", "tuple", "list[")))
            or ("list" in dt and any(tok in dt for tok in ("list[", "tuple", "array[")))
        ) and dt.count("[") > 1

        if is_nested:
            nested = _parse_nested_list(raw_value, percent_as_decimal=percent_as_decimal)
            if nested is not None:
                return InputCoercionResult(
                    value=nested,
                    confidence=confidence,
                    note=" ".join(note_parts) if note_parts else f"Parsed as nested list ({declared_type}).",
                )
        vals = _extract_numeric_list(raw_value, percent_as_decimal=percent_as_decimal)
        return InputCoercionResult(
            value=vals if vals else None,
            confidence=confidence,
            note=" ".join(note_parts) if note_parts else f"Parsed as numeric list ({declared_type}).",
        )

    if "int" in dt:
        n = _parse_number(raw_value, percent_as_decimal=percent_as_decimal)
        return InputCoercionResult(
            value=int(round(n)) if n is not None else None,
            confidence=confidence,
            note=" ".join(note_parts) if note_parts else "Parsed as integer.",
        )

    if "bool" in dt:
        s = str(raw_value or "").strip().lower()
        val: Optional[bool]
        if s in {"true", "1", "yes"}:
            val = True
        elif s in {"false", "0", "no"}:
            val = False
        else:
            val = None
        return InputCoercionResult(
            value=val,
            confidence="explicit",
            note="Parsed as boolean.",
        )

    if "str" in dt or "string" in dt or "text" in dt:
        val = str(raw_value).strip() if raw_value is not None else ""
        return InputCoercionResult(
            value=val or None,
            confidence=confidence,
            note="Parsed as string.",
        )

    # Default: scalar number.
    n = _parse_number(raw_value, percent_as_decimal=percent_as_decimal)
    if n is not None and not note_parts:
        # Build a concise note about scale conversion when relevant.
        raw_s = str(raw_value or "").strip()
        if source_scale == "explicit_percent" and schema_scale == "decimal":
            note_parts.append(f"{raw_s} converted from percent to decimal: {n}.")
        elif source_scale == "explicit_decimal" and schema_scale == "percent_point":
            note_parts.append(f"{raw_s} kept as decimal (FIC unit may be misconfigured).")
    return InputCoercionResult(
        value=n,
        confidence=confidence,
        note=" ".join(note_parts) if note_parts else f"Parsed: {raw_value} → {n}.",
    )


def _answer_match(
    question: str,
    output_value: Optional[float],
    gold_num: Optional[float],
) -> Tuple[Optional[float], Optional[bool], Optional[str]]:
    """Compare pipeline output to the gold answer.

    Returns
    -------
    (abs_error, is_correct, scale_note)

    ``scale_note`` is ``None`` when the match is straightforward.  When it
    is set, the answer is still considered correct but carries a diagnostic
    annotation.  Currently defined values:

    * ``"output_scale_x100"``  — FIC card outputs a decimal fraction
      (e.g. 0.0986) but the gold answer is in percentage-point form
      (e.g. 9.86).  The computation is numerically correct; the FIC card
      should be updated to multiply its output by 100.
    * ``"rel_tol_pass"``       — Not an exact match but within 1 % relative
      tolerance (floating-point rounding etc.).
    """
    if output_value is None or gold_num is None:
        return None, None, None

    q = question.lower()
    round_mode = any(x in q for x in ["nearest integer", "nearest whole", "round to nearest", "四捨五入", "整數"])
    if round_mode:
        lhs = float(round(output_value))
        rhs = float(round(gold_num))
        err = abs(lhs - rhs)
        return err, lhs == rhs, None

    abs_err = abs(output_value - gold_num)

    # Exact / near-exact match (rel_tol = 1 %, abs_tol = 0.01).
    if math.isclose(output_value, gold_num, rel_tol=1e-2, abs_tol=1e-2):
        scale_note = None if math.isclose(output_value, gold_num, rel_tol=1e-6, abs_tol=1e-2) else "rel_tol_pass"
        return abs_err, True, scale_note

    # ×100 output scale: FIC code returns decimal, gold is percentage-point.
    # Only check when both values are non-zero to avoid div-by-zero false hits.
    if gold_num != 0 and output_value != 0:
        if math.isclose(output_value * 100.0, gold_num, rel_tol=1e-2, abs_tol=1e-2):
            return abs_err, True, "output_scale_x100"

    return abs_err, False, None


class _AttrDict(dict):
    """Dict wrapper that also supports attribute access (inputs.foo)."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


def _safe_eval_rule(
    rule: str,
    inputs: Dict[str, Any],
    *,
    compute_fn: Optional[Any] = None,
) -> Tuple[Optional[bool], Optional[str]]:
    """Evaluate a diagnostic rule expression against provided inputs.

    Parameters
    ----------
    rule:
        The expression string to evaluate (e.g. ``inputs['x'] < 0``).
    inputs:
        The bound input values for the current FIC.
    compute_fn:
        Optional: the compiled ``compute`` function from the FIC's execution
        block.  Some FIC E-check expressions call ``compute(...)`` to inspect
        the output range, so we expose it in the eval environment when provided.
    """
    if not rule:
        return None, "empty rule expression"
    inputs_obj = _AttrDict(inputs)
    env: Dict[str, Any] = {
        # NOTE: __builtins__ must live inside env (passed as globals to eval).
        # Generator / comprehension expressions look up free variables in the
        # *globals* dict, NOT in the calling frame's locals.  Placing all
        # helpers here ensures they are visible inside generator bodies too.
        #
        # The safe-name set is owned by verifiquant.preprocessing.lint_echecks
        # (ECHECK_SAFE_BUILTINS) so the build-time E-check linter and this
        # runtime evaluator can never drift apart.
        "__builtins__": {},
        "inputs": inputs_obj,
        **ECHECK_SAFE_BUILTINS,
    }
    # Flatten input values into the env so expressions can reference them
    # directly (e.g. ``discount_rate < 0`` instead of ``inputs['discount_rate'] < 0``).
    env.update(inputs_obj)
    # Expose the FIC's compiled compute function when available so that E-check
    # expressions like ``compute({...}) > 10`` work correctly.
    if compute_fn is not None and callable(compute_fn):
        env["compute"] = compute_fn
    try:
        return bool(eval(rule, env)), None
    except Exception as exc:
        return None, str(exc)


def _infer_predicate_mode(chk: Dict[str, Any]) -> str:
    """
    Infer whether expression means:
    - violation: True => trigger
    - validity:  True => pass
    """
    explicit = str(chk.get("predicate_mode", chk.get("expression_mode", ""))).strip().lower()
    if explicit in {"violation", "validity"}:
        return explicit

    expr = str(chk.get("expression", "")).strip()
    e = re.sub(r"\s+", "", expr.lower())

    # Strong violation-pattern signals.
    if e.startswith("not(") or e.startswith("not"):
        return "violation"
    if re.search(r"(<0|<=0|<=-1|==0|!=|isnone)", e):
        return "violation"

    # Strong validity-pattern signals.
    if re.search(r"(len\(.+\)>0|all\(|any\(|isinstance\()", e):
        return "validity"
    if re.search(r"(>=0|>0|>-1|<=1|<=10)", e):
        return "validity"
    if re.search(r"<(0?\.\d+|[1-9]\d*(?:\.\d+)?)", e) and "<0" not in e:
        return "validity"
    if ("<=" in e or ">=" in e) and "not" not in e:
        return "validity"

    # Fallback to violation semantics.
    return "violation"


def _is_check_triggered(
    chk: Dict[str, Any],
    inputs: Dict[str, Any],
    *,
    compute_fn: Optional[Any] = None,
) -> Tuple[bool, Optional[str]]:
    result, err = _safe_eval_rule(str(chk.get("expression", "")), inputs, compute_fn=compute_fn)
    if result is None:
        return False, err or "evaluation returned no result"
    mode = _infer_predicate_mode(chk)
    if mode == "validity":
        return result is False, None
    return result is True, None


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


def _load_core(path: Path) -> Dict[str, Dict[str, Any]]:
    rows = _load_records(path)
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        fic_id = str(r.get("fic_id", "")).strip()
        if fic_id:
            out[fic_id] = r
    if not out:
        raise ValueError("No valid core cards loaded")
    return out


def _load_retrieval(path: Path) -> List[Dict[str, Any]]:
    rows = _load_records(path)
    out = [r for r in rows if isinstance(r, dict) and str(r.get("fic_id", "")).strip()]
    if not out:
        raise ValueError("No valid retrieval cards loaded")
    return out


def _load_repair(path: Path) -> List[Dict[str, Any]]:
    rows = _load_records(path)
    out = [r for r in rows if isinstance(r, dict) and str(r.get("rule_id", "")).strip()]
    if not out:
        raise ValueError("No valid repair rules loaded")
    return out


@dataclass
class RetrievalCandidate:
    retrieval: Dict[str, Any]
    score: float


def _retrieval_text(card: Dict[str, Any]) -> str:
    pieces: List[str] = []
    for k in ("title", "summary", "domain", "topic", "embedding_text"):
        v = card.get(k)
        if isinstance(v, str):
            pieces.append(v)
    for k in ("selection_hints", "applicable_when", "not_applicable_when", "scope_boundaries", "keywords"):
        vals = card.get(k, [])
        if isinstance(vals, list):
            pieces.extend(str(x) for x in vals)
    for c in card.get("common_confusions", []):
        if isinstance(c, dict):
            pieces.append(str(c.get("label", "")))
            pieces.append(str(c.get("difference", "")))
    return " ".join(pieces)


def retrieve_candidates(cards: List[Dict[str, Any]], query: str, top_k: int) -> List[RetrievalCandidate]:
    q_tokens = set(_tokenize(query))
    scored: List[RetrievalCandidate] = []
    for card in cards:
        c_tokens = set(_tokenize(_retrieval_text(card)))
        overlap = len(q_tokens & c_tokens)
        denom = max(len(q_tokens), 1)
        score = overlap / denom
        scored.append(RetrievalCandidate(retrieval=card, score=score))
    scored.sort(key=lambda x: x.score, reverse=True)
    return scored[:top_k]


def retrieve_candidates_from_store(
    store: SQLAlchemyArtifactStore,
    *,
    query: str,
    top_k: int,
    domain: Optional[str] = None,
    topic: Optional[str] = None,
) -> List[RetrievalCandidate]:
    rows = store.retrieve_candidates(query=query, top_k=top_k, domain=domain, topic=topic)
    return [RetrievalCandidate(retrieval=row, score=float(row.get("score", 0.0))) for row in rows]


def _schema_selection() -> Any:
    return genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "decision": genai_types.Schema(type=genai_types.Type.STRING, enum=["select_card", "abstain_m", "abstain_n"]),
            "chosen_fic_id": genai_types.Schema(type=genai_types.Type.STRING),
            "reason": genai_types.Schema(type=genai_types.Type.STRING),
            "support_gap_reason": genai_types.Schema(type=genai_types.Type.STRING),
            "ambiguity_tags": genai_types.Schema(
                type=genai_types.Type.ARRAY,
                items=genai_types.Schema(type=genai_types.Type.STRING),
            ),
            "clarification_questions": genai_types.Schema(
                type=genai_types.Type.ARRAY,
                items=genai_types.Schema(type=genai_types.Type.STRING),
            ),
        },
        required=[
            "decision",
            "chosen_fic_id",
            "reason",
            "support_gap_reason",
            "ambiguity_tags",
            "clarification_questions",
        ],
    )


def _schema_extraction() -> Any:
    """Genai schema for the LLM extraction response.

    Each input item includes ``source_scale`` (required) so that the
    confidence-tracking logic in :func:`coerce_input` has an explicit signal
    about what the *source text* looked like, not just the converted value.
    """
    return genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "inputs": genai_types.Schema(
                type=genai_types.Type.ARRAY,
                items=genai_types.Schema(
                    type=genai_types.Type.OBJECT,
                    properties={
                        "name": genai_types.Schema(type=genai_types.Type.STRING),
                        "status": genai_types.Schema(
                            type=genai_types.Type.STRING,
                            enum=["provided", "missing"],
                        ),
                        "value": genai_types.Schema(type=genai_types.Type.STRING),
                        "source_scale": genai_types.Schema(
                            type=genai_types.Type.STRING,
                            enum=[
                                "explicit_percent",   # source text had a % sign
                                "explicit_decimal",   # source text was a 0.xx decimal
                                "explicit_integer",   # unambiguous integer, no %
                                "inferred",           # LLM had to guess
                            ],
                        ),
                    },
                    required=["name", "status", "value", "source_scale"],
                ),
            ),
            "normalization_note": genai_types.Schema(type=genai_types.Type.STRING),
        },
        required=["inputs", "normalization_note"],
    )


def _schema_critic_check() -> Any:
    return genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "needs_clarification": genai_types.Schema(type=genai_types.Type.BOOLEAN),
            "triggered_hint_ids": genai_types.Schema(
                type=genai_types.Type.ARRAY,
                items=genai_types.Schema(type=genai_types.Type.STRING),
            ),
            "ambiguity_tags": genai_types.Schema(
                type=genai_types.Type.ARRAY,
                items=genai_types.Schema(type=genai_types.Type.STRING),
            ),
            "clarification_questions": genai_types.Schema(
                type=genai_types.Type.ARRAY,
                items=genai_types.Schema(type=genai_types.Type.STRING),
            ),
            "clarification_options": genai_types.Schema(
                type=genai_types.Type.ARRAY,
                items=genai_types.Schema(type=genai_types.Type.STRING),
            ),
            "reason": genai_types.Schema(type=genai_types.Type.STRING),
        },
        required=[
            "needs_clarification",
            "triggered_hint_ids",
            "ambiguity_tags",
            "clarification_questions",
            "clarification_options",
            "reason",
        ],
    )


def _llm_json(client: Any, *, model: str, prompt: str, schema: Any) -> Dict[str, Any]:
    config = genai_types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=schema,
    )
    response = client.models.generate_content(model=model, contents=prompt, config=config)
    return json.loads(response.text)


def _select_card_with_llm(client: Any, model: str, question: str, context: str, cands: List[RetrievalCandidate]) -> Dict[str, Any]:
    block = []
    for i, c in enumerate(cands, 1):
        r = c.retrieval
        block.append(
            f"Candidate {i} | fic_id={r.get('fic_id')} | score={c.score:.3f}\n"
            f"title={r.get('title')}\n"
            f"topic={r.get('topic')}\n"
            f"summary={r.get('summary','')}\n"
            f"selection_hints={r.get('selection_hints', [])}\n"
            f"applicable_when={r.get('applicable_when', [])}\n"
            f"not_applicable_when={r.get('not_applicable_when', [])}\n"
            f"scope_boundaries={r.get('scope_boundaries', [])}"
        )
    prompt = f"""
You are selecting a financial formula card from candidates.

Decision policy — judge ONLY the user's INTENT (which metric they want), NOT the quality of the
input values. Apply in order:
1) decision="select_card" — if the QUESTION or the CONTEXT explicitly names a specific
   metric/formula (e.g. "NPV", "Fixed Asset Turnover ratio", "monthly loan payment", "CAGR",
   "working ratio") and a candidate matches that named metric. This INCLUDES a clarification the
   user added to the context (e.g. a note like "use NPV" or "the answer is: Net Present Value") —
   once the user has named the metric, the ambiguity is resolved, so SELECT that card; do NOT
   abstain_m again. SELECT IT even if the provided numbers look wrong, negative, out of range,
   missing, or otherwise anomalous — validating input values is handled by later deterministic F/E
   checks, NOT by you. Do not abstain because the data looks problematic.
2) decision="abstain_m" — ONLY when the user states a GENERAL GOAL without naming a metric
   (e.g. "is this project worth it", "will it earn money", "how did this investment perform") AND
   two or more candidates could each plausibly serve that goal. You MUST abstain_m here even if the
   variables perfectly fit one candidate; DO NOT guess the "most appropriate"/"best"/"industry-
   standard" metric or argue others are "less suitable" — that choice belongs to the user. Provide
   1-3 clarification_questions naming the competing metrics (e.g. "NPV, discounted payback, or ROI?").
3) decision="abstain_n" — the named metric/logic is genuinely NOT supported by ANY candidate (the
   library has no card for it), OR the user asks you to invent/derive a new formula. Provide
   support_gap_reason. A supported metric with bad input values is NOT abstain_n — select it.

Reminder: a named metric (even with bad data) → select_card. A vague goal with several fitting
metrics → abstain_m. Truly unsupported/invented metric → abstain_n.

Question:
{question}

Context:
{context}

Candidates:
{chr(10).join(block)}

Return JSON only.
"""
    return _llm_json(client, model=model, prompt=prompt, schema=_schema_selection())


def _extract_inputs_with_llm(client: Any, model: str, question: str, context: str, core: Dict[str, Any]) -> Dict[str, Any]:
    """Call the LLM to extract and coerce input values for the chosen FIC card.

    The prompt exposes the full input schema (name, type, unit, description,
    aliases) so the LLM can honour the declared type contract without any
    formula-specific hardcoding on our side.

    The LLM is also required to report ``source_scale`` for each value,
    indicating the format of the *original* source text (not the converted
    output).  This lets :func:`coerce_input` detect true ambiguity vs.
    explicit conversions, and surfaces FIC card unit-field errors.
    """
    required = core.get("inputs", [])
    prompt = f"""You are extracting structured input values for a financial formula.

For each required input, locate its value in the Question / Context and:
1. Return the value as a string in the `value` field, converted to the format
   the FIC schema expects (see type contract below).
2. Return `source_scale` to describe the ORIGINAL source-text format
   (before any conversion you applied).

Type contract for `value`:
- type "number" / "decimal" / "float": numeric string, e.g. "0.08".
- type "integer": whole-number string, e.g. "5".
- type "boolean": "true" or "false".
- type "array[number]": flat JSON array string, e.g. "[100, 200, 300]".
- type "array[array[number]]" or nested array types:
    JSON array-of-arrays, e.g. "[[50, 240], [80, 220]]".
    Do NOT flatten nested arrays into a single list.
- type "dict[string,number]": JSON object string with string keys and numeric
  values, e.g. "{{\"Sales\": 3000, \"R&D\": 5000}}".
- type "array[object]": JSON array of objects, e.g.
  "[{{\"quantity\": 100, \"price\": 50}}, {{\"quantity\": 50, \"price\": 48}}]".
- type "object": JSON object or array string matching the input description.
- unit "decimal" or "decimal_rate": convert % to decimal (15% → 0.15).
- unit "%" or "percent": keep as percentage points (15% → 15).

Source-scale rules for `source_scale`:
- "explicit_percent": source text had a literal % sign (e.g. "15%", "8 percent").
- "explicit_decimal": source text was already in decimal form (e.g. "0.08", "0.15").
- "explicit_integer": source text was an unambiguous integer with no % sign
  (e.g. "5 years", "120 units").
- "inferred": you had to infer or assume the scale from context — be honest.

If a value is absent, set status="missing", value="", source_scale="inferred".
Document every normalisation decision in normalization_note.

Required inputs (full schema):
{json.dumps(required, ensure_ascii=False, indent=2)}

Question:
{question}

Context:
{context}

Return JSON only.
"""
    if os.environ.get("VERIFIQUANT_STRICT_BINDING", "").lower() in ("1", "true", "yes"):
        prompt += (
            "\nSTRICT BINDING MODE (active):\n"
            "- Copy each value LITERALLY from the text. Do NOT sum, subtract, scale,\n"
            "  average, annualize, or otherwise derive values, even if the schema seems\n"
            "  to expect a derived quantity.\n"
            "- If the schema requires a value that is not literally present in the text,\n"
            "  set status=\"missing\" instead of computing it.\n"
            "- The only allowed conversions are the declared unit conversions above\n"
            "  (percent to decimal, stated scale words). Everything else belongs to the\n"
            "  deterministic contract code, not to you.\n"
        )
    return _llm_json(client, model=model, prompt=prompt, schema=_schema_extraction())


def _semantic_echeck_with_llm(
    client: Any,
    model: str,
    question: str,
    context: str,
    semantic_checks: List[Dict[str, Any]],
    provided_inputs: Dict[str, Any],
) -> Dict[str, Any]:
    # Backward compatibility shim for older callers.
    return _critic_check_with_llm(
        client=client,
        model=model,
        question=question,
        context=context,
        semantic_hints=semantic_checks,
        provided_inputs=provided_inputs,
    )


# ── Deterministic hint triggers (experimental; VQ_HINT_TRIGGERS=1 to enable) ──
# Moves declared-hint *recall* from LLM judgment into the deterministic layer:
# if a convention-signal phrase appears verbatim in the question/context AND the
# selected card declares a matching semantic hint, the hint is force-triggered at
# its declared i_level. The lexicon is system-level (uniform across all cards and
# questions); per-card machine-readable trigger_keywords are future stage_core work.
_HINT_TRIGGERS_ENABLED = str(os.environ.get("VQ_HINT_TRIGGERS", "")).strip().lower() in {"1", "true", "yes"}

# Gate ablation ladder (Wave 2-A): comma-separated gates to DISABLE, e.g. "MN,F,E,I".
# A disabled gate does not exit structurally; the pipeline proceeds as if that gate were absent.
# MN disabled -> selector abstain_m/n forced to select the top candidate; the m_min_top_score
# N-style confidence floor is also lifted. F/E/I disabled -> their structured exits are skipped.
_DISABLED_GATES = {g.strip().upper() for g in os.environ.get("VQ_DISABLE_GATES", "").split(",") if g.strip()}

_HINT_TRIGGER_LEXICON: List[Tuple[List[str], List[str]]] = [
    # (phrases searched in question+context, keywords required in hint id/options text)
    (["per share", "per-share"], ["per share", "unit"]),
    (["as a percentage", "as a percent", "in percent", "expressed as a percentage"], ["percent", "scale"]),
    (["as a decimal", "in decimal form"], ["decimal", "scale"]),
    (["increase in", "decrease in", "increased by", "decreased by", "grew by", "rose by", "fell by"], ["change", "convention"]),
    (["beginning of each", "at the beginning of"], ["begin", "start", "timing"]),
    (["end of each", "at the end of"], ["end", "timing"]),
    (["pre-tax", "before tax", "before-tax"], ["tax"]),
    (["after-tax", "after tax"], ["tax"]),
]


def _lexical_hint_triggers(
    question: str,
    context: str,
    semantic_hints: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Deterministically match declared semantic hints against the case text."""
    text = f"{question}\n{context}".lower()
    hits: List[Dict[str, Any]] = []
    for hint in semantic_hints:
        hid = str(hint.get("id", "")).strip()
        if not hid:
            continue
        hint_blob = " ".join([
            hid,
            json.dumps(hint.get("options", []) or [], ensure_ascii=False),
            str(hint.get("clarification_question", "")),
        ]).lower()
        for phrases, hint_keys in _HINT_TRIGGER_LEXICON:
            phrase = next((p for p in phrases if p in text), None)
            if phrase is None:
                continue
            if any(k in hint_blob for k in hint_keys):
                hits.append({
                    "hint_id": hid,
                    "matched_phrase": phrase,
                    "i_level": _semantic_hint_level(hint),
                })
                break
    return hits


def _critic_check_with_llm(
    client: Any,
    model: str,
    question: str,
    context: str,
    semantic_hints: List[Dict[str, Any]],
    provided_inputs: Dict[str, Any],
) -> Dict[str, Any]:
    has_hints = bool(semantic_hints)
    mode_desc = "hint_oriented" if has_hints else "open"
    mode_rules = (
        "- semantic_hints are available: use them as primary reference and fill triggered_hint_ids when relevant.\n"
        "- If a hint is not relevant, do not trigger it.\n"
    ) if has_hints else (
        "- semantic_hints are empty: perform OPEN semantic ambiguity scan from question/context.\n"
        "- In open mode, triggered_hint_ids should be empty.\n"
        "- If clarification is needed, propose concise clarification_options when possible.\n"
    )
    prompt = f"""
You are the VerifiQuant I-gate Critic Agent.
Goal: detect hidden semantic ambiguity before deterministic execution.

Ambiguity examples:
- FX direction (USD/TWD vs TWD/USD)
- time basis (begin-year vs end-year cash flow)
- compounding convention (monthly vs annual)

Rules:
- If ambiguity materially changes calculation interpretation, set needs_clarification=true.
- Provide concise clarification questions.
- Mode: {mode_desc}
{mode_rules}

Question:
{question}

Context:
{context}

Semantic hints:
{json.dumps(semantic_hints, ensure_ascii=False, indent=2)}

Provided inputs:
{json.dumps(provided_inputs, ensure_ascii=False, indent=2)}

Return JSON only.
"""
    return _llm_json(client, model=model, prompt=prompt, schema=_schema_critic_check())


def _evaluate_execution(
    core: Dict[str, Any],
    provided_inputs: Dict[str, Any],
) -> Tuple[Optional[float], Optional[str], Dict[str, Any]]:
    def _json_safe(value: Any) -> Any:
        try:
            json.dumps(value, ensure_ascii=False)
            return value
        except Exception:
            return repr(value)

    execution = core.get("execution", {})
    code = str(execution.get("code", ""))
    output_name = str((core.get("output") or {}).get("name", ""))
    trace: Dict[str, Any] = {
        "engine": "deterministic_python",
        "entrypoint": "compute",
        "code": code,
        "inputs": _json_safe(provided_inputs),
        "output_name": output_name,
    }
    if "def compute(inputs)" not in code:
        trace["error"] = "execution code missing compute(inputs)"
        return None, "execution code missing compute(inputs)", trace
    if not output_name:
        trace["error"] = "output.name missing in core"
        return None, "output.name missing in core", trace

    safe_builtins = {
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "all": all,
        "any": any,
        "isinstance": isinstance,
        "bool": bool,
        "str": str,
        "float": float,
        "int": int,
        "pow": pow,
        "round": round,
        "range": range,
        "list": list,
        "dict": dict,
        "tuple": tuple,
        "set": set,
        "zip": zip,
        "enumerate": enumerate,
        "sorted": sorted,
        "ValueError": ValueError,
        "TypeError": TypeError,
        "Exception": Exception,
        "__import__": safe_fic_import,
        "math": math,
    }
    local_env: Dict[str, Any] = {"__builtins__": safe_builtins}
    try:
        exec(code, local_env, local_env)
        fn = local_env.get("compute")
        if not callable(fn):
            trace["error"] = "compute function missing after exec"
            return None, "compute function missing after exec", trace
        result = fn(provided_inputs)
        trace["raw_result"] = _json_safe(result)
        if isinstance(result, dict):
            if output_name not in result:
                err = f"output '{output_name}' missing from compute result"
                trace["error"] = err
                return None, err, trace
            raw_output = result.get(output_name)
        else:
            # Backward compatibility: some generated cards return scalar directly.
            raw_output = result
        val = _parse_number(raw_output)
        if val is None:
            trace["raw_output"] = _json_safe(raw_output)
            trace["error"] = "output value not numeric"
            return None, "output value not numeric", trace
        trace["raw_output"] = _json_safe(raw_output)
        trace["parsed_output_value"] = val
        return val, None, trace
    except Exception as exc:
        trace["error"] = str(exc)
        return None, str(exc), trace


def _build_repair_index(repair_rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in repair_rows:
        fic_id = str(row.get("fic_id", "")).strip()
        rid = str(row.get("rule_id", "")).strip()
        if fic_id and rid:
            out[(fic_id, rid)] = row
    return out


def _summarize_repairs(fic_id: str, rule_ids: List[str], repair_index: Dict[Tuple[str, str], Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for rid in rule_ids:
        rule = repair_index.get((fic_id, rid))
        if not rule:
            rule = repair_index.get((GLOBAL_SCOPE_FIC_ID, rid))
        if not rule:
            continue
        out.append(
            {
                "rule_id": rid,
                "title": rule.get("title", ""),
                "user_message": rule.get("user_message", ""),
                "repair_action": rule.get("repair_action", {}),
                "allowed_next_steps": rule.get("allowed_next_steps", []),
                "ask_user_for": rule.get("ask_user_for", []),
            }
        )
    return out


def _default_scope_repair_hint() -> List[Dict[str, Any]]:
    return [
        {
            "rule_id": str(GLOBAL_N_NOT_SUPPORTED_RULE.get("rule_id", "global_n_not_supported")),
            "title": str(GLOBAL_N_NOT_SUPPORTED_RULE.get("title", "")),
            "user_message": str(GLOBAL_N_NOT_SUPPORTED_RULE.get("user_message", "")),
            "repair_action": GLOBAL_N_NOT_SUPPORTED_RULE.get("repair_action", {}),
            "allowed_next_steps": GLOBAL_N_NOT_SUPPORTED_RULE.get("allowed_next_steps", []),
            "ask_user_for": GLOBAL_N_NOT_SUPPORTED_RULE.get("ask_user_for", []),
        }
    ]


def _i_rule_id_from_hint_id(hint_id: str, i_level: str = "soft") -> str:
    token = re.sub(r"[^a-zA-Z0-9_]+", "_", str(hint_id or "").strip().lower()).strip("_")
    level = str(i_level or "soft").strip().lower()
    if level not in {"hard", "soft"}:
        level = "soft"
    return f"i_{token or 'semantic_ambiguity'}_{level}"


def _semantic_hint_level(hint: Dict[str, Any]) -> str:
    level = str(hint.get("i_level", "soft")).strip().lower()
    if level not in {"hard", "soft"}:
        return "soft"
    return level


def _infer_open_critic_i_level(critic: Dict[str, Any]) -> str:
    reason = str(critic.get("reason", "") or "").lower()
    tags = [str(x).strip().lower() for x in critic.get("ambiguity_tags", []) if str(x).strip()]
    hard_signals = [
        "formula",
        "definition",
        "direction",
        "basis",
        "unit",
        "scale",
        "numerator",
        "denominator",
        "fx",
        "currency",
    ]
    if any(sig in reason for sig in hard_signals):
        return "hard"
    if any(any(sig in tag for sig in hard_signals) for tag in tags):
        return "hard"
    if "materially" in reason or "significantly different" in reason:
        return "hard"
    return "soft"


def _normalize_warning_options(raw_options: Any) -> List[Dict[str, Any]]:
    """Normalize semantic-hint options to structured dicts the UI can act on.

    Hint options may be dicts ({label, value, is_default, transform_spec?}) or plain
    strings. Emit a uniform shape carrying `has_transform` so the frontend can offer a
    verifiable atomic transform for options that declare one.
    """
    out: List[Dict[str, Any]] = []
    for opt in (raw_options or []):
        if isinstance(opt, dict):
            label = str(opt.get("label", opt.get("value", ""))).strip()
            value = str(opt.get("value", opt.get("label", ""))).strip()
            if not (label or value):
                continue
            out.append(
                {
                    "label": label or value,
                    "value": value or label,
                    "is_default": bool(opt.get("is_default", False)),
                    "has_transform": bool(opt.get("transform_spec")),
                }
            )
        else:
            text = str(opt).strip()
            if not text:
                continue
            out.append({"label": text, "value": text, "is_default": False, "has_transform": False})
    return out


def _build_soft_warnings(
    *,
    critic: Dict[str, Any],
    semantic_hints: List[Dict[str, Any]],
    triggered_hint_ids: List[str],
) -> List[Dict[str, Any]]:
    hint_by_id = {str(h.get("id", "")).strip(): h for h in semantic_hints if str(h.get("id", "")).strip()}
    warnings: List[Dict[str, Any]] = []

    if triggered_hint_ids:
        for hid in triggered_hint_ids:
            hint = hint_by_id.get(hid)
            if not hint:
                continue
            if _semantic_hint_level(hint) != "soft":
                continue
            warnings.append(
                {
                    "hint_id": hid,
                    "i_level": "soft",
                    "impact_scope": str(hint.get("impact_scope", "output_interpretation")),
                    "assumption_if_not_clarified": str(
                        hint.get("assumption_if_not_clarified", "Proceed with default interpretation.")
                    ),
                    "clarification_question": str(hint.get("clarification_question", "")),
                    "options": _normalize_warning_options(hint.get("options")),
                }
            )
        return warnings

    # open-mode soft fallback
    warnings.append(
        {
            "hint_id": "open_critic",
            "i_level": "soft",
            "impact_scope": "output_interpretation",
            "assumption_if_not_clarified": "Proceed with default interpretation inferred from context.",
            "clarification_question": "; ".join(
                [str(x).strip() for x in critic.get("clarification_questions", []) if str(x).strip()]
            ),
            "options": _normalize_warning_options(critic.get("clarification_options")),
        }
    )
    return warnings


def _best_scope_repair(
    candidate_ids: List[Any],
    repair_index: Dict[Tuple[str, str], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    for cid in candidate_ids:
        fic_id = str(cid or "").strip()
        if not fic_id:
            continue
        rows = _summarize_repairs(fic_id, ["global_n_not_supported"], repair_index)
        if rows:
            return rows
    rows = _summarize_repairs(GLOBAL_SCOPE_FIC_ID, ["global_n_not_supported"], repair_index)
    if rows:
        return rows
    return _default_scope_repair_hint()


def run_case(
    *,
    row: Dict[str, Any],
    core_by_id: Dict[str, Dict[str, Any]],
    retrieval_cards: List[Dict[str, Any]],
    store: Optional[SQLAlchemyArtifactStore],
    repair_index: Dict[Tuple[str, str], Dict[str, Any]],
    client: Any,
    selector_model: str,
    extractor_model: str,
    judge_model: str,
    top_k: int,
    m_min_top_score: float,
    debug_sanity: bool = False,
) -> Dict[str, Any]:
    q = str(row.get("question", "") or "")
    c = str(row.get("context", "") or "")
    domain = str(row.get("domain", "") or "").strip() or None
    topic = str(row.get("topic", "") or "").strip() or None
    case_id = str(row.get("case_id") or row.get("question_id") or row.get("id") or "")
    base = {"case_id": case_id, "has_i_soft": False, "soft_warnings": []}
    debug_prefix = f"[sanity-debug][{case_id or 'unknown_case'}]"
    pipeline_logs: List[str] = []
    step_order = [
        ("rag_cards", "1) RAG 卡片檢索"),
        ("mn_analysis", "2) M/N 意圖與卡片匹配分析"),
        ("extract_values", "3) LLM 抽取數值"),
        ("fe_checks", "4) F/E 規則檢查"),
        ("ihard_check", "5) I_hard 假設檢查"),
        ("python_execution", "6) Python 計算"),
        ("final_response", "7) 回傳結果與 I_soft 提醒"),
    ]
    timeline: Dict[str, Dict[str, Any]] = {
        key: {
            "key": key,
            "label": label,
            "status": "pending",
            "detail": "",
        }
        for key, label in step_order
    }
    selection_trace: Dict[str, Any] = {}
    extraction_trace: Dict[str, Any] = {}
    critic_trace: Dict[str, Any] = {}
    echeck_trace: Dict[str, Any] = {}
    execution_trace: Dict[str, Any] = {}

    def _set_step(step_key: str, status: str, detail: str = "") -> None:
        if step_key in timeline:
            timeline[step_key]["status"] = status
            timeline[step_key]["detail"] = detail

    def _finalize(payload: Dict[str, Any]) -> Dict[str, Any]:
        for step in timeline.values():
            if step["status"] == "pending":
                step["status"] = "skipped"
                step["detail"] = step["detail"] or "Not reached in this run."
        payload["pipeline_logs"] = list(pipeline_logs)
        payload["pipeline_timeline"] = [timeline[key] for key, _ in step_order]
        payload["selection_trace"] = selection_trace
        payload["extraction_trace"] = extraction_trace
        payload["echeck_trace"] = echeck_trace
        payload["critic_trace"] = critic_trace
        payload["execution_trace"] = execution_trace
        return payload

    def _dbg(message: str) -> None:
        pipeline_logs.append(message)
        if debug_sanity:
            print(f"{debug_prefix} {message}")

    _set_step("rag_cards", "running", "Retrieving candidate cards from artifact store.")
    if store is not None:
        candidates = retrieve_candidates_from_store(
            store,
            query=f"{q}\n{c}",
            top_k=top_k,
            domain=domain,
            topic=topic,
        )
    else:
        candidates = retrieve_candidates(retrieval_cards, f"{q}\n{c}", top_k=top_k)
    if candidates:
        top_rows = [
            {
                "fic_id": str(cand.retrieval.get("fic_id", "")).strip(),
                "score": round(float(cand.score), 6),
                "title": cand.retrieval.get("title"),
                "topic": cand.retrieval.get("topic"),
                "domain": cand.retrieval.get("domain"),
            }
            for cand in candidates
        ]
        selection_trace["retrieval_candidates"] = top_rows
        _set_step("rag_cards", "success", f"Retrieved {len(candidates)} cards.")
    else:
        _set_step("rag_cards", "error", "No candidate cards retrieved.")
    _dbg(
        "retrieval: "
        + ", ".join(
            f"{str(cand.retrieval.get('fic_id', '')).strip() or '<missing_fic_id>'}@{cand.score:.3f}"
            for cand in candidates
        )
    )
    if not candidates:
        _dbg("early-exit: N (no candidate cards retrieved)")
        _set_step("mn_analysis", "skipped", "No cards available for M/N analysis.")
        _set_step("extract_values", "skipped", "No selected FIC card.")
        _set_step("fe_checks", "skipped", "No extracted inputs to check.")
        _set_step("ihard_check", "skipped", "No extracted inputs to review.")
        _set_step("python_execution", "skipped", "No selected FIC execution path.")
        _set_step("final_response", "success", "Returned N refusal response.")
        return _finalize({
            **base,
            "status": "refusal",
            "diagnostic_type": "N",
            "funnel_layer": "Scope",
            "gate_action": "graceful_exit",
            "reason": "No candidate cards retrieved",
            "support_gap_reason": "no_candidate_cards",
            "candidate_ids": [],
            "ambiguity_tags": [],
            "clarification_request": None,
            "repair_hints": [],
        })

    _set_step("mn_analysis", "running", "Selector is deciding M/N/select_card.")
    selection = _select_card_with_llm(client, selector_model, q, c, candidates)
    decision = str(selection.get("decision", "")).strip()
    chosen_fic_id = str(selection.get("chosen_fic_id", "")).strip()
    candidate_ids = [cand.retrieval.get("fic_id") for cand in candidates]
    support_gap_reason = str(selection.get("support_gap_reason", "")).strip()
    ambiguity_tags = [str(x).strip() for x in selection.get("ambiguity_tags", []) if str(x).strip()]
    clarification_questions = [str(x).strip() for x in selection.get("clarification_questions", []) if str(x).strip()]
    selection_trace["selector"] = {
        "decision": decision,
        "chosen_fic_id": chosen_fic_id,
        "reason": str(selection.get("reason", "")).strip(),
        "support_gap_reason": support_gap_reason,
        "ambiguity_tags": ambiguity_tags,
        "clarification_questions": clarification_questions,
    }
    _dbg(
        "selector: "
        f"decision={decision or '<empty>'}, chosen_fic_id={chosen_fic_id or '<empty>'}, "
        f"reason={str(selection.get('reason', '')).strip() or '<empty>'}"
    )

    if "MN" in _DISABLED_GATES and (decision != "select_card" or not chosen_fic_id):
        # Ablation: no M/N gate -> commit to the top-scoring candidate instead of abstaining.
        forced = next((str(cid) for cid in candidate_ids if str(cid).strip()), "")
        if forced:
            _dbg(f"gate-ablation MN disabled: forcing select_card={forced}")
            decision = "select_card"
            chosen_fic_id = forced

    if decision == "abstain_m":
        _dbg("early-exit: M (selector abstain_m)")
        _set_step("mn_analysis", "error", "M ambiguity detected by selector.")
        _set_step("extract_values", "skipped", "No card selected.")
        _set_step("fe_checks", "skipped", "No extracted inputs to check.")
        _set_step("ihard_check", "skipped", "No extracted inputs to review.")
        _set_step("python_execution", "skipped", "No selected FIC execution path.")
        _set_step("final_response", "success", "Returned M clarification/refusal response.")
        return _finalize({
            **base,
            "status": "refusal",
            "diagnostic_type": "M",
            "funnel_layer": "Intent",
            "gate_action": "refusal",
            "reason": selection.get("reason", "Ambiguous task intent before card commitment"),
            "candidate_ids": candidate_ids,
            "support_gap_reason": None,
            "ambiguity_tags": ambiguity_tags,
            "clarification_request": {
                "questions": clarification_questions
                or ["Please specify the target metric (for example NPV, IRR, or Payback)."],
                "options": [],
            },
        })

    if decision == "abstain_n":
        _dbg("early-exit: N (selector abstain_n)")
        _set_step("mn_analysis", "error", "N out-of-scope detected by selector.")
        _set_step("extract_values", "skipped", "No card selected.")
        _set_step("fe_checks", "skipped", "No extracted inputs to check.")
        _set_step("ihard_check", "skipped", "No extracted inputs to review.")
        _set_step("python_execution", "skipped", "No selected FIC execution path.")
        _set_step("final_response", "success", "Returned N graceful-exit response.")
        return _finalize({
            **base,
            "status": "refusal",
            "diagnostic_type": "N",
            "funnel_layer": "Scope",
            "gate_action": "graceful_exit",
            "reason": selection.get("reason", "Requested intent is out of supported scope"),
            "candidate_ids": candidate_ids,
            "support_gap_reason": support_gap_reason or "unsupported_formula_family",
            "ambiguity_tags": ambiguity_tags,
            "clarification_request": None,
            "repair_hints": _best_scope_repair(candidate_ids, repair_index),
        })

    if decision != "select_card" or not chosen_fic_id:
        _dbg("early-exit: M (selector did not commit valid card)")
        _set_step("mn_analysis", "error", "Selector did not commit a valid card.")
        _set_step("extract_values", "skipped", "No card selected.")
        _set_step("fe_checks", "skipped", "No extracted inputs to check.")
        _set_step("ihard_check", "skipped", "No extracted inputs to review.")
        _set_step("python_execution", "skipped", "No selected FIC execution path.")
        _set_step("final_response", "success", "Returned M clarification/refusal response.")
        return _finalize({
            **base,
            "status": "refusal",
            "diagnostic_type": "M",
            "funnel_layer": "Intent",
            "gate_action": "refusal",
            "reason": "Selector did not commit a valid card decision",
            "candidate_ids": candidate_ids,
            "support_gap_reason": None,
            "ambiguity_tags": ambiguity_tags,
            "clarification_request": {
                "questions": ["Please specify the target metric explicitly (for example NPV, IRR, Payback)."],
                "options": [],
            },
        })

    top_score = candidates[0].score if candidates else 0.0
    _dbg(f"retrieval-confidence: top_score={top_score:.3f}, threshold={m_min_top_score:.3f}")
    if top_score < m_min_top_score and "MN" not in _DISABLED_GATES:
        _dbg("early-exit: N (low retrieval confidence)")
        _set_step("mn_analysis", "error", "Retrieval confidence below threshold.")
        _set_step("extract_values", "skipped", "No trusted card selected.")
        _set_step("fe_checks", "skipped", "No extracted inputs to check.")
        _set_step("ihard_check", "skipped", "No extracted inputs to review.")
        _set_step("python_execution", "skipped", "No selected FIC execution path.")
        _set_step("final_response", "success", "Returned N low-confidence response.")
        return _finalize({
            **base,
            "status": "refusal",
            "diagnostic_type": "N",
            "funnel_layer": "Scope",
            "gate_action": "graceful_exit",
            "reason": f"Low retrieval confidence before card commitment (top score={top_score:.3f})",
            "candidate_ids": candidate_ids,
            "support_gap_reason": "low_retrieval_confidence",
            "ambiguity_tags": ambiguity_tags,
            "clarification_request": None,
            "repair_hints": _best_scope_repair(candidate_ids, repair_index),
        })

    core = core_by_id.get(chosen_fic_id)
    if core is None:
        _dbg(f"early-exit: N (missing core card for chosen_fic_id={chosen_fic_id})")
        _set_step("mn_analysis", "error", "Chosen card has no core artifact.")
        _set_step("extract_values", "skipped", "No core card for extraction.")
        _set_step("fe_checks", "skipped", "No extracted inputs to check.")
        _set_step("ihard_check", "skipped", "No extracted inputs to review.")
        _set_step("python_execution", "skipped", "No selected FIC execution path.")
        _set_step("final_response", "success", "Returned N missing-core response.")
        return _finalize({
            **base,
            "status": "refusal",
            "diagnostic_type": "N",
            "funnel_layer": "Scope",
            "gate_action": "graceful_exit",
            "reason": f"Chosen fic_id '{chosen_fic_id}' has no matching core card",
            "candidate_ids": candidate_ids,
            "support_gap_reason": "missing_core_card",
            "ambiguity_tags": ambiguity_tags,
            "clarification_request": None,
            "repair_hints": _best_scope_repair(candidate_ids, repair_index),
        })

    _set_step("mn_analysis", "success", f"Selected fic_id={chosen_fic_id}.")
    _set_step("extract_values", "running", "LLM extracting and normalizing input fields.")

    # card_committed=True beyond this point; only F/E/I/C/success should appear.
    extraction = _extract_inputs_with_llm(client, extractor_model, q, c, core)
    normalization_note = str(extraction.get("normalization_note", "") or "")

    core_inputs = core.get("inputs", [])
    type_map = {str(inp.get("name")): str(inp.get("type", "number")) for inp in core_inputs}
    unit_map = {str(inp.get("name")): inp.get("unit") for inp in core_inputs}
    desc_map = {str(inp.get("name")): str(inp.get("description", "")) for inp in core_inputs}
    required_names = [str(inp.get("name")) for inp in core_inputs if bool(inp.get("required", True))]
    extracted = {str(x.get("name")): x for x in extraction.get("inputs", []) if x.get("name")}

    missing = []
    provided_inputs: Dict[str, Any] = {}
    ambiguous_fields: List[str] = []       # fields whose coercion confidence == "ambiguous"
    coercion_notes: List[str] = []         # per-field coercion log (merged into normalization_note)
    schema_contradiction_fields: List[str] = []  # fields where FIC unit clashes with source

    def _bind_field(name: str, item: Dict[str, Any]) -> Optional[Any]:
        """Coerce one extracted field; side-effects: populate ambiguous_fields, coercion_notes."""
        result = coerce_input(
            item.get("value", ""),
            type_map.get(name, "number"),
            unit=unit_map.get(name),
            description=desc_map.get(name, ""),
            source_scale=str(item.get("source_scale") or "inferred"),
        )
        if result.note:
            coercion_notes.append(f"[{name}] {result.note}")
        if result.confidence == "ambiguous":
            ambiguous_fields.append(name)
            # Detect the specific case: FIC unit says percent_point but value is decimal.
            schema_scale = _resolve_schema_scale(unit_map.get(name), desc_map.get(name, ""))
            ss = str(item.get("source_scale") or "inferred")
            if schema_scale == "percent_point" and ss == "explicit_decimal":
                schema_contradiction_fields.append(name)
        return result.value

    for name in required_names:
        item = extracted.get(name)
        if not item or item.get("status") != "provided":
            missing.append(name)
            continue
        val = _bind_field(name, item)
        if val is None:
            missing.append(name)
        else:
            provided_inputs[name] = val

    for name, item in extracted.items():
        if name in provided_inputs:
            continue
        val = _bind_field(name, item)
        if val is not None:
            provided_inputs[name] = val

    # Optional literal-grounding guard (anti-over-normalization; see strict_binding.py).
    # Ungrounded REQUIRED values are downgraded to missing -> the F gate requests them,
    # converting a potential silent binding slip into a structured ask. Opt-in.
    if os.environ.get("VERIFIQUANT_STRICT_BINDING", "").lower() in ("1", "true", "yes"):
        from verifiquant.pipeline.strict_binding import audit_binding
        _, _ungrounded = audit_binding(provided_inputs, q, c)
        for _n in _ungrounded:
            if _n in required_names:
                provided_inputs.pop(_n, None)
                if _n not in missing:
                    missing.append(_n)
                coercion_notes.append(
                    f"[{_n}] strict-binding: value not traceable to a literal context span; "
                    "requesting the field instead of trusting a derived value."
                )

    extraction_trace = {
        "required_names": required_names,
        "llm_extracted_inputs": extraction.get("inputs", []),
        "normalization_note": normalization_note,
        "missing_fields": missing,
        "provided_keys": sorted(list(provided_inputs.keys())),
        "ambiguous_fields": ambiguous_fields,
        "schema_contradiction_fields": schema_contradiction_fields,
        "coercion_notes": coercion_notes,
    }

    _dbg(
        "input-binding: "
        f"required={required_names}, missing={missing}, provided_keys={sorted(list(provided_inputs.keys()))}, "
        f"ambiguous={ambiguous_fields}, schema_contradictions={schema_contradiction_fields}"
    )

    if missing and "F" not in _DISABLED_GATES:
        _dbg("early-exit: F (missing or unparsable required inputs)")
        _set_step("extract_values", "error", f"Missing required fields: {', '.join(missing)}")
        _set_step("fe_checks", "skipped", "Cannot run checks before required fields are filled.")
        _set_step("ihard_check", "skipped", "Cannot run critic before required fields are filled.")
        _set_step("python_execution", "skipped", "Cannot execute compute() with missing required fields.")
        _set_step("final_response", "success", "Returned F slot-filling response.")
        return _finalize({
            **base,
            "status": "error",
            "diagnostic_type": "F",
            "funnel_layer": "Schema",
            "gate_action": "slot_filling",
            "reason": "Missing or unparsable required inputs",
            "fic_id": chosen_fic_id,
            "candidate_ids": candidate_ids,
            "support_gap_reason": None,
            "ambiguity_tags": [],
            "clarification_request": None,
            "requested_fields": missing,
            "provided_inputs": provided_inputs,
            "normalization_note": normalization_note,
            "repair_hints": [
                {
                    "repair_action": {"type": "request_missing_fields", "target": ",".join(missing)},
                    "ask_user_for": [{"slot": m, "label": m, "type": "text", "required": True, "options": []} for m in missing],
                    "allowed_next_steps": ["ask_followup", "rerun_same_fic"],
                }
            ],
        })

    _set_step("extract_values", "success", f"Extracted {len(provided_inputs)} input fields.")
    _set_step("fe_checks", "running", "Running deterministic F/E checks.")


    # Pre-compile the FIC's compute function so E-check expressions that call
    # ``compute({...})`` (e.g. plausibility range checks) can be evaluated.
    _fic_compute_fn: Optional[Any] = None
    _fic_code = str((core.get("execution") or {}).get("code", ""))
    if "def compute(inputs)" in _fic_code:
        _safe_builtins_exec = {
            "abs": abs, "min": min, "max": max, "sum": sum, "len": len,
            "all": all, "any": any, "isinstance": isinstance, "bool": bool,
            "str": str, "float": float, "int": int, "pow": pow, "round": round,
            "range": range, "list": list, "dict": dict, "tuple": tuple,
            "set": set, "zip": zip, "enumerate": enumerate,
            "sorted": sorted,
            "ValueError": ValueError, "TypeError": TypeError, "Exception": Exception,
            "__import__": safe_fic_import,
            "math": math,
        }
        try:
            _exec_env: Dict[str, Any] = {"__builtins__": _safe_builtins_exec}
            exec(_fic_code, _exec_env, _exec_env)  # noqa: S102
            _fic_compute_fn = _exec_env.get("compute")
        except Exception:
            pass  # SyntaxError or other issue — will surface at C-execution stage

    # Classify E-check eval errors: structural TypeError/AttributeError →  F-class;
    # everything else (NameError, SyntaxError, genuine logic errors) → C-class.
    _STRUCTURAL_ERROR_SIGNALS = (
        "is not subscriptable",
        "has no attribute",
        "is not iterable",
        "object is not callable",
        "cannot unpack",
        "must be",          # e.g. 'int' object must be ...
    )

    checks = core.get("diagnostic_checks", [])
    e_checks = [chk for chk in checks if str(chk.get("diagnostic_type", "")).upper() == "E"]
    auto_triggered: List[str] = []
    eval_errors: List[Dict[str, str]] = []        # genuine C-class issues
    structural_errors: List[Dict[str, str]] = []  # input-shape issues → F-class

    for chk in e_checks:
        rule_id = str(chk.get("rule_id", "")).strip()
        ctype = str(chk.get("check_type", "")).strip().lower()
        if ctype in {"deterministic", "normalization"}:
            is_triggered, eval_err = _is_check_triggered(chk, provided_inputs, compute_fn=_fic_compute_fn)
            if eval_err:
                err_record = {
                    "rule_id": rule_id or "<unknown>",
                    "expression": str(chk.get("expression", "")),
                    "error": eval_err,
                }
                # Decide: is this a structural (F) or logic (C) failure?
                if any(sig in eval_err for sig in _STRUCTURAL_ERROR_SIGNALS):
                    structural_errors.append(err_record)
                    _dbg(
                        f"e-check: rule_id={rule_id or '<unknown>'}, triggered=<structural-error[F]>, "
                        f"error={eval_err}"
                    )
                else:
                    eval_errors.append(err_record)
                    _dbg(
                        f"e-check: rule_id={rule_id or '<unknown>'}, triggered=<error[C]>, "
                        f"error={eval_err}"
                    )
                continue
            if is_triggered and rule_id:
                auto_triggered.append(rule_id)
            _dbg(
                f"e-check: rule_id={rule_id or '<unknown>'}, mode={_infer_predicate_mode(chk)}, "
                f"triggered={is_triggered}"
            )

    echeck_trace = {
        "total_checks": len(checks),
        "e_checks_count": len(e_checks),
        "auto_triggered_rule_ids": list(dict.fromkeys(auto_triggered)),
        "structural_errors": structural_errors,
        "evaluation_errors": eval_errors,
    }

    # Structural errors mean the input shape does not match the FIC schema → F.
    if structural_errors:
        first_s = structural_errors[0]
        missing_structural = [e["rule_id"] for e in structural_errors]
        _dbg(f"early-exit: F (structural E-check error → input shape mismatch on {first_s['rule_id']})")
        _set_step("fe_checks", "error", f"F structural mismatch in {first_s['rule_id']}.")
        _set_step("ihard_check", "skipped", "Blocked by F structural error.")
        _set_step("python_execution", "skipped", "Blocked by F structural error.")
        _set_step("final_response", "success", "Returned F slot-filling response.")
        return _finalize({
            **base,
            "status": "error",
            "diagnostic_type": "F",
            "funnel_layer": "Schema",
            "gate_action": "slot_filling",
            "reason": (
                f"Input structure mismatch detected in rule {first_s['rule_id']}: {first_s['error']}. "
                "One or more inputs have an unexpected type or shape (e.g., a flat list where a "
                "list-of-lists was expected)."
            ),
            "fic_id": chosen_fic_id,
            "candidate_ids": candidate_ids,
            "support_gap_reason": None,
            "ambiguity_tags": ["input_structure_mismatch"],
            "clarification_request": None,
            "provided_inputs": provided_inputs,
            "normalization_note": normalization_note,
            "rule_eval_errors": structural_errors,
            "repair_hints": _summarize_repairs(chosen_fic_id, missing_structural, repair_index),
        })

    if eval_errors:
        first = eval_errors[0]
        _dbg(f"early-exit: C (E-check evaluation error on {first['rule_id']})")
        _set_step("fe_checks", "error", f"C evaluation error in {first['rule_id']}.")
        _set_step("ihard_check", "skipped", "Blocked by E-check evaluation error.")
        _set_step("python_execution", "skipped", "Blocked by E-check evaluation error.")
        _set_step("final_response", "success", "Returned C audit-log response.")
        return _finalize({
            **base,
            "status": "error",
            "diagnostic_type": "C",
            "funnel_layer": "Logic",
            "gate_action": "audit_log",
            "reason": f"E-check evaluation error in rule {first['rule_id']}: {first['error']}",
            "fic_id": chosen_fic_id,
            "candidate_ids": candidate_ids,
            "support_gap_reason": None,
            "ambiguity_tags": [],
            "clarification_request": None,
            "provided_inputs": provided_inputs,
            "normalization_note": normalization_note,
            "rule_eval_errors": eval_errors,
        })

    triggered = list(dict.fromkeys(auto_triggered))
    if triggered and "E" not in _DISABLED_GATES:
        _dbg(f"early-exit: E (triggered_rule_ids={triggered})")
        reason = "Deterministic E-type checks detected potential inconsistencies."
        _set_step("fe_checks", "alert", f"E rules triggered: {', '.join(triggered)}")
        _set_step("ihard_check", "skipped", "Execution blocked by deterministic E alert.")
        _set_step("python_execution", "skipped", "Execution blocked by deterministic E alert.")
        _set_step("final_response", "success", "Returned E deterministic-alert response.")
        return _finalize({
            **base,
            "status": "alert",
            "diagnostic_type": "E",
            "funnel_layer": "Boundary",
            "gate_action": "deterministic_alert",
            "reason": reason or "Potential E-type inconsistency detected",
            "fic_id": chosen_fic_id,
            "candidate_ids": candidate_ids,
            "support_gap_reason": None,
            "ambiguity_tags": [],
            "clarification_request": None,
            "provided_inputs": provided_inputs,
            "triggered_rule_ids": triggered,
            "normalization_note": normalization_note,
            "repair_hints": _summarize_repairs(chosen_fic_id, triggered, repair_index),
        })

    _set_step("fe_checks", "success", "F/E checks passed.")
    _set_step("ihard_check", "running", "Running I-gate critic for hidden assumptions.")

    semantic_hints = [h for h in (core.get("semantic_hints") or []) if isinstance(h, dict)]
    if "I" in _DISABLED_GATES:
        # Ablation: no I gate -> never block or warn on semantic ambiguity; proceed to compute.
        _dbg("gate-ablation: I disabled (skipping critic)")
        critic = {"needs_clarification": False}
        critic_trace = {"semantic_hints_count": len(semantic_hints), "raw_critic": critic,
                        "disabled": True}
    else:
        critic = _critic_check_with_llm(
            client=client,
            model=judge_model,
            question=q,
            context=c,
            semantic_hints=semantic_hints,
            provided_inputs=provided_inputs,
        )
        critic_trace = {
            "semantic_hints_count": len(semantic_hints),
            "raw_critic": critic,
        }
    if _HINT_TRIGGERS_ENABLED and "I" not in _DISABLED_GATES:
        det_hits = _lexical_hint_triggers(q, c, semantic_hints)
        if det_hits:
            llm_ids = [str(x).strip() for x in critic.get("triggered_hint_ids", []) if str(x).strip()]
            det_ids = [h["hint_id"] for h in det_hits]
            critic_trace["deterministic_hint_triggers"] = det_hits
            critic_trace["critic_recall_misses"] = [hid for hid in det_ids if hid not in llm_ids]
            critic["triggered_hint_ids"] = list(dict.fromkeys(llm_ids + det_ids))
            if not critic.get("needs_clarification"):
                critic["needs_clarification"] = True
                critic["reason"] = (
                    str(critic.get("reason", "") or "").strip()
                    or "Deterministic hint trigger: declared convention phrase present in question/context."
                )
            _dbg(f"hint-triggers: det={det_hits}, recall_misses={critic_trace['critic_recall_misses']}")
    needs_clarification = bool(critic.get("needs_clarification"))
    soft_warnings: List[Dict[str, Any]] = []
    if needs_clarification:
        triggered_hint_ids = [str(x).strip() for x in critic.get("triggered_hint_ids", []) if str(x).strip()]
        hint_by_id = {str(h.get("id", "")).strip(): h for h in semantic_hints if str(h.get("id", "")).strip()}
        i_tags = [str(x).strip() for x in critic.get("ambiguity_tags", []) if str(x).strip()]
        clar_qs = [str(x).strip() for x in critic.get("clarification_questions", []) if str(x).strip()]
        model_options = [str(x).strip() for x in critic.get("clarification_options", []) if str(x).strip()]
        hard_triggered = [
            hid for hid in triggered_hint_ids if _semantic_hint_level(hint_by_id.get(hid, {})) == "hard"
        ]
        soft_triggered = [
            hid for hid in triggered_hint_ids if _semantic_hint_level(hint_by_id.get(hid, {})) == "soft"
        ]

        open_inferred_level = _infer_open_critic_i_level(critic) if not triggered_hint_ids else None
        is_hard_block = bool(hard_triggered) or (not triggered_hint_ids and open_inferred_level == "hard")
        _dbg(
            "critic: "
            f"needs_clarification={needs_clarification}, triggered_hint_ids={triggered_hint_ids}, "
            f"hard_triggered={hard_triggered}, soft_triggered={soft_triggered}, "
            f"open_inferred_level={open_inferred_level}, is_hard_block={is_hard_block}"
        )

        i_rule_ids = [
            _i_rule_id_from_hint_id(hid, _semantic_hint_level(hint_by_id.get(hid, {})))
            for hid in hard_triggered
        ] if is_hard_block else []
        if is_hard_block and not i_rule_ids:
            i_rule_ids = ["global_i_semantic_ambiguity"]
        i_repairs = _summarize_repairs(chosen_fic_id, i_rule_ids, repair_index) if i_rule_ids else []
        if is_hard_block and not i_repairs:
            i_repairs = _summarize_repairs(chosen_fic_id, ["global_i_semantic_ambiguity"], repair_index)
        # Structured options ({label, value, has_transform}) so the UI can offer a
        # verifiable transform for the chosen interpretation (see _normalize_warning_options).
        raw_options: List[Any] = []
        if semantic_hints:
            for hint in semantic_hints:
                if not triggered_hint_ids or str(hint.get("id", "")) in triggered_hint_ids:
                    raw_options.extend(hint.get("options", []) or [])
        else:
            raw_options.extend(model_options)
        options = []
        _seen_opt_values = set()
        for opt in _normalize_warning_options(raw_options):
            if opt["value"] in _seen_opt_values:
                continue
            _seen_opt_values.add(opt["value"])
            options.append(opt)
        # Per-question groups so the UI can render one selector per ambiguity (instead of one
        # flat button row across multiple questions). Each group carries its own options.
        clar_groups: List[Dict[str, Any]] = []
        for hid in (triggered_hint_ids or []):
            hint = hint_by_id.get(hid)
            if not hint:
                continue
            grp_opts = _normalize_warning_options(hint.get("options"))
            if not grp_opts:
                continue
            clar_groups.append({
                "hint_id": hid,
                "question": str(hint.get("clarification_question", "")).strip(),
                "i_level": _semantic_hint_level(hint),
                "options": grp_opts,
            })
        if is_hard_block:
            _dbg("early-exit: I_hard (needs clarification)")
            _set_step("ihard_check", "error", "I_hard ambiguity requires user clarification.")
            _set_step("python_execution", "skipped", "Blocked by I_hard ambiguity.")
            _set_step("final_response", "success", "Returned I_hard clarification response.")
            return _finalize({
                **base,
                "status": "needs_clarification",
                "diagnostic_type": "I_HARD",
                "funnel_layer": "Critic",
                "gate_action": "critic_intervention",
                "reason": str(critic.get("reason", "")).strip() or "Hidden semantic ambiguity detected.",
                "fic_id": chosen_fic_id,
                "candidate_ids": candidate_ids,
                "support_gap_reason": None,
                "ambiguity_tags": i_tags,
                "clarification_request": {
                    "questions": clar_qs or ["Please confirm the intended financial interpretation."],
                    "options": options,
                    "groups": clar_groups,
                    "triggered_hint_ids": triggered_hint_ids,
                    "mode": "hint_oriented" if semantic_hints else "open",
                },
                "provided_inputs": provided_inputs,
                "normalization_note": normalization_note,
                "repair_hints": i_repairs,
                "i_level": "hard",
                "has_i_soft": False,
                "soft_warnings": [],
            })

        # I_soft: do not block execution; add warning payload and continue to C-execution.
        soft_warning_ids = soft_triggered if soft_triggered else []
        soft_warnings = _build_soft_warnings(
            critic=critic,
            semantic_hints=semantic_hints,
            triggered_hint_ids=soft_warning_ids,
        )
        _dbg(f"critic: I_soft warnings={len(soft_warnings)}")

    _set_step("ihard_check", "success", "I_hard check passed.")
    _set_step("python_execution", "running", "Executing deterministic Python compute(inputs).")

    output_value, exec_err, exec_trace = _evaluate_execution(core, provided_inputs)
    execution_trace = {
        **exec_trace,
        "fic_id": chosen_fic_id,
        "fic_version": core.get("version", "v1"),
    }
    if exec_err:
        _dbg(f"early-exit: C (execution error: {exec_err})")
        _set_step("python_execution", "error", f"C execution error: {exec_err}")
        _set_step("final_response", "success", "Returned C audit-log response.")
        return _finalize({
            **base,
            "status": "error",
            "diagnostic_type": "C",
            "funnel_layer": "Logic",
            "gate_action": "audit_log",
            "reason": exec_err,
            "fic_id": chosen_fic_id,
            "candidate_ids": candidate_ids,
            "support_gap_reason": None,
            "ambiguity_tags": [],
            "clarification_request": None,
            "provided_inputs": provided_inputs,
            "normalization_note": normalization_note,
            "has_i_soft": bool(soft_warnings),
            "soft_warnings": soft_warnings,
        })

    gold = row.get("gold_answer", row.get("answer", row.get("ground_truth")))
    gold_num = _parse_number(gold)
    abs_error, is_correct, scale_note = _answer_match(q, output_value, gold_num)
    _dbg(f"success: fic_id={chosen_fic_id}, output={output_value}, gold={gold_num}, is_correct={is_correct}" + (f", scale_note={scale_note}" if scale_note else ""))

    _set_step("python_execution", "success", "Python execution completed.")
    _set_step(
        "final_response",
        "success",
        "Completed response generation." + (f" I_soft warnings={len(soft_warnings)}." if soft_warnings else ""),
    )

    return _finalize({
        **base,
        "status": "success",
        "diagnostic_type": "None",
        "funnel_layer": "Logic",
        "gate_action": "audit_log",
        "reason": "Execution completed",
        "fic_id": chosen_fic_id,
        "candidate_ids": candidate_ids,
        "support_gap_reason": None,
        "ambiguity_tags": [],
        "clarification_request": None,
        "provided_inputs": provided_inputs,
        "output_var": (core.get("output") or {}).get("name"),
        "output_value": output_value,
        "gold_answer": gold_num if gold_num is not None else gold,
        "abs_error": abs_error,
        "is_correct": is_correct,
        **({
            "output_scale_note": (
                "FIC card returns a decimal fraction; gold answer is in percentage-point form. "
                "The computation is correct — update the FIC card's output to multiply by 100."
                if scale_note == "output_scale_x100"
                else f"Matched within 1% relative tolerance (scale_note={scale_note})."
            )
        } if scale_note else {}),
        "normalization_note": normalization_note,
        "has_i_soft": bool(soft_warnings),
        "soft_warnings": soft_warnings,
    })


class ErrorClassificationAPI:
    """Reusable API wrapper around VerifiQuant diagnostic run_case()."""

    def __init__(
        self,
        *,
        core_by_id: Dict[str, Dict[str, Any]],
        retrieval_cards: List[Dict[str, Any]],
        store: Optional[SQLAlchemyArtifactStore],
        repair_index: Dict[Tuple[str, str], Dict[str, Any]],
        client: Any,
        selector_model: str = "gemini-2.5-flash",
        extractor_model: str = "gemini-2.5-flash",
        judge_model: str = "gemini-2.5-flash",
        top_k: int = 3,
        m_min_top_score: float = 0.05,
    ) -> None:
        self.core_by_id = core_by_id
        self.retrieval_cards = retrieval_cards
        self.store = store
        self.repair_index = repair_index
        self.client = client
        self.selector_model = selector_model
        self.extractor_model = extractor_model
        self.judge_model = judge_model
        self.top_k = top_k
        self.m_min_top_score = m_min_top_score

    @classmethod
    def from_db(
        cls,
        *,
        db_url: str,
        client: Any,
        selector_model: str = "gemini-2.5-flash",
        extractor_model: str = "gemini-2.5-flash",
        judge_model: str = "gemini-2.5-flash",
        top_k: int = 3,
        m_min_top_score: float = 0.05,
    ) -> "ErrorClassificationAPI":
        store = SQLAlchemyArtifactStore(db_url)
        core_by_id = store.load_core_by_id()
        repair_index = store.build_repair_index()
        return cls(
            core_by_id=core_by_id,
            retrieval_cards=[],
            store=store,
            repair_index=repair_index,
            client=client,
            selector_model=selector_model,
            extractor_model=extractor_model,
            judge_model=judge_model,
            top_k=top_k,
            m_min_top_score=m_min_top_score,
        )

    @classmethod
    def from_files(
        cls,
        *,
        core_path: Path,
        retrieval_path: Path,
        repair_path: Path,
        client: Any,
        selector_model: str = "gemini-2.5-flash",
        extractor_model: str = "gemini-2.5-flash",
        judge_model: str = "gemini-2.5-flash",
        top_k: int = 3,
        m_min_top_score: float = 0.05,
    ) -> "ErrorClassificationAPI":
        core_by_id = _load_core(core_path)
        retrieval_cards = _load_retrieval(retrieval_path)
        repair_rows = _load_repair(repair_path)
        validate_artifact_relations(
            core_cards=list(core_by_id.values()),
            retrieval_cards=retrieval_cards,
            repair_rules=repair_rows,
        )
        repair_index = _build_repair_index(repair_rows)
        return cls(
            core_by_id=core_by_id,
            retrieval_cards=retrieval_cards,
            store=None,
            repair_index=repair_index,
            client=client,
            selector_model=selector_model,
            extractor_model=extractor_model,
            judge_model=judge_model,
            top_k=top_k,
            m_min_top_score=m_min_top_score,
        )

    def diagnose_row(
        self,
        row: Dict[str, Any],
        *,
        top_k: Optional[int] = None,
        m_min_top_score: Optional[float] = None,
        debug_sanity: bool = False,
    ) -> Dict[str, Any]:
        return run_case(
            row=row,
            core_by_id=self.core_by_id,
            retrieval_cards=self.retrieval_cards,
            store=self.store,
            repair_index=self.repair_index,
            client=self.client,
            selector_model=self.selector_model,
            extractor_model=self.extractor_model,
            judge_model=self.judge_model,
            top_k=top_k if top_k is not None else self.top_k,
            m_min_top_score=m_min_top_score if m_min_top_score is not None else self.m_min_top_score,
            debug_sanity=debug_sanity,
        )

    def diagnose(
        self,
        *,
        question: str,
        context: str,
        case_id: str = "",
        top_k: Optional[int] = None,
        m_min_top_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        return self.diagnose_row(
            {
                "case_id": case_id,
                "question": question,
                "context": context,
            },
            top_k=top_k,
            m_min_top_score=m_min_top_score,
        )

    def retrieve(
        self,
        *,
        question: str,
        context: str,
        top_k: Optional[int] = None,
        domain: Optional[str] = None,
        topic: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if self.store is not None:
            rows = self.store.retrieve_candidates(
                query=f"{question}\n{context}",
                top_k=top_k or self.top_k,
                domain=domain,
                topic=topic,
            )
            return rows

        candidates = retrieve_candidates(self.retrieval_cards, f"{question}\n{context}", top_k or self.top_k)
        out: List[Dict[str, Any]] = []
        for cand in candidates:
            row = dict(cand.retrieval)
            row["score"] = cand.score
            out.append(row)
        return out


def create_genai_client_from_env() -> Any:
    if genai is None or genai_types is None:
        raise RuntimeError("Missing dependency: google.genai import failed.")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY in environment.")
    return genai.Client(api_key=api_key)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run M/N/F/E/I/C diagnostic pipeline with separated core/retrieval/repair artifacts."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input JSON/JSONL questions")
    parser.add_argument("--core", type=Path, help="fic_core JSON/JSONL")
    parser.add_argument("--retrieval", type=Path, help="fic_retrieval JSON/JSONL")
    parser.add_argument("--repair", type=Path, help="repair_rule JSON/JSONL")
    parser.add_argument("--db-url", help="Optional SQLAlchemy DB URL. If set, retrieval/core/repair are loaded from DB.")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON/JSONL results")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--selector-model", default="gemini-2.5-flash")
    parser.add_argument("--extractor-model", default="gemini-2.5-flash")
    parser.add_argument("--judge-model", default="gemini-2.5-flash")
    parser.add_argument("--m-min-top-score", type=float, default=0.05)
    parser.add_argument(
        "--debug-sanity",
        action="store_true",
        help="Print detailed per-case funnel diagnostics (retrieval, selector, checks, critic).",
    )
    args = parser.parse_args()

    try:
        client = create_genai_client_from_env()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    rows = _load_records(args.input)
    if args.max_records > 0:
        rows = rows[: args.max_records]

    store: Optional[SQLAlchemyArtifactStore] = None
    if args.db_url:
        store = SQLAlchemyArtifactStore(args.db_url)
        stats = store.validate_integrity()
        print(
            "[run_run_error_classification_pipeline] db relation check passed: "
            f"core={stats['core_count']}, retrieval={stats['retrieval_count']}, "
            f"repair={stats['repair_count']}, diagnostic_rules={stats['diagnostic_rule_count']}"
        )
        core_by_id = store.load_core_by_id()
        retrieval_cards: List[Dict[str, Any]] = []
        repair_index = store.build_repair_index()
    else:
        if not (args.core and args.retrieval and args.repair):
            raise ValueError("When --db-url is not provided, --core --retrieval --repair are all required.")
        core_by_id = _load_core(args.core)
        retrieval_cards = _load_retrieval(args.retrieval)
        repair_rows = _load_repair(args.repair)
        validate_artifact_relations(
            core_cards=list(core_by_id.values()),
            retrieval_cards=retrieval_cards,
            repair_rules=repair_rows,
        )
        repair_index = _build_repair_index(repair_rows)
    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows, 1):
        cid = row.get("case_id") or row.get("question_id") or row.get("id") or idx
        print(f"[run_run_error_classification_pipeline] processing {cid}")
        result = run_case(
            row=row,
            core_by_id=core_by_id,
            retrieval_cards=retrieval_cards,
            store=store,
            repair_index=repair_index,
            client=client,
            selector_model=args.selector_model,
            extractor_model=args.extractor_model,
            judge_model=args.judge_model,
            top_k=args.top_k,
            m_min_top_score=args.m_min_top_score,
            debug_sanity=args.debug_sanity,
        )

        for k in ("variant_type", "expected_status", "expected_diagnostic_type", "source_sample_id", "reason"):
            if k in row and k not in result:
                result[f"expected_{k}" if k == "reason" else k] = row[k]
        out.append(result)

    _dump_records(args.output, out)
    print(f"Wrote {len(out)} results to {args.output}")


if __name__ == "__main__":
    main()
