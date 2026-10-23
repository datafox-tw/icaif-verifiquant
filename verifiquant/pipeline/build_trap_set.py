"""
build_trap_set.py
-----------------
Contract-grounded trap-dataset generator (Tier-1, deterministic).

Design (see docs/paper_draft_finnlp.md §5.6 and CLAUDE.md):
  A *trap operator* is a deterministic function

      operator(clean_case, its_FIC_card) -> (perturbed_case, expected_safe_behavior)

  The label is NOT guessed by an LLM/regex — the label IS the operator, because
  the operator knows which contract field it corrupted. Evaluation is then indexed
  by *operator* (not by the funnel layer we hope it triggers): the primary metric
  is Trap SWR per operator; a secondary confusion matrix (injected operator ×
  triggered diagnostic_type) reports routing behavior.

Operators (aligned to contracts.py RefusalCategory):
  F  drop a required input declared by the FIC               -> expect F (refuse/ask)
  E  inject a value that violates a declared E diagnostic    -> expect E (abstain/repair)
  I  remove the disambiguator on a transform_map FIC         -> expect I_HARD (clarify)
  N  ask for out-of-scope logic                              -> expect N (graceful refuse)
  M  make the metric intent ambiguous                        -> expect M (ask to disambiguate)

Grounding sources:
  - clean questions JSONL (question/context/python_solution/ground_truth/function)
  - canonical run output.jsonl (case_id -> selected fic_id + provided_inputs)
  - cards_v3.db (diagnostic_checks_json -> E predicates; repair_rules -> I-class)

Tier-1 is fully deterministic. Records the operator could not ground cleanly are
emitted with needs_review=true (for Tier-2 LLM refine + human eyeball), never dropped
silently.

Usage:
    python3 verifiquant/pipeline/build_trap_set.py \
        --questions results/paper_v1/questions_50.jsonl \
        --canonical results/paper_v1/results/vq_flash_v3/output.jsonl \
        --db results/paper_v1/fic/cards_v3.db \
        --out-dir results/paper_v1/trap \
        --per-operator 10
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def _load_canonical_map(path: str) -> Dict[str, Dict[str, Any]]:
    """case_id -> {fic_id, provided_inputs, output_var} from round-1 diagnostic."""
    out: Dict[str, Dict[str, Any]] = {}
    for rec in _load_jsonl(path):
        cid = rec.get("case_id") or rec.get("source_sample_id")
        hist = rec.get("framework_guided", {}).get("history", [])
        if not cid or not hist:
            continue
        d = hist[0].get("diagnostic", {})
        out[cid] = {
            "fic_id": d.get("fic_id"),
            "provided_inputs": d.get("provided_inputs") or {},
            "output_var": d.get("output_var"),
        }
    return out


def _load_cards(db_path: str) -> Dict[str, Dict[str, Any]]:
    """fic_id -> {checks: [E-predicate dicts], i_rules: [I-class repair dicts]}."""
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cards: Dict[str, Dict[str, Any]] = {}
    for fic_id, dc in cur.execute("select fic_id, diagnostic_checks_json from core_cards"):
        checks = json.loads(dc) if dc else []
        e_checks = [c for c in checks if isinstance(c, dict) and c.get("diagnostic_type") == "E"]
        cards.setdefault(fic_id, {})["checks"] = e_checks
    for fic_id, rule_id, dt, ra in cur.execute(
        "select fic_id, rule_id, diagnostic_type, repair_action_json from repair_rules"
    ):
        if dt not in ("I", "I_HARD", "I_SOFT"):
            continue
        ra_obj = json.loads(ra) if ra else {}
        has_transform = isinstance(ra_obj, dict) and (
            "transform_map" in ra_obj or "transform_spec" in json.dumps(ra_obj)
        )
        cards.setdefault(fic_id, {}).setdefault("i_rules", []).append(
            {"rule_id": rule_id, "diagnostic_type": dt, "has_transform": has_transform}
        )
    con.close()
    return cards


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #
def _arg_names(func_sig: str) -> List[str]:
    """Parse 'def f(a: float, b: int) -> ...' -> ['a','b']."""
    m = re.search(r"def\s+\w+\s*\(([^)]*)\)", func_sig or "")
    if not m:
        return []
    args = []
    for part in m.group(1).split(","):
        name = part.split(":")[0].split("=")[0].strip()
        if name and name not in ("self", "cls"):
            args.append(name)
    return args


def _value_forms(v: Any) -> List[str]:
    """Plausible textual renderings of a numeric input value, longest first."""
    forms: List[str] = []
    try:
        f = float(v)
    except (TypeError, ValueError):
        return forms
    i = int(f) if f == int(f) else None
    if i is not None:
        forms.append(f"{i:,}")   # 1,200,000
        forms.append(str(i))     # 1200000
    else:
        forms.append(f"{f:,}")
        forms.append(str(f))
        # trailing-zero variants e.g. 0.08 vs .08
        forms.append(("%g" % f))
    # dedup preserving order, longest first so commas-form is tried before plain
    seen, out = set(), []
    for s in sorted(set(forms), key=len, reverse=True):
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _redact_value(text: str, value: Any) -> Tuple[str, bool]:
    """Remove the first textual occurrence of a numeric value (with optional $ prefix)."""
    for form in _value_forms(value):
        pat = re.compile(r"\$?\b" + re.escape(form) + r"\b")
        if pat.search(text):
            return pat.sub("[REDACTED]", text, count=1), True
    return text, False


def _replace_value(text: str, old: Any, new: str) -> Tuple[str, bool]:
    for form in _value_forms(old):
        pat = re.compile(r"\b" + re.escape(form) + r"\b")
        if pat.search(text):
            return pat.sub(new, text, count=1), True
    return text, False


# Disambiguator removal patterns for I-traps, keyed by the rule-family token that
# appears in the FIC repair rule_id (e.g. i_output_scale_soft -> "scale"). Each
# entry is (family_token, compiled_pattern). We delete the matched phrase from the
# question so the declared ambiguity is no longer resolved by the prompt text.
_DISAMBIGUATOR_PATTERNS = [
    # output / tax-rate / generic scale ambiguity: "as a percentage" vs "as a decimal"
    ("scale", re.compile(
        r"\b(?:expressed |provided |stated )?as an?\s+(?:a\s+)?(?:percentage|decimal)\b"
        r"|\bin percent(?:age)?(?: terms)?\b|\bas a percent\b", re.IGNORECASE)),
    # option type: "put option" / "call option"
    ("option_type", re.compile(r"\b(put|call)\s+option\b", re.IGNORECASE)),
    # time-unit / throughput basis
    ("time_unit", re.compile(
        r"\bin (?:hours|days|minutes|years|months|weeks)\b"
        r"|\bper (?:hour|day|year|month|week)\b", re.IGNORECASE)),
    # cash-flow timing
    ("timing", re.compile(
        r"\b(annuity[- ]due|ordinary annuity|due at the beginning|"
        r"at the (?:beginning|end) of (?:the |each )?(?:period|year|month)|"
        r"beginning[- ]of[- ]period|end[- ]of[- ]period)\b", re.IGNORECASE)),
    # compounding / rate basis
    ("rate", re.compile(
        r"\b(compounded (?:annually|monthly|quarterly|semi-?annually|daily)|"
        r"annual(?:ly)? compounded|nominal annual rate|effective annual rate)\b",
        re.IGNORECASE)),
    # fx direction
    ("fx", re.compile(r"\b([A-Z]{3}/[A-Z]{3}|direct quote|indirect quote)\b")),
]


def _tidy(text: str) -> str:
    """Collapse whitespace and fix dangling punctuation left by a phrase deletion."""
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([.,;:])", r"\1", text)   # " ." -> "."
    text = re.sub(r"([.,;:]){2,}", r"\1", text)  # ",," -> ","
    text = re.sub(r"\s+\?", "?", text)
    return text.strip()


def _family_of(rule_id: str) -> str:
    """Map a repair rule_id (e.g. 'i_output_scale_soft') to a disambiguator family token."""
    rid = (rule_id or "").lower()
    if "scale" in rid:
        return "scale"
    if "option_type" in rid or "option" in rid:
        return "option_type"
    if "time_unit" in rid or "throughput" in rid or "unit" in rid:
        return "time_unit"
    if "timing" in rid or "annuity" in rid:
        return "timing"
    if "rate" in rid or "compound" in rid:
        return "rate"
    if "fx" in rid or "currency" in rid:
        return "fx"
    return ""


# --------------------------------------------------------------------------- #
# Operators
# --------------------------------------------------------------------------- #
def op_F(case: Dict, info: Dict, card: Dict) -> Optional[Dict]:
    """Drop a required input declared by the FIC."""
    req = _arg_names(case.get("function", ""))
    provided = info.get("provided_inputs", {})
    # prefer a field whose value we can actually locate in the text
    for field in req:
        if field not in provided:
            continue
        val = provided[field]
        q2, hit_q = _redact_value(case["question"], val)
        c2, hit_c = _redact_value(case["context"], val)
        if hit_q or hit_c:
            return _record(
                case, "F", "F", "refuse_or_ask", q2, c2,
                grounding={"removed_field": field, "removed_value": val,
                           "fic_id": info.get("fic_id")},
                needs_review=False,
                reason=f"Removed required input '{field}' (value {val}) declared by FIC signature.",
            )
    # fallback: explicit marker, flag for review
    return _record(
        case, "F", "F", "refuse_or_ask", case["question"] + " [MISSING_REQUIRED_FIELD]",
        case["context"], grounding={"fic_id": info.get("fic_id"), "required": req},
        needs_review=True,
        reason="Could not locate a required input's value in text; used explicit marker.",
    )


def op_E(case: Dict, info: Dict, card: Dict) -> Optional[Dict]:
    """Inject a value that violates a declared E diagnostic check."""
    provided = info.get("provided_inputs", {})
    for chk in card.get("checks", []):
        expr = chk.get("expression", "")
        # target simple non-negativity violations: inputs['X'] < 0  (predicate_mode=violation)
        m = re.search(r"inputs\[['\"](\w+)['\"]\]\s*<\s*0", expr)
        if not m:
            continue
        field = m.group(1)
        if field not in provided:
            continue
        val = provided[field]
        neg = f"-{_value_forms(val)[0]}" if _value_forms(val) else None
        if neg is None:
            continue
        q2, hit_q = _replace_value(case["question"], val, neg)
        c2, hit_c = _replace_value(case["context"], val, neg)
        if hit_q or hit_c:
            return _record(
                case, "E", "E", "abstain_or_repair", q2, c2,
                grounding={"targeted_rule_id": chk.get("rule_id"), "expression": expr,
                           "field": field, "violating_value": neg,
                           "fic_id": info.get("fic_id")},
                needs_review=False,
                reason=f"Set '{field}' negative to violate E-check {chk.get('rule_id')}.",
            )
    return None  # not groundable deterministically; caller will try next case


_PAT_BY_FAMILY = {fam: pat for fam, pat in _DISAMBIGUATOR_PATTERNS}


def op_I(case: Dict, info: Dict, card: Dict) -> Optional[Dict]:
    """Restore a declared ambiguity by deleting the disambiguating phrase from the
    prompt. The FIC repair rule_id tells us BOTH the ambiguity family and the
    RefusalCategory: the '_hard'/'_soft' suffix maps directly to I_HARD (changes
    the computation path -> clarify) or I_SOFT (representation only -> proceed with
    explicit warning), aligning with contracts.py.
    """
    i_rules = card.get("i_rules", [])
    if not i_rules:
        return None
    # Try every declared I-rule; prefer one whose disambiguator we can actually find
    # in the prompt. Within that, prefer _hard rules (stronger trap) over _soft.
    ranked = sorted(
        i_rules,
        key=lambda r: (0 if str(r.get("rule_id", "")).endswith("_hard") else 1),
    )
    for r in ranked:
        rid = r.get("rule_id", "")
        fam = _family_of(rid)
        pat = _PAT_BY_FAMILY.get(fam)
        if pat is None:
            continue
        m = pat.search(case["question"]) or pat.search(case["context"])
        if not m:
            continue
        phrase = m.group(0)
        # Remove EVERY occurrence in BOTH fields, otherwise a duplicate phrase
        # (e.g. one in question + one in context) leaves the prompt disambiguated.
        q2 = _tidy(pat.sub("", case["question"]))
        c2 = _tidy(pat.sub("", case["context"]))
        # Validity gate: if a scale/representation instruction still survives in the
        # question, the ambiguity was not actually restored -> hand to Tier-2/human.
        survives = bool(pat.search(q2))
        is_hard = str(rid).endswith("_hard")
        expected_dt = "I_HARD" if is_hard else "I_SOFT"
        behavior = "clarify" if is_hard else "proceed_with_warning"
        return _record(
            case, "I", expected_dt, behavior, q2, c2,
            grounding={"ambiguity_family": fam, "removed_phrase": phrase,
                       "rule_id": rid, "fic_id": info.get("fic_id")},
            needs_review=survives,
            reason=(f"Removed {fam} disambiguator '{phrase}'; FIC rule {rid} "
                    f"declares this ambiguity ({expected_dt})."
                    + (" WARNING: a representation cue still survives in the question."
                       if survives else "")),
        )
    # eligible FIC but no disambiguator phrase located -> Tier-2 / human
    return _record(
        case, "I", "I_HARD", "clarify", case["question"], case["context"],
        grounding={"fic_id": info.get("fic_id"),
                   "i_rules": [r.get("rule_id") for r in i_rules]},
        needs_review=True,
        reason="FIC declares I-class ambiguity but no known disambiguator phrase found; needs Tier-2/human.",
    )


_N_TEMPLATE = ("Please price this scenario using a Heston stochastic-volatility model "
               "and report the risk-neutral value.")
_M_TEMPLATE = "Overall, is this worth doing? Please evaluate."


def op_N(case: Dict, info: Dict, card: Dict) -> Optional[Dict]:
    return _record(
        case, "N", "N", "graceful_refuse", _N_TEMPLATE, case["context"],
        grounding={"base_fic_id": info.get("fic_id")},
        needs_review=False,
        reason="Replaced question with out-of-scope (exotic-pricing) ask; no FIC should match.",
    )


def op_M(case: Dict, info: Dict, card: Dict) -> Optional[Dict]:
    return _record(
        case, "M", "M", "ask_to_disambiguate", _M_TEMPLATE, case["context"],
        grounding={"base_fic_id": info.get("fic_id")},
        needs_review=False,
        reason="Replaced question with ambiguous metric intent (no explicit metric named).",
    )


def _record(case, operator, expected_dt, expected_behavior, question, context,
            grounding, needs_review, reason) -> Dict:
    trap_id = f"{operator}-{case['question_id']}"
    return {
        "trap_id": trap_id,
        # `question_id` mirrors trap_id so the existing pipelines (which key on
        # case_id/question_id) treat each trap as a first-class case.
        "question_id": trap_id,
        "base_question_id": case["question_id"],
        "operator": operator,
        "expected_diagnostic": expected_dt,
        "expected_behavior": expected_behavior,
        # run-ready fields (gold code/answer kept for O-ITL oracle)
        "question": question,
        "context": context,
        "function": case.get("function", ""),
        "python_solution": case.get("python_solution", ""),
        "ground_truth": case.get("ground_truth"),
        "function_id": case.get("function_id"),
        # trap metadata
        "grounding": grounding,
        "needs_review": needs_review,
        "reason": reason,
    }


OPERATORS = {"F": op_F, "E": op_E, "I": op_I, "N": op_N, "M": op_M}


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def build(questions_path, canonical_path, db_path, out_dir, per_operator):
    cases = _load_jsonl(questions_path)
    canon = _load_canonical_map(canonical_path)
    cards = _load_cards(db_path)

    # deterministic order
    cases.sort(key=lambda c: c.get("question_id", ""))

    traps: List[Dict] = []
    manifest: Dict[str, Any] = {"per_operator": per_operator, "operators": {}}

    for op in ("F", "E", "I", "N", "M"):
        fn = OPERATORS[op]
        # generate every candidate once, then prefer clean (needs_review=False)
        # records and only top up remaining slots with needs_review ones.
        clean, review = [], []
        for case in cases:
            cid = case.get("question_id")
            info = canon.get(cid, {})
            card = cards.get(info.get("fic_id"), {}) if info.get("fic_id") else {}
            rec = fn(case, info, card)
            if rec is None:
                continue
            (review if rec["needs_review"] else clean).append(rec)
        produced = clean[:per_operator]
        if len(produced) < per_operator:
            produced += review[: per_operator - len(produced)]
        needs_review = sum(1 for r in produced if r["needs_review"])
        traps.extend(produced)
        manifest["operators"][op] = {
            "produced": len(produced),
            "needs_review": needs_review,
            "target": per_operator,
        }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trap_path = out_dir / "trap_set.jsonl"
    with open(trap_path, "w", encoding="utf-8") as f:
        for rec in traps:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    manifest_path = out_dir / "trap_manifest.json"
    manifest["total_traps"] = len(traps)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[trap] wrote {len(traps)} traps -> {trap_path}")
    for op, m in manifest["operators"].items():
        print(f"  {op}: {m['produced']}/{m['target']}  needs_review={m['needs_review']}")
    print(f"[trap] manifest -> {manifest_path}")


def main():
    ap = argparse.ArgumentParser(description="Contract-grounded trap-set generator (Tier-1).")
    ap.add_argument("--questions", required=True)
    ap.add_argument("--canonical", required=True, help="canonical run output.jsonl (for fic/inputs mapping)")
    ap.add_argument("--db", required=True, help="cards_v3.db")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--per-operator", type=int, default=10)
    args = ap.parse_args()
    build(args.questions, args.canonical, args.db, args.out_dir, args.per_operator)


if __name__ == "__main__":
    main()
