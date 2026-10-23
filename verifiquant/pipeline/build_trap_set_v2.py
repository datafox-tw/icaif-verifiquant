"""
build_trap_set_v2.py
--------------------
Balanced, contract-*verified* trap-set generator (P0 confirmatory build).

Supersedes build_trap_set.py for the governance confirmatory set. Three changes
fix the biases audited on the 50Q pilot (docs/2026-07-29_governance_paper_experiment_roadmap.md):

  1. E is no longer "make one field negative". It draws >=4 violation families
     (negativity, rate-out-of-[0,1], non-positive / zero-denominator, magnitude)
     with a hard cap of 8/20 negativity traps.
  2. N is no longer a single Heston sentence. It rotates >=5 out-of-scope asks
     (<=4 per template), including domain-adjacent "near-miss" asks.
  3. I is balanced to 10 I_HARD + 10 I_SOFT rather than whatever the pilot cards
     happened to yield (which was 1 hard / 9 soft, 8 of them one rule).

The decisive addition is *verification*: for every E trap we EVALUATE the FIC's
own diagnostic-check expression against the mutated inputs and require it to fire
(and require the clean value NOT to fire). The label is therefore grounded in the
contract, not in a gold answer -- satisfying the roadmap's freeze condition that
"80/80 labels are verifiable from contract/operator, 0 rely on gold number".

Determinism: no LLM calls, no network. Every candidate that cannot be grounded
cleanly is excluded with a recorded reason (never silently mutated).

Usage:
    python3 verifiquant/pipeline/build_trap_set_v2.py \
        --questions results/paper_v2_250/questions_250.jsonl \
        --canonical results/paper_v2_250/results/vq_flash/output.jsonl \
        --db        results/paper_v2_250/fic/cards.db \
        --out-dir   results/governance_confirmatory \
        --per-operator 20
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from datetime import datetime
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


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_canonical_map(path: str) -> Dict[str, Dict[str, Any]]:
    """case_id -> {fic_id, provided_inputs} from the round-1 diagnostic."""
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
        }
    return out


def _load_cards(db_path: str) -> Dict[str, Dict[str, Any]]:
    """fic_id -> {e_checks: [full check dicts], i_rules: [{rule_id, dt}]}."""
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cards: Dict[str, Dict[str, Any]] = {}
    for fic_id, dc in cur.execute("select fic_id, diagnostic_checks_json from core_cards"):
        checks = json.loads(dc) if dc else []
        e = [c for c in checks if isinstance(c, dict) and c.get("diagnostic_type") == "E"]
        cards.setdefault(fic_id, {})["e_checks"] = e
    for fic_id, rule_id, dt in cur.execute(
        "select fic_id, rule_id, diagnostic_type from repair_rules"
    ):
        if dt not in ("I", "I_HARD", "I_SOFT"):
            continue
        rid = (rule_id or "")
        is_hard = rid.lower().endswith("_hard") or dt == "I_HARD"
        cards.setdefault(fic_id, {}).setdefault("i_rules", []).append(
            {"rule_id": rid, "diagnostic_type": ("I_HARD" if is_hard else "I_SOFT")}
        )
    con.close()
    return cards


# --------------------------------------------------------------------------- #
# Text helpers (shared with v1 semantics)
# --------------------------------------------------------------------------- #
def _arg_names(func_sig: str) -> List[str]:
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
    forms: List[str] = []
    try:
        f = float(v)
    except (TypeError, ValueError):
        return forms
    i = int(f) if f == int(f) else None
    if i is not None:
        forms.append(f"{i:,}")
        forms.append(str(i))
    else:
        forms.append(f"{f:,}")
        forms.append(str(f))
        forms.append(("%g" % f))
    seen, out = set(), []
    for s in sorted(set(forms), key=len, reverse=True):
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _redact_value(text: str, value: Any) -> Tuple[str, bool]:
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


def _fmt_new(v: float) -> str:
    """Render an injected numeric value for substitution into prose."""
    if v == int(v):
        return f"{int(v):,}"
    return "%g" % v


def _tidy(text: str) -> str:
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    text = re.sub(r"([.,;:]){2,}", r"\1", text)
    text = re.sub(r"\s+\?", "?", text)
    return text.strip()


# --------------------------------------------------------------------------- #
# E-check evaluation (the verification core)
# --------------------------------------------------------------------------- #
_EVAL_NS = {
    "all": all, "any": any, "len": len, "sum": sum, "abs": abs,
    "min": min, "max": max, "int": int, "float": float, "round": round,
    "isinstance": isinstance, "range": range, "bool": bool,
}


def _eval_check(expr: str, inputs: Dict[str, Any]) -> Optional[bool]:
    """Evaluate a diagnostic-check expression against `inputs`.
    Returns the boolean result, or None if it cannot be evaluated safely
    (e.g. references compute(), unknown names, raises)."""
    if "compute(" in expr:            # needs the card's python_code; out of scope
        return None
    try:
        return bool(eval(expr, {"__builtins__": {}}, {**_EVAL_NS, "inputs": inputs}))
    except Exception:
        return None


def _fires(expr: str, mode: str, inputs: Dict[str, Any]) -> Optional[bool]:
    """Does this check FIRE (flag an error) on `inputs`?
    violation-mode: fires when expression is True. validity-mode: fires when False."""
    r = _eval_check(expr, inputs)
    if r is None:
        return None
    return r if mode == "violation" else (not r)


# candidate injections in family priority; each is (family, value_fn)
def _e_candidates(v: float) -> List[Tuple[str, float]]:
    cands: List[Tuple[str, float]] = []
    cands.append(("nonpositive_zero", 0.0))
    cands.append(("rate_out_of_unit", 2.5))            # 250% -> violates rate<=1 bounds
    cands.append(("negativity", -abs(v) if v != 0 else -1.0))
    cands.append(("magnitude", (abs(v) * 1_000_000.0) if v != 0 else 1e9))
    return cands


# --------------------------------------------------------------------------- #
# Operators
# --------------------------------------------------------------------------- #
def op_F(case: Dict, info: Dict, card: Dict) -> List[Dict]:
    """Drop a required input; one candidate per locatable required field."""
    out = []
    req = _arg_names(case.get("function", ""))
    provided = info.get("provided_inputs", {})
    for field in req:
        if field not in provided:
            continue
        val = provided[field]
        q2, hit_q = _redact_value(case["question"], val)
        c2, hit_c = _redact_value(case["context"], val)
        if hit_q or hit_c:
            out.append(_record(
                case, "F", "F", "refuse_or_ask", q2, c2,
                family=_field_kind(field),
                grounding={"removed_field": field, "removed_value": val,
                           "fic_id": info.get("fic_id")},
                reason=f"Removed required input '{field}' (value {val}) declared by FIC signature.",
            ))
    return out


def _field_kind(name: str) -> str:
    n = (name or "").lower()
    table = [
        ("denominator", ["denominator", "divisor", "base", "shares_outstanding", "total_shares"]),
        ("rate", ["rate", "yield", "coupon", "discount", "wacc", "return"]),
        ("time", ["period", "year", "month", "day", "tenor", "maturity", "time", "n_"]),
        ("count", ["count", "number", "num_", "quantity", "units"]),
        ("balance", ["balance", "assets", "equity", "liabilit", "value", "capital", "principal"]),
        ("price", ["price", "cost", "amount", "revenue", "expense", "cash", "income"]),
    ]
    for kind, keys in table:
        if any(k in n for k in keys):
            return kind
    return "other"


def op_E(case: Dict, info: Dict, card: Dict) -> List[Dict]:
    """Inject a value that PROVABLY fires a declared E-check (verified by eval)."""
    out = []
    provided = info.get("provided_inputs", {})
    for chk in card.get("e_checks", []):
        expr = (chk.get("expression") or "").strip()
        mode = chk.get("predicate_mode", "violation")
        applies = chk.get("applies_to") or []
        # candidate target fields: applies_to first, then names parsed from expr
        fields = list(applies) + re.findall(r"inputs\[['\"](\w+)['\"]\]", expr)
        seen = set()
        for field in fields:
            if field in seen:
                continue
            seen.add(field)
            if field not in provided:
                continue
            v = provided[field]
            try:
                vf = float(v)
            except (TypeError, ValueError):
                continue                       # scalar-only injection
            # clean value must be valid (check must NOT fire on it)
            if _fires(expr, mode, {**provided, field: vf}) is not False:
                continue
            for family, newval in _e_candidates(vf):
                mut = {**provided, field: newval}
                if _fires(expr, mode, mut) is not True:
                    continue                   # this candidate does not trigger the check
                new_txt = _fmt_new(newval)
                q2, hit_q = _replace_value(case["question"], v, new_txt)
                c2, hit_c = _replace_value(case["context"], v, new_txt)
                if not (hit_q or hit_c):
                    continue                   # cannot realize in the prompt text
                out.append(_record(
                    case, "E", "E", "abstain_or_repair", q2, c2,
                    family=family,
                    grounding={"targeted_rule_id": chk.get("rule_id"), "expression": expr,
                               "predicate_mode": mode, "field": field,
                               "clean_value": v, "violating_value": newval,
                               "verified": True, "fic_id": info.get("fic_id")},
                    reason=(f"Set '{field}'={new_txt} to fire E-check {chk.get('rule_id')} "
                            f"[{family}]; verified against contract predicate."),
                ))
                break                          # one family per (check, field)
    return out


_DISAMBIGUATOR_PATTERNS = [
    ("scale", re.compile(
        r"\b(?:expressed |provided |stated )?as an?\s+(?:a\s+)?(?:percentage|decimal)\b"
        r"|\bin percent(?:age)?(?: terms)?\b|\bas a percent\b", re.IGNORECASE)),
    # delete only the put/call word (keep "option") so prose stays clean and the
    # put-vs-call ambiguity is restored: "European call option" -> "European option"
    ("option_type", re.compile(r"\b(put|call)\b(?=\s+option)", re.IGNORECASE)),
    ("time_unit", re.compile(
        r"\bin (?:hours|days|minutes|years|months|weeks)\b"
        r"|\bper (?:hour|day|year|month|week)\b", re.IGNORECASE)),
    ("timing", re.compile(
        r"\b(annuity[- ]due|ordinary annuity|due at the beginning|"
        r"at the (?:beginning|end) of (?:the |each )?(?:period|year|month)|"
        r"beginning[- ]of[- ]period|end[- ]of[- ]period)\b", re.IGNORECASE)),
    ("rate", re.compile(
        r"\b(compounded (?:annually|monthly|quarterly|semi-?annually|daily)|"
        r"annual(?:ly)? compounded|nominal annual rate|effective annual rate)\b",
        re.IGNORECASE)),
    ("basis", re.compile(
        r"\b(per[- ]share|per share|in total|aggregate|on a total basis|before[- ]tax|after[- ]tax|pre[- ]tax|net of tax)\b",
        re.IGNORECASE)),
    ("fx", re.compile(r"\b([A-Z]{3}/[A-Z]{3}|direct quote|indirect quote)\b")),
    # method/basis hard ambiguities that DO have a deletable cue phrase
    ("option_style", re.compile(
        r"\b(american|european)[- ]?(?:style|exercise)\b|\b(american|european)\s+option\b",
        re.IGNORECASE)),
    ("interest_method", re.compile(r"\b(simple|compound)\s+interest\b", re.IGNORECASE)),
    ("inventory", re.compile(
        r"\b(fifo|lifo|first[- ]in[, ]+first[- ]out|last[- ]in[, ]+first[- ]out|"
        r"weighted[- ]average(?:\s+cost)?)\b", re.IGNORECASE)),
]
_PAT_BY_FAMILY = {fam: pat for fam, pat in _DISAMBIGUATOR_PATTERNS}


def _family_of(rule_id: str) -> str:
    rid = (rule_id or "").lower()
    # order matters: more specific tokens before generic ones
    for key, fam in [("scale", "scale"),
                     ("exercise", "option_style"),
                     ("inventory", "inventory"),
                     ("interest_calc", "interest_method"),
                     ("interest_type", "interest_method"),
                     ("interest_method", "interest_method"),
                     ("deferred_interest", "interest_method"),
                     ("option", "option_type"),
                     ("time_unit", "time_unit"), ("throughput", "time_unit"),
                     ("unit", "time_unit"), ("timing", "timing"), ("annuity", "timing"),
                     ("compound", "rate"), ("rate", "rate"),
                     ("basis", "basis"), ("pershare", "basis"), ("per_share", "basis"),
                     ("tax", "basis"), ("fx", "fx"), ("currency", "fx")]:
        if key in rid:
            return fam
    return ""


def op_I(case: Dict, info: Dict, card: Dict) -> List[Dict]:
    """Delete the disambiguating phrase; hard/soft taken from the FIC rule suffix."""
    out = []
    for r in card.get("i_rules", []):
        rid = r.get("rule_id", "")
        fam = _family_of(rid)
        pat = _PAT_BY_FAMILY.get(fam)
        if pat is None:
            continue
        if not (pat.search(case["question"]) or pat.search(case["context"])):
            continue
        phrase_m = pat.search(case["question"]) or pat.search(case["context"])
        q2 = _tidy(pat.sub("", case["question"]))
        c2 = _tidy(pat.sub("", case["context"]))
        if pat.search(q2):                      # a cue survives -> ambiguity not restored
            continue
        is_hard = r.get("diagnostic_type") == "I_HARD"
        out.append(_record(
            case, "I", "I_HARD" if is_hard else "I_SOFT",
            "clarify" if is_hard else "proceed_with_warning", q2, c2,
            family="hard" if is_hard else "soft",
            grounding={"ambiguity_family": fam, "removed_phrase": phrase_m.group(0),
                       "rule_id": rid, "fic_id": info.get("fic_id")},
            reason=f"Removed {fam} disambiguator '{phrase_m.group(0)}'; FIC rule {rid} "
                   f"declares this ambiguity ({'I_HARD' if is_hard else 'I_SOFT'}).",
        ))
    return out


# >=5 out-of-scope asks; `near_miss` ones stay in the base's domain but name a
# metric with no matching FIC, `exotic` ones are plainly out of library.
_N_TEMPLATES = [
    ("exotic_heston", "Please price this scenario using a Heston stochastic-volatility "
                      "model and report the risk-neutral value."),
    ("exotic_american_pde", "Value the embedded American-style early-exercise option here "
                            "using a finite-difference PDE solver and report its price."),
    ("exotic_var_hist", "Compute the one-day 99% Value-at-Risk of this position using "
                        "historical simulation over the return distribution."),
    ("exotic_cds", "Bootstrap the hazard-rate term structure and quote the fair credit "
                   "default swap spread implied by this scenario."),
    ("exotic_copula", "Estimate the portfolio's tail dependence with a t-copula and report "
                      "the joint 1% tail loss."),
    ("near_miss_keyrate", "Report the key-rate duration bucketed by the 2y, 5y and 10y "
                          "tenors for this position."),
    ("near_miss_montecarlo", "Run a Monte-Carlo simulation of this scenario and report the "
                             "95% confidence interval of the outcome."),
]


def op_N(case: Dict, info: Dict, card: Dict, template_idx: int) -> List[Dict]:
    key, text = _N_TEMPLATES[template_idx % len(_N_TEMPLATES)]
    kind = "near_miss" if key.startswith("near_miss") else "exotic"
    return [_record(
        case, "N", "N", "graceful_refuse", text, case["context"],
        family=kind,
        grounding={"base_fic_id": info.get("fic_id"), "template": key},
        reason=f"Replaced question with out-of-scope ask ({key}); no FIC should match.",
    )]


def _record(case, operator, expected_dt, expected_behavior, question, context,
            family, grounding, reason) -> Dict:
    trap_id = f"{operator}-{case['question_id']}"
    return {
        "trap_id": trap_id,
        "question_id": trap_id,
        "base_question_id": case["question_id"],
        "operator": operator,
        "family": family,
        "expected_diagnostic": expected_dt,
        "expected_behavior": expected_behavior,
        "question": question,
        "context": context,
        "function": case.get("function", ""),
        "python_solution": case.get("python_solution", ""),
        "ground_truth": case.get("ground_truth"),
        "function_id": case.get("function_id"),
        "grounding": grounding,
        "needs_review": False,
        "reason": reason,
    }


# --------------------------------------------------------------------------- #
# Selection with quotas + base-question dedup
# --------------------------------------------------------------------------- #
# per-operator family quotas (sum == per_operator when per_operator==20)
_E_QUOTA = {"negativity": 8, "rate_out_of_unit": 4, "nonpositive_zero": 4, "magnitude": 4}
_I_QUOTA = {"hard": 10, "soft": 10}
_N_QUOTA = {"exotic": 14, "near_miss": 6}
_N_PER_TEMPLATE_CAP = 4
_F_MAX_PER_KIND = 5          # forces >=4 field-kinds across 20 F traps


def _select(candidates: List[Dict], operator: str, per_operator: int,
            used_bases: Dict[str, int], max_ops_per_base: int) -> Tuple[List[Dict], Dict]:
    """Greedy pick honoring family quotas and the global base-dedup budget."""
    chosen: List[Dict] = []
    fam_count: Dict[str, int] = {}
    tmpl_count: Dict[str, int] = {}
    picked_bases = set()

    def base_ok(b):
        return used_bases.get(b, 0) < max_ops_per_base and b not in picked_bases

    # deterministic order
    candidates = sorted(candidates, key=lambda r: (r["family"], r["base_question_id"]))

    if operator == "E":
        quota = dict(_E_QUOTA)
    elif operator == "I":
        quota = dict(_I_QUOTA)
    elif operator == "N":
        quota = dict(_N_QUOTA)
    else:
        quota = None                          # F: uniform per-kind cap instead

    for r in candidates:
        if len(chosen) >= per_operator:
            break
        b = r["base_question_id"]
        if not base_ok(b):
            continue
        fam = r["family"]
        if quota is not None:
            if fam_count.get(fam, 0) >= quota.get(fam, 0):
                continue
        elif operator == "F":
            if fam_count.get(fam, 0) >= _F_MAX_PER_KIND:
                continue
        if operator == "N":
            tk = r["grounding"].get("template", "")
            if tmpl_count.get(tk, 0) >= _N_PER_TEMPLATE_CAP:
                continue
            tmpl_count[tk] = tmpl_count.get(tk, 0) + 1
        chosen.append(r)
        fam_count[fam] = fam_count.get(fam, 0) + 1
        picked_bases.add(b)
        used_bases[b] = used_bases.get(b, 0) + 1

    # top-up: if family quotas could not all be met, fill remaining slots with any
    # base-legal candidate (records the shortfall in the manifest via fam_count).
    if len(chosen) < per_operator:
        for r in candidates:
            if len(chosen) >= per_operator:
                break
            b = r["base_question_id"]
            if not base_ok(b) or r in chosen:
                continue
            if operator == "N":
                tk = r["grounding"].get("template", "")
                if tmpl_count.get(tk, 0) >= _N_PER_TEMPLATE_CAP:
                    continue
                tmpl_count[tk] = tmpl_count.get(tk, 0) + 1
            chosen.append(r)
            fam_count[r["family"]] = fam_count.get(r["family"], 0) + 1
            picked_bases.add(b)
            used_bases[b] = used_bases.get(b, 0) + 1

    return chosen, {"family_counts": fam_count, "target": per_operator,
                    "produced": len(chosen)}


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def build(questions_path, canonical_path, db_path, out_dir, per_operator, max_ops_per_base):
    cases = _load_jsonl(questions_path)
    canon = _load_canonical_map(canonical_path)
    cards = _load_cards(db_path)
    cases.sort(key=lambda c: c.get("question_id", ""))

    # Generate all candidates per operator.
    cand: Dict[str, List[Dict]] = {"F": [], "E": [], "I": [], "N": []}
    for idx, case in enumerate(cases):
        cid = case.get("question_id")
        info = canon.get(cid, {})
        if not info.get("fic_id"):
            continue
        card = cards.get(info["fic_id"], {})
        cand["F"].extend(op_F(case, info, card))
        cand["E"].extend(op_E(case, info, card))
        cand["I"].extend(op_I(case, info, card))
        cand["N"].extend(op_N(case, info, card, template_idx=idx))

    # Audit hard-I supply: why the hard stratum is capped (honest freeze record).
    hard_supply = {"cases_with_hard_rule": 0, "groundable": 0,
                   "phrase_absent": 0, "no_pattern": 0}
    for case in cases:
        info = canon.get(case.get("question_id"), {})
        fic = info.get("fic_id")
        if not fic:
            continue
        hrules = [r for r in cards.get(fic, {}).get("i_rules", [])
                  if r["diagnostic_type"] == "I_HARD"]
        if hrules:
            hard_supply["cases_with_hard_rule"] += 1
        txt = case["question"] + " " + case["context"]
        for r in hrules:
            pat = _PAT_BY_FAMILY.get(_family_of(r["rule_id"]))
            if pat is None:
                hard_supply["no_pattern"] += 1
            elif pat.search(txt):
                hard_supply["groundable"] += 1
            else:
                hard_supply["phrase_absent"] += 1

    # Select with a shared base-dedup budget. Order E,I first (scarcer families),
    # then F, then N (N can attach to almost any base).
    used_bases: Dict[str, int] = {}
    traps: List[Dict] = []
    op_report: Dict[str, Any] = {}
    for op in ("E", "I", "F", "N"):
        picked, rep = _select(cand[op], op, per_operator, used_bases, max_ops_per_base)
        traps.extend(picked)
        op_report[op] = {**rep, "candidates": len(cand[op])}

    # Freeze.
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trap_path = out_dir / "trap_set.jsonl"
    with open(trap_path, "w", encoding="utf-8") as f:
        for rec in traps:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    base_clusters: Dict[str, List[str]] = {}
    for r in traps:
        base_clusters.setdefault(r["base_question_id"], []).append(r["operator"])

    manifest = {
        "protocol": "governance_confirmatory_v2",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "per_operator": per_operator,
        "max_ops_per_base": max_ops_per_base,
        "total_traps": len(traps),
        "needs_review": sum(1 for r in traps if r["needs_review"]),
        "sources": {
            "questions": {"path": questions_path, "sha256": _sha256(questions_path)},
            "canonical": {"path": canonical_path, "sha256": _sha256(canonical_path)},
            "cards_db": {"path": db_path, "sha256": _sha256(db_path)},
        },
        "operators": op_report,
        "distinct_base_questions": len(base_clusters),
        "base_cluster_sizes": {
            "max_ops_on_one_base": max((len(v) for v in base_clusters.values()), default=0),
        },
        "i_hard_supply": hard_supply,
        "known_constraints": [
            "I_HARD is capped by phrase-deletion supply: only "
            f"{hard_supply['groundable']} of {hard_supply['cases_with_hard_rule']} "
            "hard-ambiguity cases carry a deletable disambiguator phrase; the rest are "
            "context-implied or method/basis choices needing hand/Tier-2 construction "
            "(deferred to the ACL-phase hard set). Report I_HARD and I_SOFT FNAR "
            "separately; do not generalize the small hard stratum.",
            "F field-kind coverage is data-limited to kinds present in selected FICs; "
            "'rate'/'time' removals were scarce in this pool.",
        ],
        "trap_set_sha256": _sha256(str(trap_path)),
    }
    manifest_path = out_dir / "trap_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # Console report.
    print(f"[trap-v2] wrote {len(traps)} traps -> {trap_path}")
    for op in ("F", "E", "I", "N"):
        rep = op_report[op]
        print(f"  {op}: {rep['produced']}/{rep['target']}  "
              f"families={rep['family_counts']}  (cands={rep['candidates']})")
    print(f"  distinct base questions: {len(base_clusters)}  "
          f"(max ops on one base: {manifest['base_cluster_sizes']['max_ops_on_one_base']})")
    print(f"  needs_review: {manifest['needs_review']}")
    print(f"[trap-v2] manifest -> {manifest_path}")
    return manifest


def main():
    ap = argparse.ArgumentParser(description="Balanced contract-verified trap generator (v2).")
    ap.add_argument("--questions", required=True)
    ap.add_argument("--canonical", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--per-operator", type=int, default=20)
    ap.add_argument("--max-ops-per-base", type=int, default=2)
    args = ap.parse_args()
    build(args.questions, args.canonical, args.db, args.out_dir,
          args.per_operator, args.max_ops_per_base)


if __name__ == "__main__":
    main()
