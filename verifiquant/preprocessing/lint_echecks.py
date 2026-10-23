"""
lint_echecks.py
---------------
Post-generation gate for FIC E-check expressions.

E-check rules in ``core["diagnostic_checks"]`` are evaluated by the runtime via
``eval()`` as a *single expression* against a restricted namespace
(see ``run_error_classification_pipeline._safe_eval_rule``).  Gemini regularly
emits two failure modes that ``eval()`` cannot handle and that surface at runtime
as spurious **C** (audit_log) early-exits:

  1. Assignment-statement form, e.g.
         ``result = compute(inputs); not (-100 <= result <= 100)``
     -> ``SyntaxError`` (eval cannot run statements).  The canonical-correct
     idiom is inline ``compute(inputs)``:
         ``not (-100 <= compute(inputs) <= 100)``
     This is exactly the bug class that was hand-patched V1->V3 and that
     reappears on every fresh Flash regeneration.

  2. Names not present in the eval namespace (``datetime``, ``json``, ``round``)
     -> ``NameError``.  The namespace is hardened (see ``ECHECK_SAFE_BUILTINS``)
     so these names now resolve; anything still unknown is reported for manual
     review.

This module is the single source of truth for the eval namespace
(``ECHECK_SAFE_BUILTINS`` / ``echeck_allowed_names``) so the linter and the
runtime can never drift.  It exposes:

  - ``lint_core_card(core, autofix=True)``  -> per-card report (mutates in place)
  - ``lint_core_cards(cards, autofix=True)`` -> aggregate report
  - a CLI to re-lint an already-built ``core.jsonl`` without regenerating.

It runs after every card generation (wired into ``fic_generation_pipeline``) and
can also be re-run standalone after any regeneration:

    python3 -m verifiquant.preprocessing.lint_echecks \\
        --core results/<run>/fic/core.jsonl --write \\
        --report results/<run>/fic/echeck_lint_report.json
"""
from __future__ import annotations

import argparse
import ast
import datetime as _datetime
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# --------------------------------------------------------------------------- #
# Single source of truth for the E-check eval namespace.
# run_error_classification_pipeline._safe_eval_rule builds its globals from this
# dict, so the linter's notion of "allowed names" and the runtime's actual
# namespace can never drift apart.
# --------------------------------------------------------------------------- #
ECHECK_SAFE_BUILTINS: Dict[str, Any] = {
    # numeric / sequence builtins
    "abs": abs, "min": min, "max": max, "len": len, "sum": sum,
    "all": all, "any": any, "isinstance": isinstance,
    "float": float, "int": int, "bool": bool, "str": str,
    "list": list, "tuple": tuple, "dict": dict, "set": set,
    "range": range, "zip": zip, "enumerate": enumerate,
    "round": round, "pow": pow,
    # stdlib modules / types that E-check expressions legitimately reference
    "math": math,
    "json": json,
    "datetime": _datetime.datetime,
    "timedelta": _datetime.timedelta,
    "date": _datetime.date,
}

# Names always present in the eval env that are not in ECHECK_SAFE_BUILTINS.
# ``inputs`` (the bound input dict) and ``compute`` (the FIC's compiled function)
# are injected per-card by the runtime; the literal constants resolve natively.
ECHECK_RUNTIME_NAMES: Set[str] = {"inputs", "compute", "True", "False", "None"}

# E-check scope that the runtime actually evaluates (mirrors the pipeline filter).
_EVALUATED_CHECK_TYPES = {"deterministic", "normalization"}


def echeck_allowed_names(input_names: Set[str]) -> Set[str]:
    """All free names that may legally appear in an E-check expression."""
    return set(ECHECK_SAFE_BUILTINS) | ECHECK_RUNTIME_NAMES | set(input_names)


# --------------------------------------------------------------------------- #
# AST helpers
# --------------------------------------------------------------------------- #
def _input_names(core: Dict[str, Any]) -> Set[str]:
    out: Set[str] = set()
    inp = core.get("inputs")
    if isinstance(inp, dict):
        out |= {str(k) for k in inp.keys()}
    elif isinstance(inp, list):
        for x in inp:
            if isinstance(x, dict) and x.get("name"):
                out.add(str(x["name"]))
            elif isinstance(x, str):
                out.add(x)
    return out


def _free_load_names(tree: ast.AST) -> Set[str]:
    """Names looked up in Load context, minus comprehension/lambda-local binds."""
    loaded: Set[str] = set()
    bound: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            loaded.add(node.id)
        elif isinstance(node, ast.comprehension):
            for t in ast.walk(node.target):
                if isinstance(t, ast.Name):
                    bound.add(t.id)
        elif isinstance(node, ast.Lambda):
            for a in list(node.args.args) + list(node.args.posonlyargs) + list(node.args.kwonlyargs):
                bound.add(a.arg)
            if node.args.vararg:
                bound.add(node.args.vararg.arg)
            if node.args.kwarg:
                bound.add(node.args.kwarg.arg)
    return loaded - bound


class _Substituter(ast.NodeTransformer):
    """Replace Load Names found in ``subs`` with their bound expression AST."""

    def __init__(self, subs: Dict[str, ast.expr]) -> None:
        self._subs = subs

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Load) and node.id in self._subs:
            return ast.copy_location(self._subs[node.id], node)
        return node


def _reduce_statements_to_expr(module: ast.Module) -> Optional[str]:
    """Collapse a statement sequence into one boolean expression.

    Handles the observed Gemini failure modes:
      - leading ``import`` lines (the imported names live in the eval env) -> dropped
      - leading ``name = <expr>`` assignments -> inlined into later references
      - a final bare expression statement -> the predicate

    Returns the unparsed single-expression string, or None if the body uses a
    construct we will not silently rewrite (loops, aug-assign, tuple targets...).
    """
    subs: Dict[str, ast.expr] = {}
    final_expr: Optional[ast.expr] = None

    for stmt in module.body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            continue  # names resolved by the hardened eval env
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                return None
            value = _Substituter(subs).visit(stmt.value)
            subs[stmt.targets[0].id] = value
            continue
        if isinstance(stmt, ast.Expr):
            final_expr = _Substituter(subs).visit(stmt.value)
            continue
        return None  # unsupported statement type -> manual review

    if final_expr is None:
        return None
    wrapped = ast.Expression(body=final_expr)
    ast.fix_missing_locations(wrapped)
    return ast.unparse(wrapped)


def normalize_expression(expr: str, allowed: Set[str]) -> Tuple[str, Optional[str], List[str]]:
    """Classify (and where possible auto-fix) a single E-check expression.

    Returns ``(status, fixed_expr_or_None, problems)`` where status is one of:
      - ``"ok"``          : valid expression, all names known; no change
      - ``"autofixed"``   : was unparseable as an expression; rewritten to a
                            valid single expression with all names known
      - ``"unknown_name"``: parses as an expression but references names absent
                            from the eval env (NameError at runtime); ``fixed``
                            may carry an auto-fix that still has unknown names
      - ``"syntax_error"``: cannot be reduced to a valid single expression
    """
    expr = (expr or "").strip()
    if not expr:
        return "syntax_error", None, ["empty expression"]

    # 1) Already a valid single expression?
    try:
        tree = ast.parse(expr, mode="eval")
        unknown = _free_load_names(tree) - allowed
        if not unknown:
            return "ok", None, []
        return "unknown_name", None, [f"unknown name(s): {sorted(unknown)}"]
    except SyntaxError:
        pass

    # 2) Statement sequence -> try to inline into a single expression.
    try:
        module = ast.parse(expr, mode="exec")
    except SyntaxError as exc:
        return "syntax_error", None, [f"unparseable: {exc.msg}"]

    reduced = _reduce_statements_to_expr(module)
    if reduced is None:
        return "syntax_error", None, ["statement form cannot be reduced to a single expression"]

    try:
        rtree = ast.parse(reduced, mode="eval")
    except SyntaxError as exc:
        return "syntax_error", None, [f"auto-fix produced invalid expression: {exc.msg}"]

    unknown = _free_load_names(rtree) - allowed
    if unknown:
        return "unknown_name", reduced, [f"auto-fix ok but unknown name(s): {sorted(unknown)}"]
    return "autofixed", reduced, []


# --------------------------------------------------------------------------- #
# Card-level / batch API
# --------------------------------------------------------------------------- #
def lint_core_card(core: Dict[str, Any], *, autofix: bool = True) -> Dict[str, Any]:
    """Lint (and optionally auto-fix) one core card's E-check expressions.

    Mutates ``core['diagnostic_checks'][*]['expression']`` in place when a fix is
    applied, and stamps a summary onto ``core['source_meta']['echeck_lint']``.
    """
    fic_id = str(core.get("fic_id", ""))
    allowed = echeck_allowed_names(_input_names(core))
    fixed: List[Dict[str, Any]] = []
    unresolved: List[Dict[str, Any]] = []

    for chk in core.get("diagnostic_checks", []) or []:
        if not isinstance(chk, dict):
            continue
        expr = chk.get("expression")
        if not isinstance(expr, str) or not expr.strip():
            continue
        in_scope = (
            str(chk.get("diagnostic_type", "")).upper() == "E"
            and str(chk.get("check_type", "")).lower() in _EVALUATED_CHECK_TYPES
        )
        status, new_expr, problems = normalize_expression(expr, allowed)
        if status == "ok":
            continue
        rule_id = str(chk.get("rule_id", "")) or "<unknown>"
        record = {
            "fic_id": fic_id,
            "rule_id": rule_id,
            "diagnostic_type": str(chk.get("diagnostic_type", "")),
            "check_type": str(chk.get("check_type", "")),
            "runtime_evaluated": in_scope,
            "status": status,
            "original": expr,
            "fixed": new_expr,
            "problems": problems,
        }
        if status == "autofixed" and autofix and new_expr:
            chk["expression"] = new_expr
            fixed.append(record)
        else:
            unresolved.append(record)

    summary = {
        "fic_id": fic_id,
        "fixed_count": len(fixed),
        "unresolved_count": len(unresolved),
        "fixed": fixed,
        "unresolved": unresolved,
    }
    if fixed or unresolved:
        meta = core.get("source_meta")
        if not isinstance(meta, dict):
            meta = {}
            core["source_meta"] = meta
        meta["echeck_lint"] = {
            "fixed_count": len(fixed),
            "unresolved_count": len(unresolved),
            "fixed_rule_ids": [r["rule_id"] for r in fixed],
            "unresolved_rule_ids": [r["rule_id"] for r in unresolved],
        }
    return summary


def lint_core_cards(cards: List[Dict[str, Any]], *, autofix: bool = True) -> Dict[str, Any]:
    """Lint a list of core cards; returns an aggregate report."""
    per_card = [lint_core_card(c, autofix=autofix) for c in cards]
    fixed = [r for s in per_card for r in s["fixed"]]
    unresolved = [r for s in per_card for r in s["unresolved"]]
    return {
        "cards_total": len(cards),
        "cards_with_fixes": sum(1 for s in per_card if s["fixed_count"]),
        "cards_with_unresolved": sum(1 for s in per_card if s["unresolved_count"]),
        "fixed_count": len(fixed),
        "unresolved_count": len(unresolved),
        "fixed": fixed,
        "unresolved": unresolved,
    }


# --------------------------------------------------------------------------- #
# CLI — re-lint an already-built core.jsonl without regenerating.
# --------------------------------------------------------------------------- #
def _load_core(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    return payload if isinstance(payload, list) else [payload]


def _dump_core(path: Path, cards: List[Dict[str, Any]]) -> None:
    if path.suffix.lower() == ".jsonl":
        with path.open("w", encoding="utf-8") as fh:
            for row in cards:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        path.write_text(json.dumps(cards, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lint / auto-fix FIC E-check expressions in a core.jsonl (post-generation gate)."
    )
    parser.add_argument("--core", type=Path, required=True, help="fic_core JSON/JSONL to lint")
    parser.add_argument("--write", action="store_true", help="Apply auto-fixes back to --core (or --out).")
    parser.add_argument("--out", type=Path, help="Where to write fixed cards (defaults to --core when --write).")
    parser.add_argument("--report", type=Path, help="Optional JSON report path.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any unresolved (non-auto-fixable) E-check remains.",
    )
    args = parser.parse_args()

    cards = _load_core(args.core)
    report = lint_core_cards(cards, autofix=True)

    print(
        f"[echeck-lint] cards={report['cards_total']} "
        f"fixed={report['fixed_count']} (on {report['cards_with_fixes']} cards) "
        f"unresolved={report['unresolved_count']} (on {report['cards_with_unresolved']} cards)"
    )
    for r in report["fixed"]:
        print(f"  [fixed] {r['fic_id']} :: {r['rule_id']}")
        print(f"          - {r['original']}")
        print(f"          + {r['fixed']}")
    for r in report["unresolved"]:
        print(f"  [MANUAL] {r['fic_id']} :: {r['rule_id']} ({r['status']}) -> {r['problems']}")
        print(f"           {r['original']}")

    if args.write:
        out = args.out or args.core
        _dump_core(out, cards)
        print(f"[echeck-lint] wrote {len(cards)} cards to {out}")
    elif report["fixed_count"]:
        print("[echeck-lint] (dry run — pass --write to apply auto-fixes)")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[echeck-lint] report written to {args.report}")

    if args.strict and report["unresolved_count"]:
        raise SystemExit(f"{report['unresolved_count']} unresolved E-check(s) require manual fix.")


if __name__ == "__main__":
    main()
