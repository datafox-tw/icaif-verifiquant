from __future__ import annotations

import ast
import json
import math
import re
from typing import Any, Dict, List, Optional

from verifiquant.preprocessing.common import (
    ConversionInput,
    call_gemini_json,
    normalize_input_type,
    normalize_label,
    optional_str,
    require_compute_code,
    safe_id,
    safe_fic_import,
)
from verifiquant.taxonomy import TAXONOMY, taxonomy_json, validate_domain_topic

try:
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover
    genai_types = None


STAGE_CORE_PROMPT = """Role:
You are a financial software engineer generating a FIC core card.

Task:
Convert one dataset case into a reusable `fic_core` object with:
- fic_id, name, short_description
- domain, topic
- inputs, output
- execution (deterministic python)
- selection_hints
- semantic_hints
- diagnostic_checks

Critical rules:
1) `python_solution` is PRIMARY for executable logic.
2) `function` is SECONDARY for naming and financial semantics.
3) Do not hardcode case-specific constants into execution; use inputs.
4) Use `article_content_excerpt` as supporting context to generate higher-quality `semantic_hints`.
5) If a diagnostic rule expression needs to validate the formula's output, you MUST call `compute(inputs)` to obtain the output value. You CANNOT access output variables directly from the `inputs` dictionary.

Input/output type policy:
- Allowed input types: `number`, `integer`, `boolean`, `string`, `array[number]`,
  `array[array[number]]`, `array[object]`, `dict[string,number]`, and `object`.
- Use `dict[string,number]` for mappings such as department square footage or
  credit-rating proportions.
- Use `array[object]` for lists of structured records such as orders with
  `quantity` and `price` keys.
- Use `array[array[number]]` for numeric matrices such as correlation matrices.
- Outputs MUST be scalar for now (`number`, `integer`, `boolean`, or `string`).
  If the source function returns a dictionary but the question asks for one
  field, make `compute(inputs)` return that requested scalar field directly.
  Do NOT emit nested dictionary outputs unless the runtime contract is extended.

Taxonomy policy:
- Domain MUST be one of taxonomy domains.
- Topic should prefer existing topics under that domain.
- If no existing topic is suitable, you MAY create a concise snake_case topic under that existing domain.
- Never invent a new domain.

Diagnostic rules policy:
- Provide 3 to 8 checks.
- check_type must be one of: deterministic, normalization.
- deterministic checks must include a valid python boolean `expression` using input names. If the expression needs to check the computed output, you MUST call `compute(inputs)` instead of reading the output name from the `inputs` dictionary.
- For deterministic/normalization checks, include `predicate_mode`:
  - `violation`: expression=True means the rule is triggered (preferred default).
  - `validity`: expression=True means the input is valid, and expression=False triggers the rule.
- In diagnostics checks, use ONLY F or E. Do NOT output M/N/I/C.
- Why: M is handled at retrieval/card-selection stage before card commitment.
- Mapping guidance:
  - F is STRICTLY for schema/spec incompleteness only:
    - required input missing
    - input cannot be parsed to expected type
    - malformed structure (for example expected list but got scalar and cannot normalize)
    - for array inputs whose elements are tuples/pairs (e.g. each element must be
      [quantity, price]), the STRUCTURE check (each element is a list/tuple of the
      expected length) is F — because malformed structure means the input cannot be
      parsed to the expected schema, even if numbers were provided.
  - E is for value-level or interpretation-level issues AFTER inputs exist:
    - unit/scale mismatch (8 vs 0.08, 15 vs 0.15)
    - bound/plausibility violations (negative rate, impossible ratio)
    - binding/direction/frequency inconsistencies
  - Prefer E over F when a value is present but suspicious or needs confirmation.
  - Do NOT classify percentage-range checks as F when the value exists; classify them as E with deterministic alert.
  - "Hard formula preconditions" should generally be E unless the variable is truly missing/unparseable.
  - If an array input is described as a list of (quantity, price) pairs or similar
    multi-element tuples, you MUST include an F-class check that verifies the structure
    of each element (e.g. `all(isinstance(item, (list, tuple)) and len(item) == 2 for item in inputs['field'])`)
    with predicate_mode="validity".
- severity must be error or alert.
- semantic_hints are for I-gate only (hidden ambiguity checks like FX direction/time basis),
  and must include clarification question plus fixed options.
- semantic_hints must be traceable to this case's function/question/context/article excerpt.
- Do not output generic options like "Option A/Option B".
- Keep semantic_hints concise and bounded: at most {semantic_hint_max} hints per card.
- For each semantic_hint include:
  - id
  - ambiguity_type
  - trigger_signal
  - clarification_question
  - options  ← NOW STRUCTURED (see below)
  - i_level (hard|soft)
  - assumption_if_not_clarified
  - impact_scope (input_binding|output_interpretation|directionality)

Semantic hint options policy (IMPORTANT — options are now objects, not strings):
Each option must be:
  {{ "label": "...", "value": "...", "is_default": true/false, "transform_spec": null | <spec> }}

- Exactly one option must have is_default=true (the baseline the FIC code already implements).
- The DEFAULT option's transform_spec MUST be null. The FIC code IS the default — no
  post-processing is needed for the baseline interpretation. Do not write a trivial
  identity transform like `result_expr: "result"` for the default.
- A non-default option MAY include a transform_spec ONLY IF the alternative interpretation
  has a pure-arithmetic relationship to the default output. If no such algebraic relationship
  exists (e.g. the alternative would require choosing a different input source, rescaling
  an INPUT before computation, or recomputing from scratch), the option MUST have
  transform_spec=null.
- result_expr is a PURE ARITHMETIC EXPRESSION. It MUST NOT contain function calls of any
  kind — no compute(), no exp/log/sin/cos, no min/max, no abs. Only arithmetic operators
  (+, -, *, /, **) over `result` and input names. If the only way to express the
  alternative is via a function call, set transform_spec=null.
- Reference inputs by BARE NAME ONLY. Write `discount_rate`, not `inputs['discount_rate']`.
  Write `initial_investment`, not `inputs["initial_investment"]`. Subscript notation will
  be REJECTED by the verifier.
- ONLY use names that actually appear in this FIC's `inputs` list. If you need a quantity
  that is not in the inputs (e.g. `inflation_rate` for a real-rate transform on a FIC that
  only has nominal-rate inputs), set transform_spec=null — do NOT fabricate input names.
- result_expr MUST reference `result`. It is a transformation OF the original output, not
  a recomputation of the formula. If the alternative requires recomputing from scratch
  (rather than adjusting `result`), set transform_spec=null.
- transform_spec schema:
  {{
    "patch_type": "result_postprocess",
    "result_expr": "<Python expression using `result` and any input variable names>",
    "max_expr_nodes": <int, expected AST node count of result_expr>,
    "invariant": "<a Python `lhs == rhs` equality over result_old / result_new / input names>",
    "affected_inputs": ["<list of input names referenced in result_expr / invariant>"]
  }}
- result_expr rules:
  * Use only `result` (the output of the original execution) and names that appear in the FIC inputs.
  * No function calls, no imports, no side effects. Pure arithmetic expression only.

WHEN TO USE transform_spec — domain-driven guidance.
Pick the ambiguity that genuinely applies to THIS FIC's domain/inputs. Do NOT copy patterns
from unrelated domains. The hint must be traceable to the FIC's function/inputs.

| FIC domain / topic family                | Typical I-class ambiguity            | result_expr pattern                                  |
|------------------------------------------|--------------------------------------|------------------------------------------------------|
| time_value_money / annuity / NPV / PV    | end vs beginning of period (annuity) | `(result + initial_investment) * (1 + r) - initial_investment` |
| time_value_money / compounding           | discrete vs continuous compounding   | depends on formula; often `result * exp(r) / (1+r)`  |
| interest_rates / yield                   | nominal APR ↔ effective APY          | `(1 + result/n) ** n - 1`                            |
| interest_rates / rate conversion         | monthly ↔ annual rate                | `(1 + result) ** 12 - 1` or `(1 + result) ** (1/12) - 1` |
| any rate output                          | percent vs decimal scale             | `result / 100.0` or `result * 100.0`                 |
| fx / foreign_exchange                    | direct vs indirect quote             | `1.0 / result`                                       |
| cash_flow / valuation                    | inflow-positive vs outflow-positive  | `-result`                                            |
| returns / performance                    | pre-tax vs post-tax                  | `result * (1 - tax_rate)`                            |
| returns / performance                    | nominal vs real                      | `(1 + result) / (1 + inflation_rate) - 1`            |
| risk / volatility                        | period vol vs annualized             | `result * (252 ** 0.5)` or `result * (12 ** 0.5)`    |
| statistics / dispersion                  | sample vs population variance        | `result * (n - 1) / n`                               |
| bonds                                    | clean vs dirty price                 | `result + accrued_interest`                          |

If THIS FIC's domain is not in the table above, do NOT invent a forced transform_spec —
prefer transform_spec=null and let the system fall back to clarification-only.

ANTI-PATTERNS — set transform_spec=null in these cases:
1. The ambiguity is about WHICH INPUT to use (e.g. "use WACC vs cost of equity" for
   discount_rate, "use S&P 500 vs Treasury" for benchmark). No post-processing can fix
   input choice — the user must rerun with the right input.
2. The alternative interpretation requires re-running the formula with a structurally
   different input (e.g. "use only operating cash flows vs all cash flows").
3. The ambiguity is purely semantic categorization (e.g. "is this a hurdle rate or
   required return?") with no numerical adjustment.
4. The mapping would need to change loop logic or formula shape (not a closed-form
   post-process of `result`). Leave it null; future work will support code_patch.

WORKED EXAMPLE 1 — NPV / annuity-due (time_value_money):
{{
  "id": "payment_timing",
  "ambiguity_type": "time_basis",
  "trigger_signal": "payment period not explicitly stated as end-of-period",
  "clarification_question": "Are cash flows paid at END of each period (ordinary annuity) or BEGINNING (annuity-due)?",
  "i_level": "soft",
  "assumption_if_not_clarified": "End of period (ordinary annuity)",
  "impact_scope": "output_interpretation",
  "options": [
    {{ "label": "End of period — DEFAULT", "value": "end_of_period", "is_default": true,  "transform_spec": null }},
    {{
      "label": "Beginning of period (annuity-due)", "value": "beginning_of_period", "is_default": false,
      "transform_spec": {{
        "patch_type": "result_postprocess",
        "result_expr": "(result + initial_investment) * (1 + discount_rate) - initial_investment",
        "max_expr_nodes": 15,
        "invariant": "result_new + initial_investment == (result_old + initial_investment) * (1 + discount_rate)",
        "affected_inputs": ["discount_rate", "initial_investment"]
      }}
    }}
  ]
}}

WORKED EXAMPLE 2 — monthly ↔ annual rate (interest_rates):
{{
  "id": "rate_period_basis",
  "ambiguity_type": "rate_period",
  "trigger_signal": "rate may be quoted monthly or annually",
  "clarification_question": "Is the rate of return quoted on a monthly or annual basis?",
  "i_level": "soft",
  "assumption_if_not_clarified": "Annual rate",
  "impact_scope": "output_interpretation",
  "options": [
    {{ "label": "Annual — DEFAULT", "value": "annual", "is_default": true, "transform_spec": null }},
    {{
      "label": "Monthly (annualize via compounding)", "value": "monthly", "is_default": false,
      "transform_spec": {{
        "patch_type": "result_postprocess",
        "result_expr": "(1 + result) ** 12 - 1",
        "max_expr_nodes": 7,
        "invariant": "result_new == (1 + result_old) ** 12 - 1",
        "affected_inputs": []
      }}
    }}
  ]
}}

WORKED EXAMPLE 3 — percent vs decimal output (any rate-returning FIC):
{{
  "id": "output_scale",
  "ambiguity_type": "unit_scale",
  "trigger_signal": "question asks for percentage, FIC outputs decimal (or vice versa)",
  "clarification_question": "Should the answer be expressed as a decimal (0.08) or percent (8.0)?",
  "i_level": "soft",
  "assumption_if_not_clarified": "Decimal",
  "impact_scope": "output_interpretation",
  "options": [
    {{ "label": "Decimal — DEFAULT", "value": "decimal", "is_default": true, "transform_spec": null }},
    {{
      "label": "Percent", "value": "percent", "is_default": false,
      "transform_spec": {{
        "patch_type": "result_postprocess",
        "result_expr": "result * 100.0",
        "max_expr_nodes": 3,
        "invariant": "result_new == result_old * 100.0",
        "affected_inputs": []
      }}
    }}
  ]
}}

WORKED EXAMPLE 4 — ANTI-PATTERN (benchmark choice, transform_spec MUST be null):
{{
  "id": "benchmark_source",
  "ambiguity_type": "input_binding",
  "trigger_signal": "benchmark return source not explicitly named",
  "clarification_question": "Is the benchmark return based on the S&P 500, a sector ETF, or the risk-free rate?",
  "i_level": "soft",
  "assumption_if_not_clarified": "Market benchmark provided in inputs",
  "impact_scope": "input_binding",
  "options": [
    {{ "label": "Use given benchmark_return as-is — DEFAULT", "value": "as_given", "is_default": true, "transform_spec": null }},
    {{ "label": "Reconsider with alternative benchmark", "value": "alternative_benchmark", "is_default": false, "transform_spec": null }}
  ]
}}
(Why transform_spec is null here: choosing a different benchmark means re-running the formula
with different input values — there is no closed-form expression mapping `result` from one
benchmark to another.)

Input case:
<DEFINITION_JSON>
{definition_json}
</DEFINITION_JSON>

Taxonomy:
<TAXONOMY_JSON>
{taxonomy_json}
</TAXONOMY_JSON>

Output:
Return JSON only.
"""

SEMANTIC_HINT_MAX = 6

def stage_core_schema() -> Any:
    if genai_types is None:
        return None

    transform_spec_schema = genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "patch_type": genai_types.Schema(
                type=genai_types.Type.STRING,
                enum=["result_postprocess"],
            ),
            "result_expr": genai_types.Schema(type=genai_types.Type.STRING),
            "max_expr_nodes": genai_types.Schema(type=genai_types.Type.INTEGER),
            "invariant": genai_types.Schema(type=genai_types.Type.STRING),
            "affected_inputs": genai_types.Schema(
                type=genai_types.Type.ARRAY,
                items=genai_types.Schema(type=genai_types.Type.STRING),
            ),
        },
        required=["patch_type", "result_expr", "max_expr_nodes", "invariant", "affected_inputs"],
    )

    hint_option_schema = genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "label": genai_types.Schema(type=genai_types.Type.STRING),
            "value": genai_types.Schema(type=genai_types.Type.STRING),
            "is_default": genai_types.Schema(type=genai_types.Type.BOOLEAN),
            # transform_spec is omitted from required; null for default option.
            "transform_spec": transform_spec_schema,
        },
        required=["label", "value", "is_default"],
    )

    semantic_hint = genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "id": genai_types.Schema(type=genai_types.Type.STRING),
            "ambiguity_type": genai_types.Schema(type=genai_types.Type.STRING),
            "trigger_signal": genai_types.Schema(type=genai_types.Type.STRING),
            "clarification_question": genai_types.Schema(type=genai_types.Type.STRING),
            "options": genai_types.Schema(
                type=genai_types.Type.ARRAY,
                items=hint_option_schema,
            ),
            "i_level": genai_types.Schema(
                type=genai_types.Type.STRING,
                enum=["hard", "soft"],
            ),
            "assumption_if_not_clarified": genai_types.Schema(type=genai_types.Type.STRING),
            "impact_scope": genai_types.Schema(
                type=genai_types.Type.STRING,
                enum=["input_binding", "output_interpretation", "directionality"],
            ),
        },
        required=[
            "id",
            "ambiguity_type",
            "trigger_signal",
            "clarification_question",
            "options",
            "i_level",
            "assumption_if_not_clarified",
            "impact_scope",
        ],
    )
    check = genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "rule_id": genai_types.Schema(type=genai_types.Type.STRING),
            "name": genai_types.Schema(type=genai_types.Type.STRING),
            "check_type": genai_types.Schema(
                type=genai_types.Type.STRING,
                enum=["deterministic", "normalization"],
            ),
            "diagnostic_type": genai_types.Schema(
                type=genai_types.Type.STRING,
                enum=["F", "E"],
            ),
            "severity": genai_types.Schema(
                type=genai_types.Type.STRING,
                enum=["error", "alert"],
            ),
            "predicate_mode": genai_types.Schema(
                type=genai_types.Type.STRING,
                enum=["violation", "validity"],
            ),
            "expression": genai_types.Schema(type=genai_types.Type.STRING),
            "applies_to": genai_types.Schema(
                type=genai_types.Type.ARRAY,
                items=genai_types.Schema(type=genai_types.Type.STRING),
            ),
            "description": genai_types.Schema(type=genai_types.Type.STRING),
        },
        required=["rule_id", "name", "check_type", "diagnostic_type", "severity", "description"],
    )

    return genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "fic_id": genai_types.Schema(type=genai_types.Type.STRING),
            "name": genai_types.Schema(type=genai_types.Type.STRING),
            "short_description": genai_types.Schema(type=genai_types.Type.STRING),
            "domain": genai_types.Schema(type=genai_types.Type.STRING),
            "topic": genai_types.Schema(type=genai_types.Type.STRING),
            "selection_hints": genai_types.Schema(
                type=genai_types.Type.ARRAY,
                items=genai_types.Schema(type=genai_types.Type.STRING),
            ),
            "semantic_hints": genai_types.Schema(
                type=genai_types.Type.ARRAY,
                items=semantic_hint,
            ),
            "inputs": genai_types.Schema(
                type=genai_types.Type.ARRAY,
                items=genai_types.Schema(
                    type=genai_types.Type.OBJECT,
                    properties={
                        "name": genai_types.Schema(type=genai_types.Type.STRING),
                        "type": genai_types.Schema(type=genai_types.Type.STRING),
                        "required": genai_types.Schema(type=genai_types.Type.BOOLEAN),
                        "description": genai_types.Schema(type=genai_types.Type.STRING),
                        "unit": genai_types.Schema(type=genai_types.Type.STRING),
                        "aliases": genai_types.Schema(
                            type=genai_types.Type.ARRAY,
                            items=genai_types.Schema(type=genai_types.Type.STRING),
                        ),
                    },
                    required=["name", "type", "required", "description"],
                ),
            ),
            "output": genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "name": genai_types.Schema(type=genai_types.Type.STRING),
                    "type": genai_types.Schema(type=genai_types.Type.STRING),
                    "unit": genai_types.Schema(type=genai_types.Type.STRING),
                    "description": genai_types.Schema(type=genai_types.Type.STRING),
                },
                required=["name", "type"],
            ),
            "execution": genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "language": genai_types.Schema(type=genai_types.Type.STRING),
                    "entrypoint": genai_types.Schema(type=genai_types.Type.STRING),
                    "code": genai_types.Schema(type=genai_types.Type.STRING),
                    "deterministic": genai_types.Schema(type=genai_types.Type.BOOLEAN),
                },
                required=["language", "entrypoint", "code", "deterministic"],
            ),
            "diagnostic_checks": genai_types.Schema(
                type=genai_types.Type.ARRAY,
                items=check,
            ),
        },
        required=[
            "fic_id",
            "name",
            "short_description",
            "domain",
            "topic",
            "selection_hints",
            "semantic_hints",
            "inputs",
            "output",
            "execution",
            "diagnostic_checks",
        ],
    )


def build_stage_core_prompt(defn: ConversionInput) -> str:
    definition_json = json.dumps(
        {
            "source_meta": defn.source_meta,
            "article_title": defn.article_title,
            "article_doc_id": defn.article_doc_id,
            "article_content_excerpt": defn.article_content_excerpt,
            "function": defn.function,
            "python_solution": defn.python_solution,
            "context": defn.context,
            "question": defn.question,
        },
        ensure_ascii=False,
        indent=2,
    )
    return STAGE_CORE_PROMPT.format(
        definition_json=definition_json,
        taxonomy_json=taxonomy_json(indent=2),
        semantic_hint_max=SEMANTIC_HINT_MAX,
    )


def _normalize_input_item(item: Dict[str, Any]) -> Dict[str, Any]:
    aliases = item.get("aliases", [])
    if not isinstance(aliases, list):
        aliases = []
    return {
        "name": normalize_label(item.get("name", "")),
        "type": normalize_input_type(item.get("type", "number")),
        "required": bool(item.get("required", True)),
        "description": str(item.get("description", "")).strip(),
        "unit": optional_str(item.get("unit")),
        "aliases": [normalize_label(a) for a in aliases if normalize_label(a)],
    }


def _normalize_output(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": normalize_label(item.get("name", "")),
        "type": normalize_input_type(item.get("type", "number")),
        "unit": optional_str(item.get("unit")),
        "description": optional_str(item.get("description")) or "",
    }


def _infer_predicate_mode_from_expression(expression: str) -> str:
    expr = "".join(str(expression or "").strip().lower().split())
    if not expr:
        return "violation"
    if expr.startswith("not(") or expr.startswith("not"):
        return "violation"
    if any(tok in expr for tok in ("<0", "<=0", "<=-1", "==0", "!=", "isnone")):
        return "violation"
    if any(tok in expr for tok in (">=0", ">0", ">-1", "<=1", "<=10")):
        return "validity"
    if "<=" in expr or ">=" in expr:
        return "validity"
    if "<" in expr and "<0" not in expr:
        return "validity"
    return "violation"


def _normalize_check(check: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "rule_id": normalize_label(check.get("rule_id", "")),
        "name": str(check.get("name", "")).strip(),
        "check_type": str(check.get("check_type", "")).strip().lower(),
        "diagnostic_type": str(check.get("diagnostic_type", "")).strip().upper(),
        "severity": str(check.get("severity", "")).strip().lower(),
        "predicate_mode": str(check.get("predicate_mode", "")).strip().lower(),
        "expression": str(check.get("expression", "") or "").strip(),
        "applies_to": [normalize_label(x) for x in (check.get("applies_to") or []) if normalize_label(x)],
        "description": str(check.get("description", "")).strip(),
    }

    if not out["rule_id"]:
        out["rule_id"] = f"rule_{normalize_label(out['name']) or 'generated'}"
    if out["check_type"] not in {"deterministic", "normalization"}:
        out["check_type"] = "normalization"
    # Core checks are post-card-commit guards, so they should be F/E only.
    if out["diagnostic_type"] == "M":
        if out["check_type"] == "normalization":
            out["diagnostic_type"] = "E"
        else:
            out["diagnostic_type"] = "F"
    elif out["diagnostic_type"] not in {"F", "E"}:
        out["diagnostic_type"] = "E"
    if out["severity"] not in {"error", "alert"}:
        out["severity"] = "alert"
    if out["predicate_mode"] not in {"violation", "validity"}:
        out["predicate_mode"] = _infer_predicate_mode_from_expression(out["expression"])
    if not out["expression"]:
        raise ValueError(f"check '{out['rule_id']}' missing expression")
    if not out["description"]:
        out["description"] = out["name"] or out["rule_id"]
    return out


def _norm_str_list(items: Any) -> List[str]:
    if not isinstance(items, list):
        return []
    out: List[str] = []
    for item in items:
        s = str(item or "").strip()
        if s:
            out.append(s)
    return out


def _normalize_hint_option(opt: Any) -> Any:
    """Normalize one semantic hint option.

    Accepts both legacy string format and new structured dict format.
    For dicts, preserves transform_spec so stage_repair._global_i_rules()
    can embed it into the repair rule's transform_map.
    """
    if isinstance(opt, str):
        text = opt.strip()
        return text if text else None
    if not isinstance(opt, dict):
        return None
    label = str(opt.get("label", "") or "").strip()
    value = str(opt.get("value", "") or "").strip() or label
    if not value:
        return None
    out: Dict[str, Any] = {
        "label": label or value,
        "value": value,
        "is_default": bool(opt.get("is_default", False)),
    }
    spec = opt.get("transform_spec")
    # Default options must never carry a transform_spec: the FIC code IS the default.
    # Strip trivial identity transforms and any spec on default options to keep the
    # downstream transform_map clean (stage_repair already filters by is_default).
    if out["is_default"]:
        return out
    if isinstance(spec, dict) and spec:
        spec = dict(spec)
        result_expr = (spec.get("result_expr") or "").strip()
        # Drop trivial identity (Gemini sometimes emits this for non-default options too).
        if result_expr in ("", "result"):
            return out
        # Reject subscript-style references (`inputs['name']`) — verifier will fail them.
        if "inputs[" in result_expr or "inputs ['" in result_expr:
            return out
        # result_expr must reference `result`. If it doesn't, it's a recomputation,
        # not a post-process — drop it.
        try:
            import ast as _ast
            tree = _ast.parse(result_expr, mode="eval")
            # Reject any function calls (e.g. round(), exp(), compute()).
            if any(isinstance(n, _ast.Call) for n in _ast.walk(tree)):
                return out
            names_used = {n.id for n in _ast.walk(tree) if isinstance(n, _ast.Name)}
            if "result" not in names_used:
                return out
            actual_nodes = sum(1 for _ in _ast.walk(tree))
            spec["max_expr_nodes"] = actual_nodes
        except SyntaxError:
            return out
        out["transform_spec"] = spec
    return out


def _normalize_hint_options(raw: Any) -> List[Any]:
    """Normalize the full options list for a semantic hint.

    Returns a mixed list: strings (legacy) or dicts (new structured format).
    Falls back to two placeholder strings if nothing valid comes through.
    """
    if not isinstance(raw, list):
        return ["Confirm intended interpretation", "Use default assumption"]
    out = [_normalize_hint_option(o) for o in raw]
    out = [o for o in out if o is not None]
    if not out:
        return ["Confirm intended interpretation", "Use default assumption"]
    return out


def _normalize_semantic_hint(item: Dict[str, Any], idx: int) -> Dict[str, Any]:
    hint_id = normalize_label(item.get("id", "")) or f"semantic_hint_{idx}"
    ambiguity_type = normalize_label(item.get("ambiguity_type", "")) or "generic_ambiguity"
    trigger_signal = str(item.get("trigger_signal", "") or "").strip()
    clarification_question = str(item.get("clarification_question", "") or "").strip()
    options = _normalize_hint_options(item.get("options", []))
    i_level = normalize_label(item.get("i_level", "")) or "soft"
    if i_level not in {"hard", "soft"}:
        i_level = "soft"
    assumption_if_not_clarified = str(item.get("assumption_if_not_clarified", "") or "").strip()
    impact_scope = normalize_label(item.get("impact_scope", "")) or "output_interpretation"
    if impact_scope not in {"input_binding", "output_interpretation", "directionality"}:
        impact_scope = "output_interpretation"
    if not clarification_question:
        clarification_question = "Please clarify the intended financial interpretation."
    if not assumption_if_not_clarified:
        assumption_if_not_clarified = "Proceed with default interpretation inferred from context."
    return {
        "id": hint_id,
        "ambiguity_type": ambiguity_type,
        "trigger_signal": trigger_signal,
        "clarification_question": clarification_question,
        "options": options,
        "i_level": i_level,
        "assumption_if_not_clarified": assumption_if_not_clarified,
        "impact_scope": impact_scope,
    }


def _dummy_input_value(input_type: str, name: str) -> Any:
    t = str(input_type or "").strip().lower()
    n = str(name or "").strip().lower()
    if t == "integer":
        if n == "year" or n.endswith("_year"):
            return 2000
        if "year" in n or "period" in n or "term" in n:
            return 2
        return 1
    if t == "boolean":
        return True
    if t == "string":
        if "option_type" in n or n == "option":
            return "call"
        if "date" in n:
            return "2020-01-01"
        if "bracket" in n:
            return json.dumps({"0": 0.26, "100000": 0.28})
        return "x"
    if t == "array[number]":
        return [1.0, 2.0, 3.0, 4.0, 5.0]
    if t == "array[array[number]]":
        return [[1.0, 0.0], [0.0, 1.0]]
    if t == "array[object]":
        if "buy" in n or "bid" in n:
            return [{"quantity": 100, "price": 50}, {"quantity": 50, "price": 48}]
        if "sell" in n or "ask" in n:
            return [{"quantity": 80, "price": 49}, {"quantity": 70, "price": 50}]
        return [{"value": 1.0}, {"value": 2.0}]
    if t == "dict[string,number]":
        if "rating" in n and "map" in n:
            return {"AAA": 1.0, "AA": 2.0, "BBB": 4.0}
        if "rating" in n or "proportion" in n:
            return {"AAA": 0.5, "BBB": 0.5}
        if "square" in n or "department" in n:
            return {"Sales": 3000.0, "R&D": 5000.0, "Administration": 2000.0}
        if "limit" in n:
            return {"2015": 5500.0, "2016": 5500.0}
        if "contribution" in n:
            return {"2015": 6000.0, "2016": 7500.0}
        return {"A": 1.0, "B": 2.0}
    if t == "object":
        return {"value": 1.0}
    if "rate" in n or "yield" in n or "ratio" in n:
        return 0.1
    return 1.0


class _InputAutoFixer(ast.NodeTransformer):
    def __init__(self, canonical_name_by_lower: Dict[str, str]) -> None:
        super().__init__()
        self.canonical_name_by_lower = canonical_name_by_lower
        self.fixes: List[str] = []
        self.unknown_input_keys: List[str] = []

    def _canonicalize_key(self, key: str) -> str:
        return self.canonical_name_by_lower.get(key.lower(), key)

    def _extract_subscript_key(self, node: ast.Subscript) -> Optional[str]:
        slice_node = node.slice
        if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
            return slice_node.value
        return None

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == "inputs":
            original = str(node.attr)
            # Keep dict API method calls untouched, e.g. inputs.get("x").
            if original in {
                "get",
                "keys",
                "values",
                "items",
                "pop",
                "setdefault",
                "update",
                "copy",
                "clear",
                "__getitem__",
                "__setitem__",
                "__contains__",
            }:
                return node
            canonical = self._canonicalize_key(original)
            if canonical != original:
                self.fixes.append(f"inputs.{original} -> inputs['{canonical}']")
            else:
                self.fixes.append(f"inputs.{original} -> inputs['{original}']")
            return ast.copy_location(
                ast.Subscript(
                    value=ast.Name(id="inputs", ctx=ast.Load()),
                    slice=ast.Constant(value=canonical),
                    ctx=node.ctx,
                ),
                node,
            )
        return node

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        self.generic_visit(node)
        if not isinstance(node.value, ast.Name) or node.value.id != "inputs":
            return node
        key = self._extract_subscript_key(node)
        if key is None:
            return node
        canonical = self._canonicalize_key(key)
        if canonical != key:
            self.fixes.append(f"inputs['{key}'] -> inputs['{canonical}']")
            node.slice = ast.Constant(value=canonical)
        elif canonical.lower() not in self.canonical_name_by_lower:
            self.unknown_input_keys.append(key)
        return node


def _pre_smoke_lint_and_autofix(code: str, inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
    canonical_name_by_lower: Dict[str, str] = {}
    for inp in inputs:
        name = str(inp.get("name", "")).strip()
        if name:
            canonical_name_by_lower[name.lower()] = name

    try:
        tree = ast.parse(code)
    except Exception as exc:
        return {
            "code": code,
            "autofix_applied": False,
            "fixes": [],
            "unknown_input_keys": [],
            "parse_error": str(exc),
        }

    fixer = _InputAutoFixer(canonical_name_by_lower)
    fixed_tree = fixer.visit(tree)
    ast.fix_missing_locations(fixed_tree)
    fixed_code = ast.unparse(fixed_tree)
    return {
        "code": fixed_code,
        "autofix_applied": bool(fixer.fixes),
        "fixes": fixer.fixes,
        "unknown_input_keys": sorted(set(fixer.unknown_input_keys)),
        "parse_error": None,
    }


def _infer_unpack_arity_by_input(code: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    try:
        tree = ast.parse(code)
    except Exception:
        return out
    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        if not isinstance(node.target, ast.Tuple):
            continue
        if not isinstance(node.iter, ast.Name):
            continue
        iter_name = node.iter.id
        # Look for assignment `iter_name = inputs['field']`
        for stmt in ast.walk(tree):
            if not isinstance(stmt, ast.Assign):
                continue
            if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                continue
            if stmt.targets[0].id != iter_name:
                continue
            if not isinstance(stmt.value, ast.Subscript):
                continue
            if not isinstance(stmt.value.value, ast.Name) or stmt.value.value.id != "inputs":
                continue
            slice_node = stmt.value.slice
            if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
                out[slice_node.value] = max(1, len(node.target.elts))
    return out



def _auto_inject_structure_checks(
    inputs: List[Dict[str, Any]],
    code: str,
    existing_checks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Automatically append F-class structural checks for array inputs whose
    execution code iterates elements as tuples (multi-element unpack).

    Uses the already-existing ``_infer_unpack_arity_by_input`` to detect which
    input fields are consumed as ``for a, b, ... in field``.  For each such
    field an F-class ``validity`` check is injected *unless* the existing checks
    already contain a structural expression for that field.

    This is schema-general — it works for any formula that iterates an array as
    a sequence of fixed-width tuples, not just LIFO inventory.
    """
    arity_map = _infer_unpack_arity_by_input(code)
    if not arity_map:
        return existing_checks

    # Build a set of input fields already covered by some structural check.
    existing_rules = [str(chk.get("expression", "")) for chk in existing_checks]
    covered = set()
    for field, arity in arity_map.items():
        for expr in existing_rules:
            if field in expr and ("isinstance" in expr or "len(" in expr):
                covered.add(field)
                break

    name_map = {str(inp.get("name", "")): inp for inp in inputs}
    injected = list(existing_checks)

    for field, arity in sorted(arity_map.items()):
        if arity < 2 or field in covered:
            continue
        inp = name_map.get(field, {})
        description = str(inp.get("description", "") or "").strip()
        rule_id = f"auto_structure_{field}"
        arity_s = str(arity)
        element_desc = " and ".join(f"element {i+1}" for i in range(arity))
        injected.append({
            "rule_id": rule_id,
            "name": f"{field} element structure",
            "check_type": "deterministic",
            "diagnostic_type": "F",
            "severity": "error",
            "predicate_mode": "validity",
            "expression": (
                f"isinstance(inputs['{field}'], list) and "
                f"all(isinstance(item, (list, tuple)) and len(item) == {arity_s} "
                f"for item in inputs['{field}'])"
            ),
            "applies_to": [field],
            "description": (
                f"Each element of '{field}' must be a list or tuple with exactly "
                f"{arity_s} numeric values ({element_desc})."
                + (f" {description}" if description else "")
            ),
        })

    return injected


def _reverse_op(op: str) -> str:
    return {">": "<=", ">=": "<", "<": ">=", "<=": ">"}.get(op, op)


def _extract_field_constraints(
    *,
    checks: List[Dict[str, Any]],
    field_name: str,
) -> List[tuple[str, float]]:
    constraints: List[tuple[str, float]] = []
    # inputs['x'] <op> 123
    pattern_a = re.compile(r"inputs\[['\"]([^'\"]+)['\"]\]\s*(<=|>=|<|>)\s*(-?\d+(?:\.\d+)?)")
    # 123 <op> inputs['x']
    pattern_b = re.compile(r"(-?\d+(?:\.\d+)?)\s*(<=|>=|<|>)\s*inputs\[['\"]([^'\"]+)['\"]\]")
    reverse_cmp = {"<": ">", "<=": ">=", ">": "<", ">=": "<="}

    for chk in checks:
        ctype = str(chk.get("check_type", "")).strip().lower()
        if ctype not in {"deterministic", "normalization"}:
            continue
        expr = str(chk.get("expression", "") or "")
        if not expr:
            continue
        # Only harvest constraints from simple guard expressions.  Reversing
        # each comparator inside a compound predicate such as
        # ``x > 0 and compute(inputs) < threshold`` is not logically valid and
        # can create dummy inputs that intentionally violate formula preconditions.
        if any(token in expr for token in (" and ", " or ", "compute(")):
            continue
        mode = str(chk.get("predicate_mode", "")).strip().lower()
        if mode not in {"violation", "validity"}:
            mode = "violation"

        found: List[tuple[str, float]] = []
        for m in pattern_a.finditer(expr):
            name, op, num = str(m.group(1)), str(m.group(2)), float(m.group(3))
            if name == field_name:
                found.append((op, num))
        for m in pattern_b.finditer(expr):
            num, op, name = float(m.group(1)), str(m.group(2)), str(m.group(3))
            if name == field_name:
                found.append((reverse_cmp.get(op, op), num))
        if mode == "validity":
            constraints.extend(found)
        else:
            constraints.extend((_reverse_op(op), num) for op, num in found)
    return constraints


def _apply_constraints_to_dummy(
    *,
    value: Any,
    input_type: str,
    constraints: List[tuple[str, float]],
) -> Any:
    if isinstance(value, bool):
        return value
    if not isinstance(value, (int, float)):
        return value
    is_int = str(input_type or "").strip().lower() == "integer"
    eps = 1.0 if is_int else 0.1
    lb: Optional[float] = None
    ub: Optional[float] = None
    for op, num in constraints:
        if op == ">":
            lb = max(lb if lb is not None else -1e18, num + eps)
        elif op == ">=":
            lb = max(lb if lb is not None else -1e18, num)
        elif op == "<":
            ub = min(ub if ub is not None else 1e18, num - eps)
        elif op == "<=":
            ub = min(ub if ub is not None else 1e18, num)
    x = float(value)
    if lb is not None and x < lb:
        x = lb
    if ub is not None and x > ub:
        x = ub
    if lb is not None and ub is not None and lb > ub:
        # Conflicting bounds: choose middle to keep smoke progressing.
        x = (lb + ub) / 2.0
    return int(round(x)) if is_int else x


def _build_smart_dummy_inputs(
    *,
    code: str,
    inputs: List[Dict[str, Any]],
    checks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    dummy_inputs = {
        str(inp.get("name", "")): _dummy_input_value(str(inp.get("type", "number")), str(inp.get("name", "")))
        for inp in inputs
        if str(inp.get("name", "")).strip()
    }
    for name in list(dummy_inputs):
        if name.startswith("initial_"):
            final_name = "final_" + name[len("initial_"):]
            if final_name in dummy_inputs and dummy_inputs[final_name] == dummy_inputs[name]:
                dummy_inputs[final_name] = 2.0
    if "weights" in dummy_inputs and isinstance(dummy_inputs["weights"], list):
        n = max(1, len(dummy_inputs["weights"]))
        dummy_inputs["weights"] = [1.0 / n for _ in range(n)]
    if "prices" in dummy_inputs and "weights" in dummy_inputs:
        if isinstance(dummy_inputs["prices"], list) and isinstance(dummy_inputs["weights"], list):
            n = min(len(dummy_inputs["prices"]), len(dummy_inputs["weights"])) or 1
            dummy_inputs["prices"] = dummy_inputs["prices"][:n]
            dummy_inputs["weights"] = [1.0 / n for _ in range(n)]
    if "factor_returns" in dummy_inputs and "betas" in dummy_inputs:
        if isinstance(dummy_inputs["factor_returns"], list) and isinstance(dummy_inputs["betas"], list):
            dummy_inputs["betas"] = [0.1 for _ in range(len(dummy_inputs["factor_returns"]) + 1)]
    if "number" in dummy_inputs and isinstance(dummy_inputs["number"], str):
        dummy_inputs["number"] = "4532015112830365"
    if "years" in dummy_inputs:
        for name, value in list(dummy_inputs.items()):
            if name != "years" and isinstance(value, list):
                dummy_inputs["years"] = len(value)
                break
    if "target_year" in dummy_inputs and "years" in dummy_inputs:
        dummy_inputs["target_year"] = 1
    if "initial_period" in dummy_inputs and "years" in dummy_inputs:
        try:
            dummy_inputs["initial_period"] = max(0, int(dummy_inputs["years"]) - 1)
        except Exception:
            pass
    if "shares_outstanding" in dummy_inputs and "shares_reduced" in dummy_inputs:
        dummy_inputs["shares_outstanding"] = 2
        dummy_inputs["shares_reduced"] = 1
    if "par_value" in dummy_inputs and "market_price" in dummy_inputs:
        dummy_inputs["par_value"] = 2.0
        dummy_inputs["market_price"] = 1.0
    if "annual_operating_expenses" in dummy_inputs and "noncash_charges" in dummy_inputs:
        dummy_inputs["annual_operating_expenses"] = 2.0
        dummy_inputs["noncash_charges"] = 1.0
    if "target_department_name" in dummy_inputs and "department_square_footages" in dummy_inputs:
        dummy_inputs["target_department_name"] = "Sales"
    if "rating_proportions" in dummy_inputs and "credit_rating_map" in dummy_inputs:
        dummy_inputs["rating_proportions"] = {"AAA": 0.5, "BBB": 0.5}
        dummy_inputs["credit_rating_map"] = {"AAA": 1.0, "BBB": 4.0}
    if "weights" in dummy_inputs and "variances" in dummy_inputs and "correlations" in dummy_inputs:
        dummy_inputs["weights"] = [0.5, 0.5]
        dummy_inputs["variances"] = [0.04, 0.09]
        dummy_inputs["correlations"] = [[1.0, 0.2], [0.2, 1.0]]
    if {"stock_price", "strike_price"} <= set(dummy_inputs):
        dummy_inputs["stock_price"] = 100.0
        dummy_inputs["strike_price"] = 100.0
    if "time_to_expiration" in dummy_inputs:
        dummy_inputs["time_to_expiration"] = 1.0
    if "risk_free_rate" in dummy_inputs:
        dummy_inputs["risk_free_rate"] = 0.05
    if "option_market_price" in dummy_inputs:
        dummy_inputs["option_market_price"] = 10.0
    # Small cross-field heuristic for common depreciation constraints.
    if "cost" in dummy_inputs and "salvage_value" in dummy_inputs:
        dummy_inputs["cost"] = 100.0
        dummy_inputs["salvage_value"] = 10.0

    unpack_arity = _infer_unpack_arity_by_input(code)
    for inp in inputs:
        name = str(inp.get("name", "")).strip()
        if not name:
            continue
        input_type = str(inp.get("type", "number"))
        constraints = _extract_field_constraints(checks=checks, field_name=name)
        dummy_inputs[name] = _apply_constraints_to_dummy(
            value=dummy_inputs.get(name),
            input_type=input_type,
            constraints=constraints,
        )
        if input_type == "array[number]" and name in unpack_arity and unpack_arity[name] > 1:
            arity = unpack_arity[name]
            row = [1.0 for _ in range(arity)]
            dummy_inputs[name] = [list(row), list(row)]
    return dummy_inputs


def _execution_smoke_test(
    code: str,
    inputs: List[Dict[str, Any]],
    output_name: str,
    checks: List[Dict[str, Any]],
) -> Dict[str, Any]:
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
    lint = _pre_smoke_lint_and_autofix(code, inputs)
    code_to_run = str(lint.get("code", "") or code)
    dummy_inputs = _build_smart_dummy_inputs(code=code_to_run, inputs=inputs, checks=checks)
    env: Dict[str, Any] = {}
    try:
        env["__builtins__"] = safe_builtins
        exec(code_to_run, env, env)
        fn = env.get("compute")
        if not callable(fn):
            return {
                "ok": False,
                "error": "compute function missing after exec",
                "autofix_applied": bool(lint.get("autofix_applied")),
                "lint_fixes": lint.get("fixes", []),
                "effective_code": code_to_run,
            }
        result = fn(dummy_inputs)
        if isinstance(result, dict) and output_name and output_name not in result:
            return {
                "ok": False,
                "error": f"output '{output_name}' missing in compute result dict",
                "autofix_applied": bool(lint.get("autofix_applied")),
                "lint_fixes": lint.get("fixes", []),
                "effective_code": code_to_run,
            }
        return {
            "ok": True,
            "error": None,
            "autofix_applied": bool(lint.get("autofix_applied")),
            "lint_fixes": lint.get("fixes", []),
            "effective_code": code_to_run,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "autofix_applied": bool(lint.get("autofix_applied")),
            "lint_fixes": lint.get("fixes", []),
            "effective_code": code_to_run,
        }


def _collect_diagnostic_checks(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    diagnostics = raw.get("diagnostics")
    checks_raw: List[Dict[str, Any]] = []
    if isinstance(diagnostics, dict):
        for key, default_type in (("invariants", "deterministic"), ("scale_checks", "normalization")):
            rows = diagnostics.get(key, [])
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                fixed = dict(row)
                fixed["check_type"] = str(fixed.get("check_type", "")).strip().lower() or default_type
                checks_raw.append(fixed)

    # Backward compatibility with v1 generation output.
    if not checks_raw and isinstance(raw.get("diagnostic_checks"), list):
        checks_raw = [x for x in raw.get("diagnostic_checks", []) if isinstance(x, dict)]
    return [_normalize_check(chk) for chk in checks_raw]


def formalize_core_payload(
    raw: Dict[str, Any],
    *,
    source_meta: Dict[str, Any],
    article_title: str,
    article_doc_id: Optional[int],
    article_content_excerpt: str,
    fallback_id: str,
    allow_new_topic: bool,
) -> Dict[str, Any]:
    # Keep fic_id style deterministic across cards:
    # prefer source function_id over model-proposed id.
    fic_id = safe_id(source_meta.get("function_id") or raw.get("fic_id") or raw.get("id") or fallback_id)
    name = str(article_title or "").strip() or str(raw.get("name", "")).strip()
    short_description = str(raw.get("short_description", "")).strip()
    domain, topic, is_new_topic = validate_domain_topic(
        str(raw.get("domain", "")),
        str(raw.get("topic", "")),
        allow_new_topic=allow_new_topic,
    )

    inputs_raw = raw.get("inputs")
    if not isinstance(inputs_raw, list) or not inputs_raw:
        raise ValueError("Stage core requires non-empty inputs.")
    inputs = [_normalize_input_item(x) for x in inputs_raw]
    if any(not inp["name"] for inp in inputs):
        raise ValueError("Every input must include a normalized name.")

    output = _normalize_output(raw.get("output") or {"name": raw.get("output_var")})
    if not output["name"]:
        raise ValueError("output.name is required")
    if output["type"] not in {"number", "integer", "boolean", "string"}:
        raise ValueError(
            f"output.type must be scalar for the current evaluator, got {output['type']!r}"
        )

    execution = dict(raw.get("execution") or {})
    execution["language"] = "python"
    execution["entrypoint"] = "compute"
    execution["deterministic"] = True
    execution["code"] = require_compute_code(execution.get("code", ""))

    checks = _collect_diagnostic_checks(raw)
    if not checks:
        raise ValueError("diagnostics checks must be a non-empty list")
    # Semi-automatically inject F-class structural checks for array inputs
    # whose execution code iterates elements as tuples.  This catches cases where
    # the LLM-authored checks omit the element-structure guard, which would
    # otherwise surface as a TypeError at runtime (misclassified as C-class).
    checks = _auto_inject_structure_checks(inputs, execution["code"], checks)

    selection_hints = _norm_str_list(raw.get("selection_hints", []))
    if not selection_hints:
        selection_hints = [name, short_description, topic]

    semantic_hints_raw = raw.get("semantic_hints", [])
    if not isinstance(semantic_hints_raw, list):
        semantic_hints_raw = []
    semantic_hints = [
        _normalize_semantic_hint(h, idx)
        for idx, h in enumerate(semantic_hints_raw, start=1)
        if isinstance(h, dict)
    ][:SEMANTIC_HINT_MAX]

    if not name:
        raise ValueError("name is required")
    if not short_description:
        raise ValueError("short_description is required")

    execution_smoke = _execution_smoke_test(execution["code"], inputs, output["name"], checks)
    effective_code = str(execution_smoke.get("effective_code", "") or execution["code"])
    execution["code"] = require_compute_code(effective_code)
    if not execution_smoke.get("ok"):
        raise ValueError(f"execution smoke failed: {execution_smoke.get('error', 'unknown error')}")
    source_meta_out = dict(source_meta)
    source_meta_out["execution_smoke_ok"] = bool(execution_smoke.get("ok"))
    source_meta_out["execution_autofix_applied"] = bool(execution_smoke.get("autofix_applied"))
    lint_fixes = execution_smoke.get("lint_fixes", [])
    if isinstance(lint_fixes, list) and lint_fixes:
        source_meta_out["execution_lint_fixes"] = lint_fixes
    if not execution_smoke.get("ok"):
        source_meta_out["execution_smoke_error"] = str(execution_smoke.get("error", "unknown error"))

    return {
        "fic_id": fic_id,
        "name": name,
        "short_description": short_description,
        "domain": domain,
        "topic": topic,
        "topic_extension": is_new_topic,
        "version": "v2",
        "source_meta": source_meta_out,
        "selection_hints": selection_hints,
        "semantic_hints": semantic_hints,
        "inputs": inputs,
        "output": output,
        "execution": execution,
        "diagnostic_checks": checks,
        "execution_smoke_test": execution_smoke,
        "article_title": article_title,
        "article_doc_id": article_doc_id,
        "article_content_excerpt": article_content_excerpt,
    }


def generate_core(
    *,
    client: Any,
    model: str,
    defn: ConversionInput,
    allow_new_topic: bool,
) -> Dict[str, Any]:
    if not defn.python_solution.strip():
        raise ValueError("Record missing python_solution")
    schema = stage_core_schema()
    if schema is None:
        raise RuntimeError("Stage core schema is unavailable because google.genai is missing.")

    fallback_id = f"{defn.source_meta.get('function_id') or 'unknown'}_{defn.source_meta.get('question_id') or 'unknown'}"
    base_prompt = build_stage_core_prompt(defn)
    corrective_suffix = ""
    last_error: Optional[Exception] = None

    # Some model responses still violate compute(inputs) despite schema constraints.
    # Retry with progressively stricter corrective hints before failing hard.
    for attempt in range(1, 4):
        prompt = f"{base_prompt}\n{corrective_suffix}" if corrective_suffix else base_prompt
        raw = call_gemini_json(
            client,
            model=model,
            prompt=prompt,
            schema=schema,
        )
        try:
            return formalize_core_payload(
                raw,
                source_meta=defn.source_meta,
                article_title=defn.article_title,
                article_doc_id=defn.article_doc_id,
                article_content_excerpt=defn.article_content_excerpt,
                fallback_id=fallback_id,
                allow_new_topic=allow_new_topic,
            )
        except ValueError as err:
            last_error = err
            corrective_suffix = (
                "CRITICAL CORRECTION:\n"
                f"- Previous output failed validation: {err}\n"
                "- Ensure domain is from taxonomy.\n"
                "- Ensure topic is non-empty snake_case.\n"
                "- Ensure execution.code contains exactly def compute(inputs):\n"
                "- Ensure execution.code compiles and passes the execution smoke test.\n"
                "- Ensure diagnostics checks have valid expression and predicate_mode.\n"
                "- Use richer input types when needed: dict[string,number], array[object], array[array[number]].\n"
                "- Keep output scalar; if source logic returns a dict, return the requested scalar field directly.\n"
                f"- This is retry #{attempt} with stricter validation requirements."
            )

    raise ValueError(
        f"Stage core failed after retries for record {fallback_id}: {last_error}"
    )


def default_topic_for_domain(domain: str) -> Optional[str]:
    topics = TAXONOMY.get(domain, [])
    if not topics:
        return None
    return topics[0]
