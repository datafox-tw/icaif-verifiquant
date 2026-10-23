"""PoC: code_patch + result_postprocess cross-verification.

Scenario: NPV with itemized cash flows (period-by-period list).
  - FIC code uses enumerate(cash_flows, start=1)  → end-of-period (ordinary annuity).
  - User says: "these are beginning-of-period payments."
  - We cannot ask the user to shift their data.
  - Instead: patch `start=1` → `start=0` in the FIC code.
  - Verify: the patched code's output == result_postprocess result (cross-check).

Run: python3 tests/test_transform_poc.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verifiquant.preprocessing.stage_repair import _global_i_rules
from verifiquant.preprocessing.stage_transform import (
    CodePatchSpec,
    TransformSpec,
    apply_code_patch,
    apply_result_postprocess,
    apply_transform,
    ast_diff_count,
    count_ast_nodes,
    verify_code_patch,
    verify_transform,
    get_transform_spec_for_choice,
)


def separator(title: str) -> None:
    print(f"\n{'─' * 65}")
    print(f"  {title}")
    print('─' * 65)


# ═══════════════════════════════════════════════════════════════════
# FIC card: NPV with itemized cash flows (end-of-period baseline)
# ═══════════════════════════════════════════════════════════════════
#
# The critical line: enumerate(cash_flows, start=1)
# t=1 means "year 1 end", t=2 means "year 2 end", etc.
#
NPV_ITEMIZED_CODE = """\
def compute_npv_itemized(cash_flows, discount_rate, initial_investment):
    npv = 0.0
    for t, cf in enumerate(cash_flows, start=1):
        npv += cf / (1 + discount_rate) ** t
    return npv - initial_investment
"""

MOCK_FIC = {
    "fic_id": "fic_npv_itemized",
    "name": "NPV — Itemized Cash Flows",
    "inputs": [
        {"name": "cash_flows",         "type": "list",   "required": True,  "description": "Cash flow list"},
        {"name": "discount_rate",      "type": "number", "required": True,  "description": "Discount rate per period"},
        {"name": "initial_investment", "type": "number", "required": True,  "description": "Initial outlay"},
    ],
    "execution": {
        "language": "python",
        "entrypoint": "compute_npv_itemized",
        "deterministic": True,
        "code": NPV_ITEMIZED_CODE,
    },
    "semantic_hints": [
        {
            "id": "payment_timing",
            "ambiguity_type": "time_basis",
            "trigger_signal": "payment period not explicitly stated",
            "clarification_question": "Are cash flows paid at END of each period (ordinary annuity) or BEGINNING (annuity-due)?",
            "i_level": "soft",
            "assumption_if_not_clarified": "End of period",
            "impact_scope": "output_interpretation",
            "options": [
                {
                    "label": "End of period — DEFAULT (code uses start=1)",
                    "value": "end_of_period",
                    "is_default": True,
                },
                {
                    "label": "Beginning of period (annuity-due)",
                    "value": "beginning_of_period",
                    "is_default": False,
                    # --- Two complementary specs for the same semantic change ---
                    # Option A: code_patch — surgically change the loop offset.
                    # Option B: result_postprocess — adjust the final result.
                    # Cross-verification: both must produce the same number.
                    "transform_spec": {
                        "patch_type": "code_patch",
                        "target_pattern": "start=1",
                        "replacement":    "start=0",
                        "max_changed_nodes": 2,
                        # Cross-verify: patched output must match this expression
                        # applied to the ORIGINAL output.
                        # Derivation:
                        #   original:  Σ CF_t / (1+r)^t  for t=1..n
                        #   patched:   Σ CF_t / (1+r)^t  for t=0..n-1
                        #            = (original + initial_investment) * (1+r) - initial_investment
                        "cross_verify_result_expr": (
                            "(result + initial_investment) * (1 + discount_rate) "
                            "- initial_investment"
                        ),
                        "cross_verify_max_nodes": 25,
                        "invariant": (
                            "result_new + initial_investment == "
                            "(result_old + initial_investment) * (1 + discount_rate)"
                        ),
                        "affected_inputs": ["discount_rate", "initial_investment"],
                    },
                },
            ],
        },
    ],
    "diagnostic_checks": [],
}

SAMPLE_INPUTS = [
    {
        "cash_flows": [100_000, 200_000, 300_000],
        "discount_rate": 0.10,
        "initial_investment": 400_000,
    },
    {
        "cash_flows": [50_000, 50_000, 50_000, 50_000],
        "discount_rate": 0.08,
        "initial_investment": 150_000,
    },
    {
        "cash_flows": [80_000, 120_000],
        "discount_rate": 0.05,
        "initial_investment": 180_000,
    },
]

# ── reference function for manual verification ──────────────────────
def _npv_end(cash_flows, discount_rate, initial_investment):
    return sum(cf / (1 + discount_rate) ** t
               for t, cf in enumerate(cash_flows, start=1)) - initial_investment

def _npv_beg(cash_flows, discount_rate, initial_investment):
    return sum(cf / (1 + discount_rate) ** t
               for t, cf in enumerate(cash_flows, start=0)) - initial_investment


# ═══════════════════════════════════════════════════════════════════
# 1. Derive repair rules
# ═══════════════════════════════════════════════════════════════════
separator("Step 1 — _global_i_rules(): transform_map from semantic_hints")
repair_rules = _global_i_rules(MOCK_FIC)
for rule in repair_rules:
    tm = rule["repair_action"].get("transform_map", {})
    print(f"rule_id: {rule['rule_id']}  |  severity: {rule['severity']}")
    for val, spec in tm.items():
        print(f"  option '{val}':")
        print(f"    patch_type     : {spec['patch_type']}")
        print(f"    target_pattern : {spec.get('target_pattern', '-')!r}")
        print(f"    replacement    : {spec.get('replacement', '-')!r}")
        print(f"    cross_verify   : {spec.get('cross_verify_result_expr', '-')!r}")

# ═══════════════════════════════════════════════════════════════════
# 2. Apply the code patch and show the diff
# ═══════════════════════════════════════════════════════════════════
separator("Step 2 — apply_code_patch: show the surgical diff")
rule = repair_rules[0]
raw_spec = rule["repair_action"]["transform_map"]["beginning_of_period"]
patch_spec = CodePatchSpec.from_dict(raw_spec)

original_code = MOCK_FIC["execution"]["code"]
patched_code  = apply_code_patch(original_code, patch_spec)

print("Original:")
for line in original_code.strip().splitlines():
    print(f"  {line}")
print("\nPatched:")
for orig_line, new_line in zip(original_code.strip().splitlines(),
                                patched_code.strip().splitlines()):
    marker = "→" if orig_line != new_line else " "
    print(f" {marker} {new_line}")

changed = ast_diff_count(original_code, patched_code)
print(f"\nAST nodes changed: {changed}  (declared max: {patch_spec.max_changed_nodes})")

# ═══════════════════════════════════════════════════════════════════
# 3. Numerical: what does the patch actually produce?
# ═══════════════════════════════════════════════════════════════════
separator("Step 3 — numerical results: original vs patched vs expected")
print(f"{'inputs':40s}  {'end-of-period':>14}  {'patched':>14}  {'expected_beg':>14}  match")
print("─" * 90)
for inp in SAMPLE_INPUTS:
    orig_result    = _npv_end(**inp)
    patched_result = _npv_beg(**inp)     # = what patched code produces
    expected_beg   = _npv_beg(**inp)
    match = "✓" if abs(patched_result - expected_beg) < 1e-6 else "✗"
    label = f"CF={inp['cash_flows']}, r={inp['discount_rate']}"
    print(f"  {label:38s}  {orig_result:>14.2f}  {patched_result:>14.2f}  {expected_beg:>14.2f}  {match}")

# ═══════════════════════════════════════════════════════════════════
# 4. Cross-verification: patched code == result_postprocess?
# ═══════════════════════════════════════════════════════════════════
separator("Step 4 — cross-verify: code_patch result == result_postprocess result?")
print(f"  cross_verify_expr: {patch_spec.cross_verify_result_expr!r}\n")

import math as _math
for inp in SAMPLE_INPUTS:
    # Path A: patched code
    pa = _npv_beg(**inp)
    # Path B: original + result_postprocess expr
    orig = _npv_end(**inp)
    inv = inp["initial_investment"]
    r   = inp["discount_rate"]
    pb = (orig + inv) * (1 + r) - inv
    delta = abs(pa - pb)
    ok = "✓" if delta < 1e-6 else "✗"
    print(f"  [{ok}] CF={inp['cash_flows']}")
    print(f"       path A (code_patch)         : {pa:.6f}")
    print(f"       path B (result_postprocess) : {pb:.6f}")
    print(f"       delta                       : {delta:.2e}")

# ═══════════════════════════════════════════════════════════════════
# 5. Full verify_code_patch (automated)
# ═══════════════════════════════════════════════════════════════════
separator("Step 5 — verify_code_patch: three-layer automated check")
result = verify_code_patch(patch_spec, MOCK_FIC, SAMPLE_INPUTS)
print(result.summary())
print(f"  occurrence_count        : {result.occurrence_count}")
print(f"  ast_changed_nodes       : {result.ast_changed_nodes} / {result.max_changed_nodes}")
print(f"  cross_verify_ok         : {result.cross_verify_ok}")
print(f"  cross_verify_max_delta  : {result.cross_verify_max_delta:.2e}" if result.cross_verify_max_delta is not None else "  cross_verify_max_delta  : n/a")
print(f"  numerical_samples       : {result.numerical_samples}")
print(f"  invariant_check         : {result.invariant_check}")
print(f"  error                   : {result.error}")

# ═══════════════════════════════════════════════════════════════════
# 6. Rejection tests
# ═══════════════════════════════════════════════════════════════════
separator("Step 6 — rejection tests")

# 6a: pattern not found
bad_notfound = CodePatchSpec(
    patch_type="code_patch",
    target_pattern="start=99",   # does not exist
    replacement="start=0",
    max_changed_nodes=2,
    cross_verify_result_expr="result",
)
r1 = verify_code_patch(bad_notfound, MOCK_FIC, SAMPLE_INPUTS[:1])
status = "✓ BLOCKED" if not r1.passed else "✗ NOT BLOCKED"
print(f"\n  [{status}] pattern not found → {r1.error}")

# 6b: pattern found but too many AST nodes change (someone sneaks in extra logic)
bad_blast = CodePatchSpec(
    patch_type="code_patch",
    target_pattern="start=1",
    replacement="start=0",
    max_changed_nodes=0,   # claim zero changes allowed → impossible, will fail
    cross_verify_result_expr="(result + initial_investment) * (1 + discount_rate) - initial_investment",
    cross_verify_max_nodes=25,
)
r2 = verify_code_patch(bad_blast, MOCK_FIC, SAMPLE_INPUTS[:1])
status = "✓ BLOCKED" if not r2.passed else "✗ NOT BLOCKED"
print(f"\n  [{status}] blast radius exceeded → {r2.error}")

# 6c: math is wrong — cross_verify_expr says wrong thing
bad_math = CodePatchSpec(
    patch_type="code_patch",
    target_pattern="start=1",
    replacement="start=0",
    max_changed_nodes=2,
    cross_verify_result_expr="result * 2",   # completely wrong
    cross_verify_max_nodes=10,
)
r3 = verify_code_patch(bad_math, MOCK_FIC, SAMPLE_INPUTS)
status = "✓ BLOCKED" if not r3.passed else "✗ NOT BLOCKED"
print(f"\n  [{status}] wrong cross_verify_expr → cross_verify_ok={r3.cross_verify_ok}, "
      f"max_delta={r3.cross_verify_max_delta:.2f}" if r3.cross_verify_max_delta is not None
      else f"\n  [{status}] wrong cross_verify_expr → {r3.error}")

# 6d: confirm result_postprocess from earlier still passes
separator("Step 7 — confirm result_postprocess still works (regression)")
rp_spec = TransformSpec(
    patch_type="result_postprocess",
    result_expr="(result + initial_investment) * (1 + discount_rate) - initial_investment",
    max_expr_nodes=25,
    invariant="result_new + initial_investment == (result_old + initial_investment) * (1 + discount_rate)",
    affected_inputs=["discount_rate", "initial_investment"],
)
rp_result = verify_transform(rp_spec, MOCK_FIC, SAMPLE_INPUTS)
print(rp_result.summary())

print("\n✓ PoC complete\n")
