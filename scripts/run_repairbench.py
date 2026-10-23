#!/usr/bin/env python3
"""RepairBench: systematic validation of C4 (verifiable atomic transforms).

Answers the reviewer question "how many bad patches does cross-verification actually
catch?" by mutation testing over every result_postprocess transform_spec in the
250-card library (191 specs / 159 families). Zero LLM calls — fully deterministic.

Attack model: the declared contract (invariant, max_expr_nodes) is trusted; an
adversarial/wrong *patch expression* is proposed against it. For each valid spec we
generate mutated expressions and check that verify_transform rejects them:

  op_swap        first *<->/ or +<->- swapped        (wrong operator)
  coeff_perturb  first numeric constant doubled       (wrong coefficient)
  var_swap       one input name -> another input      (wrong variable)
  identity       expr = "result"                      (no-op repair)
  call_inject    min(expr, 1e9)                       (function call, whitelist breach)
  oversize       expr padded past max_expr_nodes      (AST-bound breach)

Reported: valid-accept rate, per-class rejection rate, FALSE-ACCEPT rate (mutant
passes), rejection-reason breakdown, and point-of-use (1 sample) vs multi-sample (5)
comparison. The "two coincident errors" blind spot (mutated expr + invariant rewritten
to match) passes by construction and is reported as a demonstration, not a rate.

Usage: PYTHONPATH=. python3 scripts/run_repairbench.py \
         [--core results/paper_v2_250/fic/core.jsonl] [--seed 0]
"""
from __future__ import annotations

import argparse, ast, copy, json, math, os, random
from collections import Counter, defaultdict
from datetime import datetime

from verifiquant.preprocessing.stage_transform import (
    TransformSpec, verify_transform, count_ast_nodes, _run_fic_code,
    apply_result_postprocess,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE = os.path.join(ROOT, "results/paper_v2_250/fic/core.jsonl")

# ---------------------------------------------------------------- input synthesis
MONEY = ("price", "value", "amount", "principal", "investment", "balance", "sales",
         "income", "cost", "revenue", "cash", "debt", "asset", "expense", "payment",
         "premium", "deposit", "loan", "capital", "dividend", "earnings", "proceeds")
COUNTY = ("year", "period", "term", "month", "day", "count", "quantity", "n_", "num",
          "shares", "units", "times", "installment")


def _num_for(name: str, rng: random.Random) -> float:
    n = name.lower()
    if "probability" in n or "prob" in n:
        return round(rng.uniform(0.1, 0.9), 3)
    if "rate" in n or "ratio" in n or "yield" in n or "pct" in n or "percent" in n \
       or "margin" in n or "growth" in n or "return" in n:
        return round(rng.uniform(0.02, 0.30), 4)
    if any(k in n for k in COUNTY):
        return rng.randint(2, 20)
    if any(k in n for k in MONEY):
        return round(rng.uniform(1e3, 1e6), 2)
    return round(rng.uniform(1.5, 100.0), 3)


def synth_inputs(inputs: list, rng: random.Random):
    """One draw of plausible inputs; None if a type we do not synthesize appears."""
    out = {}
    for spec in inputs:
        name, typ = spec.get("name"), str(spec.get("type", "number"))
        if typ == "number":
            out[name] = float(_num_for(name, rng))
        elif typ == "integer":
            v = _num_for(name, rng)
            out[name] = max(2, int(v))
        elif typ == "boolean":
            out[name] = True
        elif typ == "array[number]":
            out[name] = [round(rng.uniform(1e2, 1e4), 2) for _ in range(rng.randint(3, 5))]
        else:  # string / dict / array[object] / nested arrays: skip this card
            return None
    return out


def sample_ok(core, sample) -> bool:
    ex = core.get("execution", {})
    try:
        v = _run_fic_code(ex.get("code", ""), ex.get("entrypoint", ""), sample)
        return isinstance(v, (int, float)) and math.isfinite(float(v)) and abs(float(v)) < 1e15
    except Exception:
        return False


def draw_samples(core, k: int, rng: random.Random, tries: int = 60):
    got = []
    for _ in range(tries):
        s = synth_inputs(core.get("inputs", []), rng)
        if s is None:
            return None  # unsynthesizable types
        if sample_ok(core, s):
            got.append(s)
            if len(got) == k:
                return got
    return got if got else []


# ---------------------------------------------------------------- expr mutations
class _OpSwap(ast.NodeTransformer):
    def __init__(self): self.done = False
    def visit_BinOp(self, node):
        self.generic_visit(node)
        if not self.done:
            m = {ast.Mult: ast.Div, ast.Div: ast.Mult, ast.Add: ast.Sub, ast.Sub: ast.Add}
            for k, v in m.items():
                if isinstance(node.op, k):
                    node.op = v(); self.done = True; break
        return node


class _CoeffPerturb(ast.NodeTransformer):
    def __init__(self): self.done = False
    def visit_Constant(self, node):
        if not self.done and isinstance(node.value, (int, float)):
            node = ast.copy_location(
                ast.Constant(value=(node.value * 2) if node.value else 1.0), node)
            self.done = True
        return node


def mutate(expr: str, input_names: list, max_nodes: int, rng: random.Random):
    """Yield (class, mutated_expr) pairs; skip classes that do not apply."""
    out = []
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return out

    t = copy.deepcopy(tree); tr = _OpSwap(); tr.visit(t)
    if tr.done:
        out.append(("op_swap", ast.unparse(t)))
    t = copy.deepcopy(tree); tr = _CoeffPerturb(); tr.visit(t)
    if tr.done:
        out.append(("coeff_perturb", ast.unparse(t)))

    used = sorted({n.id for n in ast.walk(tree)
                   if isinstance(n, ast.Name) and n.id in input_names})
    if used:
        pool = [n for n in input_names if n not in used] or \
               [n for n in input_names if n != used[0]]
        if pool:
            repl = rng.choice(sorted(pool))
            t = copy.deepcopy(tree)
            for n in ast.walk(t):
                if isinstance(n, ast.Name) and n.id == used[0]:
                    n.id = repl
            m = ast.unparse(t)
            if m != expr:
                out.append(("var_swap", m))

    if expr.strip() != "result":
        out.append(("identity", "result"))
    out.append(("call_inject", f"min({expr}, 10**9)"))

    pad = expr
    while count_ast_nodes(pad) <= max_nodes:
        pad += " + 0*result"
    out.append(("oversize", pad))
    return out


def classify_reject(vr) -> str:
    if vr.passed:
        return "PASSED"
    if vr.error and vr.error.startswith("AST guard"):
        return "ast_guard"
    if vr.error and vr.error.startswith("Numerical check"):
        return "numerical"
    if vr.invariant_check and vr.invariant_check.startswith("counterexample"):
        return "invariant"
    return "other"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--core", default=CORE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results/repairbench"))
    args = ap.parse_args()
    rng = random.Random(args.seed)

    specs = []
    for line in open(args.core):
        c = json.loads(line)
        for h in (c.get("semantic_hints") or []):
            for o in (h.get("options") or []):
                ts = o.get("transform_spec") if isinstance(o, dict) else None
                if ts and ts.get("patch_type", "result_postprocess") == "result_postprocess":
                    specs.append((c, h.get("id"), o.get("value"), ts))

    os.makedirs(args.out_dir, exist_ok=True)
    ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_rows = open(os.path.join(args.out_dir, f"repairbench_{ts_tag}.jsonl"), "w")

    n_specs = len(specs)
    skipped_synth = 0
    valid = Counter()          # k=1 / k=5 valid-accept
    mut_total = Counter()      # per class
    mut_reject = defaultdict(Counter)   # class -> reason counter (k=samples)
    false_accept = defaultdict(list)    # class -> [(fic,hint,expr,equivalent)]
    fa_k1 = Counter(); fa_k5 = Counter()  # false accepts per class by sample count
    fa_equiv = Counter()       # false accepts that are EQUIVALENT mutants (same output)
    families = set()

    def outputs_equal(spec_a, spec_b, core, samples) -> bool:
        """True if both exprs give (nearly) identical outputs on every sample."""
        ex = core.get("execution", {})
        try:
            for s in samples:
                old = _run_fic_code(ex.get("code", ""), ex.get("entrypoint", ""), s)
                a = float(apply_result_postprocess(old, spec_a, s))
                b = float(apply_result_postprocess(old, spec_b, s))
                if not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12):
                    return False
            return True
        except Exception:
            return False

    for card, hid, oval, raw in specs:
        core = {"inputs": card.get("inputs", []), "execution": card.get("execution", {})}
        samples = draw_samples(core, args.samples, rng)
        if samples is None or not samples:
            skipped_synth += 1
            continue
        families.add(card["fic_id"])
        spec = TransformSpec.from_dict(raw)
        s1, sk = samples[:1], samples

        vr1 = verify_transform(spec, core, s1)
        vrk = verify_transform(spec, core, sk)
        valid["k1"] += bool(vr1.passed); valid["k5"] += bool(vrk.passed)
        valid["n"] += 1
        out_rows.write(json.dumps({"kind": "valid", "fic": card["fic_id"], "hint": hid,
                                   "value": oval, "k1": vr1.passed, "k5": vrk.passed,
                                   "k5_reason": classify_reject(vrk)}) + "\n")

        input_names = [i["name"] for i in core["inputs"]]
        for cls, mexpr in mutate(spec.result_expr, input_names, spec.max_expr_nodes, rng):
            mspec = TransformSpec.from_dict({**raw, "result_expr": mexpr})
            mut_total[cls] += 1
            r1 = verify_transform(mspec, core, s1)
            rk = verify_transform(mspec, core, sk)
            mut_reject[cls][classify_reject(rk)] += 1
            equiv = None
            if r1.passed: fa_k1[cls] += 1
            if rk.passed:
                fa_k5[cls] += 1
                equiv = outputs_equal(mspec, spec, core, sk)
                if equiv: fa_equiv[cls] += 1
                false_accept[cls].append((card["fic_id"], hid, mexpr, equiv))
            out_rows.write(json.dumps({"kind": "mutant", "class": cls,
                                       "fic": card["fic_id"], "hint": hid,
                                       "expr": mexpr, "k1": r1.passed, "k5": rk.passed,
                                       "equivalent": equiv,
                                       "reason": classify_reject(rk)}) + "\n")
    out_rows.close()

    N = valid["n"]
    print(f"specs={n_specs} usable={N} (families={len(families)}) "
          f"skipped_unsynthesizable={skipped_synth}")
    print(f"valid-accept: k=1 {valid['k1']}/{N} ({100*valid['k1']/N:.1f}%)  "
          f"k=5 {valid['k5']}/{N} ({100*valid['k5']/N:.1f}%)")
    print(f"{'class':14}{'n':>6}{'reject@5':>10}{'FA@5':>7}{'equiv':>7}{'trueFA':>8}{'FA@1':>7}   reasons@5")
    summary_rows = []
    for cls in sorted(mut_total):
        n = mut_total[cls]
        rej = n - fa_k5[cls]
        true_fa = fa_k5[cls] - fa_equiv[cls]
        reasons = dict(mut_reject[cls])
        print(f"{cls:14}{n:>6}{100*rej/n:>9.1f}%{fa_k5[cls]:>7}{fa_equiv[cls]:>7}{true_fa:>8}{fa_k1[cls]:>7}   {reasons}")
        summary_rows.append({"class": cls, "n": n, "reject_rate_k5": rej / n,
                             "false_accept_k5": fa_k5[cls],
                             "equivalent_mutants": fa_equiv[cls],
                             "true_false_accept_k5": true_fa,
                             "false_accept_k1": fa_k1[cls],
                             "reasons_k5": reasons})
    tot = sum(mut_total.values()); fa5 = sum(fa_k5.values()); fa1 = sum(fa_k1.values())
    eq = sum(fa_equiv.values()); tfa = fa5 - eq
    print(f"{'TOTAL':14}{tot:>6}{100*(tot-fa5)/tot:>9.1f}%{fa5:>7}{eq:>7}{tfa:>8}{fa1:>7}")
    print(f"effective rejection (non-equivalent mutants): "
          f"{tot-eq-tfa}/{tot-eq} = {100*(tot-eq-tfa)/(tot-eq):.1f}%")
    if any(false_accept.values()):
        print("TRUE false accepts (k=5, non-equivalent):")
        for cls, rows in false_accept.items():
            for fic, hid, expr, equiv in rows:
                if not equiv:
                    print(f"  [{cls}] {fic}/{hid}: {expr}")

    summary = {"ran_at": ts_tag, "core": args.core, "seed": args.seed,
               "samples_k": args.samples, "specs_total": n_specs, "usable": N,
               "families": len(families), "skipped_unsynthesizable": skipped_synth,
               "valid_accept_k1": valid["k1"], "valid_accept_k5": valid["k5"],
               "mutants_total": tot, "false_accept_k5_total": fa5,
               "equivalent_mutants_total": eq, "true_false_accept_k5_total": tfa,
               "false_accept_k1_total": fa1, "per_class": summary_rows}
    with open(os.path.join(args.out_dir, f"repairbench_summary_{ts_tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("summary written to", f.name)


if __name__ == "__main__":
    main()
