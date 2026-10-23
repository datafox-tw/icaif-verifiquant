"""
score_messy_external.py
------------------------
RQ4 scorer: fills tab:v2ext (Human-FNAR / Match / Sel.Acc) for CoT, R-PAL, R-PAL+, VQ
on the 80-item human-labeled messy-query set. Offline; reads the four run files +
ground_truth.jsonl + messy_pilot_provenance.json. See scripts/score_messy_external_SPEC.md.

Decisions locked (per spec):
  D1: I_SOFT counts as emit=True (a warned number is still committed).
  D2: 4-way action where the system exposes structured decision (R-PAL+, VQ);
      binary answer/abstain Match where it does not (CoT, R-PAL plain).
  Sel.Acc denominator = all answered (emit=True) items.

RETIRED (2026-08-09): the `sel_acc` field is still computed but is NO LONGER reported in
the paper. It compares each emitted value against `source_gold`, which is the answer to the
*source* FinanceReasoning question, not to the rewritten query. The author model generated
fresh numbers (mq-05 asks 2020->2021 where the source asked 2018->2019; mq-02's source gold
is 3750 where the rewritten inputs give 5274), so no gold transfers and most "correct" hits
were coincidences inside the 1% tolerance. Table 5 now reports answer coverage instead,
which needs no gold. Do not resurrect sel_acc without building real per-item gold first.
"""
from __future__ import annotations
import json, re, sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

_NUM_RE = re.compile(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?")

def _to_float(o):
    """Best-effort numeric parse of a raw system output. Returns None if unparseable.
    Handles plain floats/ints, '$'/'%'/',' formatted strings, and falls back to the
    first numeric token in a free-form string (e.g. a stray dict-repr answer)."""
    if o is None:
        return None
    if isinstance(o, (int, float)):
        return float(o)
    s = str(o).strip()
    try:
        return float(s)
    except ValueError:
        pass
    cleaned = s.replace("$", "").replace(",", "").replace("%", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        pass
    m = _NUM_RE.search(s)
    return float(m.group()) if m else None

def _pct(x):
    """Format a rate as a 1-decimal percentage with round-half-up, matching the
    rounding scripts/assert_paper_numbers.py checks the paper's cells against.
    Python's own .1f rounds half-to-even, which disagrees on exact .X5 ties
    (e.g. 53/80 = 66.25%)."""
    return f"{Decimal(100 * x).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)}"


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from verifiquant.pipeline.run_error_classification_pipeline import _answer_match  # reuse tolerance

DIR = ROOT / "results/messy_query_pilot"
RUNS = DIR / "agreement/runs"

def load_gt():
    gt = {}
    for line in open(DIR / "ground_truth.jsonl"):
        r = json.loads(line)
        gt[r["id"]] = r["label"]
    return gt

def load_gold():
    prov = json.load(open(DIR / "messy_pilot_provenance.json"))
    return {r["id"]: r.get("source_gold") for r in prov}

def load_questions():
    q = {}
    for line in open(DIR / "messy_pilot.jsonl"):
        r = json.loads(line)
        q[r["id"]] = r["question"]
    return q

def adapt_cot():
    out = {}
    for line in open(RUNS / "cot_single_shot.jsonl"):
        r = json.loads(line)
        fa = r["cot_self_improve"]["final_answer"]
        emit = fa is not None
        out[r["case_id"]] = {"emit": emit, "value": fa, "action": ("A" if emit else None)}
    return out

def adapt_rpal_plain():
    out = {}
    for line in open(RUNS / "rpal_plain.jsonl"):
        r = json.loads(line)
        o = r.get("output")
        emit = o not in (None, "")
        val = _to_float(o) if emit else None
        out[r["case_id"]] = {"emit": emit, "value": val, "action": ("A" if emit else None)}
    return out

def adapt_rpal_plus():
    # Structured: assessment.decision / assessment.diagnostic -> 4-way action.
    diag_map = {
        ("execute", "READY"): "A",
        ("warn", "I_SOFT"): "A",       # D1: warned-but-executed counts as emit
        ("refuse", "N"): "O",
        ("clarify", "F"): "R",
        ("clarify", "M"): "O",         # paper convention: N/M -> O
        ("clarify", "I_HARD"): "C",
    }
    out = {}
    for line in open(RUNS / "rpal_plus.jsonl"):
        r = json.loads(line)
        a = r.get("assessment") or {}
        key = (a.get("decision"), a.get("diagnostic"))
        action = diag_map.get(key)
        o = r.get("output")
        emit = o not in (None, "") and action == "A"
        val = _to_float(o) if emit else None
        out[r["case_id"]] = {"emit": emit, "value": val, "action": action}  # action=None => unresolved (2 parse failures)
    return out

def adapt_vq(m_as_clarify=False):
    """m_as_clarify: if True, an M-refusal that carries an actual clarification
    question is scored as C (the funnel behaviorally asked a clarifying question,
    it just exited through the intent gate rather than I_HARD). If False, M follows
    the paper's literal \\textsc{out-of-scope}<->N/M convention and maps to O."""
    def action_of(status, dtype, has_question):
        if status == "success":
            return "A"
        if dtype == "F":          # F wins over generic status=error, per locked rule
            return "R"
        if dtype == "M" and m_as_clarify and has_question:
            return "C"
        if dtype in ("N", "M", "E"):
            return "O"
        return None
    out = {}
    for line in open(RUNS / "vq_k1.jsonl"):
        r = json.loads(line)
        has_q = bool((r.get("clarification_request") or {}).get("questions"))
        action = action_of(r["status"], r["diagnostic_type"], has_q)
        emit = action == "A"       # includes I_SOFT rows: they are status=success
        val = r.get("output_value") if emit else None
        out[r["case_id"]] = {"emit": emit, "value": val, "action": action}
    return out

def score(name, recs, gt, gold, questions):
    ids = sorted(gt)
    rco = [i for i in ids if gt[i] != "A"]        # 46 items, human says non-answer
    n_answered = sum(1 for i in ids if recs[i]["emit"])
    fnar_hits = sum(1 for i in rco if recs[i]["emit"])
    human_fnar = fnar_hits / len(rco)

    # Match: 4-way only if the system exposes a genuine R/C/O sub-classification on
    # its non-answers (R-PAL+, VQ); CoT and plain R-PAL only ever expose "A or
    # abstain" (action is 'A' or None), so scoring those as 4-way would silently
    # drop credit for correct abstention -- fall back to binary answer/abstain there.
    # Unresolved items (action=None on a structured system) count as non-matches,
    # not a reason to demote the whole system to binary.
    has_real_taxonomy = any(recs[i]["action"] in ("R", "C", "O") for i in ids)
    if has_real_taxonomy:
        match_hits = sum(1 for i in ids if recs[i]["action"] == gt[i])
        match_mode = "4-way"
    else:
        match_hits = sum(1 for i in ids if recs[i]["emit"] == (gt[i] == "A"))
        match_mode = "binary(answer-vs-abstain)"
    match = match_hits / len(ids)

    correct = 0
    for i in ids:
        if not recs[i]["emit"]:
            continue
        g = gold.get(i)
        if g is None:
            continue
        _, is_correct, _ = _answer_match(questions.get(i, ""), recs[i]["value"], float(g))
        if is_correct:
            correct += 1
    sel_acc = (correct / n_answered) if n_answered else None

    unresolved = sum(1 for i in ids if recs[i]["action"] is None and not recs[i]["emit"])
    return {
        "system": name, "n_answered": n_answered, "human_fnar": human_fnar,
        "fnar_hits": fnar_hits, "fnar_denom": len(rco),
        "match": match, "match_mode": match_mode,
        "sel_acc": sel_acc, "sel_acc_correct": correct, "sel_acc_denom": n_answered,
        "unresolved_action": unresolved,
    }

def main():
    gt = load_gt(); gold = load_gold(); questions = load_questions()
    systems = {
        "CoT (single-shot)": adapt_cot(),
        "R-PAL (top-4 retrieval)": adapt_rpal_plain(),
        "R-PAL+ (taxonomy prompt)": adapt_rpal_plus(),
        "VerifiQuant (K=1, M->O)": adapt_vq(m_as_clarify=False),
        "VerifiQuant (K=1, M->C)": adapt_vq(m_as_clarify=True),
    }
    results = []
    for name, recs in systems.items():
        missing = set(gt) - set(recs)
        if missing:
            print(f"WARNING {name}: missing ids {missing}", file=sys.stderr)
        results.append(score(name, recs, gt, gold, questions))

    print(json.dumps(results, indent=2))
    print("\n-- LaTeX rows (tab:v2ext) --")
    for r in results:
        sa = f"{_pct(r['sel_acc'])}\\%" if r["sel_acc"] is not None else "---"
        print(f"{r['system']:28s} & {_pct(r['human_fnar'])}\\% & {_pct(r['match'])}\\% & {sa} \\\\  "
              f"% match_mode={r['match_mode']} n_answered={r['n_answered']} unresolved={r['unresolved_action']}")

    out_dir = DIR / "agreement"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "score_report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
