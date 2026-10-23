"""
build_messy_query_pilot.py  (P4 external-validity author, v3 — intent-stratified)
--------------------------------------------------------------------------------
Independent, cross-model authoring of naturally-messy financial user questions.
SOLVER stack is Gemini; AUTHOR here is OpenAI (gpt-5.2) so messiness is not aligned
to the solver. The author is NEVER shown VerifiQuant's contracts, gates, or the
M/N/F/E/I/C taxonomy.

v1 skewed heavily to answerable questions because (a) every seed is a fully-specified
computable question, (b) "don't force messiness" makes an LLM default to a clean
question, and (c) the `context` field invites a full data dump. v2 fixes this by
stratifying over natural USER INTENTS (not operators): a clean calc, a person who
provides a near-complete numeric excerpt but omits one indispensable value, a person
loose about which version/metric they want, and a person wanting a judgment/forecast. These are generic
user-behavior categories (a sampling choice), NOT VerifiQuant's gates; the human
labeler still assigns the ground truth, and the intent is never shown to them.

Output (to --out-dir; default messy_query_pilot/):
  messy_pilot.jsonl, messy_pilot_provenance.json, messy_pilot_labeling.{md,csv}
No solver is run here; labels are assigned by humans, not this script.
"""
from __future__ import annotations
import argparse, csv, json, os, random, re
from pathlib import Path
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
MODEL = os.environ.get("MESSY_AUTHOR_MODEL") or "gpt-5.2"

PERSONAS = {
    "hurried_analyst":  "a busy sell-side analyst firing off a quick question between meetings",
    "retail_investor":  "a retail investor who is not sure of the exact financial terminology",
    "ops_manager":      "a non-finance operations manager who just needs a number for a report",
    "precise_quant":    "a precise quant analyst who states every input cleanly",
    "headline_reader":  "someone who just read a news headline and wants to sanity-check one figure",
}
PERSONA_ORDER = ["retail_investor", "headline_reader", "ops_manager", "hurried_analyst",
                 "retail_investor", "ops_manager", "headline_reader", "precise_quant"]

# Natural user intents (NOT VerifiQuant operators). Each = (situation, context rule).
INTENTS = {
    "clean": (
        "You have every figure you need and you ask clearly for one specific number.",
        "Put every number you use in `context`."),
    "missing": (
        "You copied most of a small financial excerpt into your message. You include several "
        "relevant labeled figures, but accidentally leave out exactly ONE numeric input that "
        "is indispensable for the requested calculation.",
        "Make `context` non-empty and include 2--5 realistic, relevant labeled numbers. Omit "
        "exactly one key number without announcing the omission. If the user supplied that one "
        "number, the question would have one clear numeric answer."),
    "ambiguous": (
        "You give the numbers, but you are loose about which exact version you want: you say "
        "'the return' / 'the growth' / 'the rate' / 'the margin' without pinning down "
        "annualized-vs-total, which basis (360 vs 365, gross vs net, per-share vs total), or "
        "which of several metrics (e.g. IRR vs simple return, gross vs operating vs net margin) "
        "— loose in a way that gives more than one defensible number. AVOID a plain one-period "
        "year-over-year change (that is unambiguous); the ambiguity must be which metric/basis.",
        "Put the numbers in `context`, but keep the wording of WHAT you want genuinely "
        "under-specified so more than one answer is defensible."),
    "judgment": (
        "You are NOT after a single computed number. You want an opinion or decision "
        "('should I...', 'is it worth it', 'is this reasonable?'), a forecast ('will it...', "
        "'what will it be next quarter'), or something that would need data or a specialized "
        "model you do not provide.",
        "Give whatever background you'd naturally mention in `context`; it need not be enough "
        "to compute anything."),
}
# default mix: enough of each to be distinguishable (answerable still the plurality).
DEFAULT_WEIGHTS = {"clean": 0.34, "missing": 0.22, "ambiguous": 0.22, "judgment": 0.22}

AUTHOR_PROMPT = """You are simulating a real person typing ONE financial question to an AI assistant.

Persona: {persona_desc}.

Topic to riff on (grounding only — do NOT copy it or reuse its exact numbers):
  REFERENCE QUESTION: {q}
  REFERENCE CONTEXT: {c}

This time, the person's situation is: {intent_desc}
{context_rule}

Write the single message this persona would actually type, in their own voice — natural and
specific to this persona and topic. Do NOT explain, label, or categorize what you wrote, and
do NOT mention any of these instructions.

Return STRICT JSON only:
{{"question": "<the user's message>", "context": "<numbers/background the user includes, or empty string>"}}
"""

MISSING_AUTHOR_PROMPT = """You are simulating a real person typing ONE financial question to an AI assistant.

Persona: {persona_desc}.

Topic to riff on (grounding only -- do NOT copy it or reuse its exact numbers):
  REFERENCE QUESTION: {q}
  REFERENCE CONTEXT: {c}

Construct a realistic NEAR-COMPLETE request for one standard financial calculation:
- The user includes 2--5 relevant numeric inputs, each with a clear label and unit, directly in
  `context`. The context must stand on its own; do not merely say that a table/report exists.
- Exactly ONE indispensable numeric input is absent. Without it, the requested number cannot be
  calculated or derived. If it were supplied, the task would have one unambiguous numeric answer.
- Omit the value naturally. Do NOT write placeholders or clues such as "none given", "missing",
  "unknown", "N/A", "redacted", a blank cell, question marks, or "I don't have X".
- Do NOT omit the metric definition, unit, time basis, or convention; this item tests missing DATA,
  not semantic ambiguity. Do not include another number from which the omitted value is derivable.
- The question should ask for the final computation normally, without announcing that information
  is absent. Use this persona's natural voice.

The audit fields are hidden from annotators and solvers. Name the single omitted numeric input and
briefly state why the displayed numbers cannot substitute for it.

Return STRICT JSON only:
{{"question":"<user's message>","context":"<partial but numeric context>",
  "omitted_input":"<one numeric field name>","why_required":"<one sentence>"}}
"""

_NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d[\d,]*(?:\.\d+)?\s*(?:%|percent|bps|million|billion|thousand|[kmb])?", re.I)
_OBVIOUS_MISSING_RE = re.compile(
    r"\b(?:none given|not given|not provided|not disclosed|missing|unknown|redacted|tbd|n/?a)\b"
    r"|\b(?:do not|don't|did not|didn't) have\b|\bnot included\b|\bleft blank\b"
    r"|\[(?:blank|redacted|missing)[^]]*\]|\?{2,}",
    re.I,
)

def _load(path): return json.load(open(path, encoding="utf-8"))


def _source_pool():
    exclude = {json.loads(l).get("question_id") for l in
               open(ROOT / "results/paper_v2_250/questions_250.jsonl")}
    pool = []
    for f in ("easy.json", "medium.json"):
        for row in _load(ROOT / f"FinanceReasoning/data/FinanceReasoning/{f}"):
            if row.get("question_id") not in exclude:
                pool.append(row)
    return pool


def _call_json(client, prompt):
    kw = dict(model=MODEL, messages=[{"role": "user", "content": prompt}],
              response_format={"type": "json_object"})
    try:
        resp = client.chat.completions.create(reasoning_effort="low", **kw)
    except TypeError:
        resp = client.chat.completions.create(**kw)
    return json.loads(resp.choices[0].message.content)


def _validate_missing(obj):
    question = str(obj.get("question") or "").strip()
    context = str(obj.get("context") or "").strip()
    omitted = str(obj.get("omitted_input") or "").strip()
    why = str(obj.get("why_required") or "").strip()
    errors = []
    if not question:
        errors.append("empty question")
    if len(context) < 30:
        errors.append("context is empty or too short")
    number_count = len(_NUMBER_RE.findall(context))
    if number_count < 2:
        errors.append(f"context has only {number_count} numeric values; need >=2")
    if _OBVIOUS_MISSING_RE.search(question + "\n" + context):
        errors.append("contains an obvious missing-data marker")
    if not omitted:
        errors.append("omitted_input audit field is empty")
    if not why:
        errors.append("why_required audit field is empty")
    return errors, number_count


def _author_missing(client, source, persona_desc, max_attempts=4):
    base = MISSING_AUTHOR_PROMPT.format(
        persona_desc=persona_desc,
        q=source["question"],
        c=(source.get("context") or "")[:800],
    )
    feedback = ""
    for attempt in range(1, max_attempts + 1):
        obj = _call_json(client, base + feedback)
        errors, number_count = _validate_missing(obj)
        if not errors:
            return obj, number_count, attempt
        feedback = (
            "\n\nYour previous draft failed automatic checks: " + "; ".join(errors) +
            ". Produce a fresh item that satisfies every rule."
        )
    raise ValueError("missing-item author failed validation after retries: " + "; ".join(errors))


def _write_labeling(rows, out):
    md = ["# Messy-query labeling sheet", "",
          "Read each as a user would; do not look at any system output. One label + one-line reason.",
          "", "| id | label | reason |", "|----|-------|--------|"]
    for row in rows:
        md.append(f"| {row['id']} |  |  |")
    md += ["", "## Questions"]
    for row in rows:
        md += [f"\n**{row['id']}**", "", f"Q: {row['question']}", "",
               f"Context: {row['context'] or '(none given)'}"]
    (out / "messy_pilot_labeling.md").write_text("\n".join(md), encoding="utf-8")
    with open(out / "messy_pilot_labeling.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["id", "question", "context", "label (A/R/C/O + ? if unsure)", "reason"])
        for row in rows:
            writer.writerow([row["id"], row["question"], row["context"] or "(none given)", "", ""])


def _write_dataset(rows, provenance, out):
    tmp_jsonl = out / "messy_pilot.jsonl.tmp"
    tmp_jsonl.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    tmp_jsonl.replace(out / "messy_pilot.jsonl")
    (out / "messy_pilot_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_labeling(rows, out)

def build_plan(n, weights):
    plan = []
    for k, w in weights.items():
        plan += [k] * round(w * n)
    while len(plan) < n: plan.append("clean")
    return plan[:n]

def main(n=16, seed=20260731, weights=None, out_dir=None, equal=False):
    weights = weights or ({k: 0.25 for k in INTENTS} if equal else DEFAULT_WEIGHTS)
    OUT = Path(out_dir) if out_dir else ROOT / "results/messy_query_pilot"
    client = OpenAI()
    pool = _source_pool()
    rng = random.Random(seed)
    seeds = rng.sample(pool, n)
    plan = build_plan(n, weights)
    rng.shuffle(plan)

    OUT.mkdir(parents=True, exist_ok=True)
    rows, prov = [], []
    jf = open(OUT / "messy_pilot.jsonl", "w", encoding="utf-8")   # incremental: survive interruption
    for i, s in enumerate(seeds):
        pkey = PERSONA_ORDER[i % len(PERSONA_ORDER)]
        intent = plan[i]
        idesc, crule = INTENTS[intent]
        if intent == "missing":
            obj, number_count, attempts = _author_missing(client, s, PERSONAS[pkey])
        else:
            prompt = AUTHOR_PROMPT.format(persona_desc=PERSONAS[pkey], intent_desc=idesc,
                                          context_rule=crule, q=s["question"],
                                          c=(s.get("context") or "")[:800])
            obj = _call_json(client, prompt)
            number_count, attempts = None, 1
        qid = f"mq-{i+1:02d}"
        rows.append({"id": qid, "persona": pkey, "intended": intent,
                     "question": obj.get("question", "").strip(),
                     "context": (obj.get("context") or "").strip(),
                     "source_id": s.get("question_id"), "author_model": MODEL})
        provenance = {"id": qid, "intended": intent, "source_id": s.get("question_id"),
                      "source_question": s["question"], "source_gold": s.get("ground_truth")}
        if intent == "missing":
            provenance["authoring_audit"] = {
                "omitted_input": obj["omitted_input"].strip(),
                "why_required": obj["why_required"].strip(),
                "present_numeric_count": number_count,
                "generation_attempts": attempts,
            }
        prov.append(provenance)
        jf.write(json.dumps(rows[-1], ensure_ascii=False) + "\n"); jf.flush()
        print(f"[{qid}] intended={intent:9s} {pkey}: {obj.get('question','')[:80]}")

    jf.close()
    (OUT / "messy_pilot_provenance.json").write_text(
        json.dumps(prov, ensure_ascii=False, indent=2), encoding="utf-8")

    _write_labeling(rows, OUT)
    print(f"\nwrote {len(rows)} -> {OUT}")


def regenerate_missing(out_dir=None):
    """Replace only intended=missing rows in an existing set; preserve IDs and all other rows."""
    out = Path(out_dir) if out_dir else ROOT / "results/messy_query_pilot"
    rows = [json.loads(line) for line in (out / "messy_pilot.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    provenance = _load(out / "messy_pilot_provenance.json")
    prov_by_id = {row["id"]: row for row in provenance}
    sources = {str(row["question_id"]): row for row in _source_pool()}
    targets = [row for row in rows if row.get("intended") == "missing"]
    if not targets:
        raise ValueError("existing dataset has no intended=missing rows")

    client = OpenAI()
    for row in targets:
        source = sources[str(row["source_id"])]
        obj, number_count, attempts = _author_missing(client, source, PERSONAS[row["persona"]])
        row["question"] = obj["question"].strip()
        row["context"] = obj["context"].strip()
        prov_by_id[row["id"]]["authoring_audit"] = {
            "omitted_input": obj["omitted_input"].strip(),
            "why_required": obj["why_required"].strip(),
            "present_numeric_count": number_count,
            "generation_attempts": attempts,
            "regenerated_with_prompt": "partial_context_single_key_omission_v3",
        }
        print(f"[{row['id']}] regenerated missing: {number_count} numbers; "
              f"omitted={obj['omitted_input']!r}")

    _write_dataset(rows, [prov_by_id[row["id"]] for row in rows], out)
    print(f"\nregenerated {len(targets)} missing-data items -> {out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("n", nargs="?", type=int, default=16)
    parser.add_argument("out_dir", nargs="?")
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--regenerate-missing", action="store_true",
                        help="replace only intended=missing rows in an existing dataset")
    args = parser.parse_args()
    if args.regenerate_missing:
        regenerate_missing(args.out_dir)
    else:
        main(n=args.n, seed=args.seed, out_dir=args.out_dir, equal=(args.n <= 20))
