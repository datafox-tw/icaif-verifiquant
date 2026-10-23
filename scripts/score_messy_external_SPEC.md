# Spec — RQ4 messy-query external-validity scorer

Fills the 12 cells of `tab:v2ext` (Human-FNAR / Match / Sel.Acc for CoT, R-PAL, R-PAL+, VQ).
You write the script; this is the contract.

## Two steps (keep them separate)

1. **Run each system on the 80 messy queries** (needs `GEMINI_API_KEY` + `cards.db`) with the
   *existing* runners, saving one output file per system. Do NOT rebuild runners.
   - VQ (K=1): `run_error_classification_pipeline.py` on `messy_pilot.jsonl`.
   - R-PAL / R-PAL+ / CoT: the same baseline runners used for the 250Q eval, pointed at
     `messy_pilot.jsonl`.
2. **This scorer** (mostly offline): read the run outputs + human GT, emit the table.

## Inputs

- `results/messy_query_pilot/ground_truth.jsonl`
  — `{id, label∈{A,R,C,O}, votes, n_raters}`, 80 rows. (already frozen; 0 disagreement pool)
- `results/messy_query_pilot/messy_pilot_provenance.json`
  — join by `id` to get `source_gold` (numeric). **`messy_pilot.jsonl` has NO gold** — get it here.
- Per-system run output (one file each): must expose, per `id`, enough to derive
  `(emit, value, action)` below.

## Normalize every system to one record per id

```
emit   : bool     # did the system COMMIT a numeric answer?
value  : float|None
action : "A"|"R"|"C"|"O"   # the system's behavior mapped to the human taxonomy
```

### VQ adapter (structured — from DiagnosticReport)

| status / diagnostic_type            | action | emit  |
|-------------------------------------|--------|-------|
| status=success                      | A      | True (value=result) |
| status=needs_clarification (I_HARD) | C      | False |
| status=refusal, type=F              | R      | False |
| status=refusal/alert, type=N or M   | O      | False |
| status=alert, type=E (boundary)     | O      | False |
| status=error                        | O      | False |

- **Decision D1 (I_SOFT):** I_SOFT warns but still computes → `emit=True, action=A`. Count it as an
  emitted number for FNAR (a warned number is still a committed number). Note this in the caption.

### Baseline adapter (CoT / R-PAL / R-PAL+) — the hard part

- `emit` = a committed number was produced. Reuse the pipeline's answer-extraction so
  "cannot solve" / parse-fail / raise / a clarifying question all → `emit=False`.
- `action`:
  - `emit=True` → `A`.
  - `emit=False` → classify the refusal text into `R/C/O`. **Decision D2 — pick one:**
    - (a) **LLM judge** with the exact A/R/C/O rubric (from `LABELING_GUIDELINE.md`) over the
      refusal text. Fair 4-way, but adds an LLM dependency; hand-check ~10 to validate.
    - (b) **Binary Match instead**: collapse Match to "answer-vs-abstain agreement" (system answered
      iff human=A) and relabel the column. No classifier needed. Cleaner, less informative.
  - Recommendation: **(b) for the first submittable table** (robust, no new LLM noise), and keep
    Human-FNAR as the headline. Upgrade to (a) later if a reviewer wants the 4-way.

## Metrics (exact)

Let `RCO = {id : GT=R,C,O}` (46 items), `AA = {id : GT=A}` (34 items).

- **Human-FNAR(S)** = `#{id∈RCO : emit_S} / 46`. Lower = safer. **This is the headline** (mirrors
  Panel B: emitting a number where humans say none is legitimate).
- **Match(S)** = 4-way `#{id : action_S == GT} / 80` (VQ), or binary per D2 for baselines.
- **Sel.Acc(S)** = `#{id : emit_S AND GT=A AND num_match(value_S, gold)} / #{id : emit_S}`.
  Answering any R/C/O item is automatically not correct (no valid single gold), so it depresses
  Sel.Acc — that is intended.
- Also record, for the caption / honesty: `n_answered_S = #{emit_S}`, and FNAR denominator = 46.

`num_match(pred, gold)`: **reuse** the pipeline comparison
(`run_error_classification_pipeline.py` ~L441–470: `rel_tol=1e-6, abs_tol=1e-2`, plus the
`output_scale_x100` decimal-vs-percentage-point handling). Do not invent a new tolerance.

## Output

1. `agreement/score_report.json`: per system `{human_fnar, match, sel_acc, n_answered}` + denominators.
2. Printed LaTeX rows ready to paste into `tab:v2ext` (replace the 12 `\TODO{}`), e.g.
   `CoT (single-shot) & 61.0\% & 48.8\% & 79.1\% \\` (illustrative only).

## Edge cases / gotchas

- **A3 covers 79/80 items** → one item has `n_raters=4` (majority still ≥3). GT already handles it;
  keep the per-id `n_raters` in mind if you weight anything.
- **Gold on R/C/O items is meaningless** (the messy edit removed/mangled the input). Only use gold
  for `num_match` on `GT=A` items. Never "credit" an answered R/C/O item as correct.
- **Ambiguous (C) gold** is one interpretation only; an answered C item is still a false acceptance.
- **VQ E-alert**: decide once (D1/E→O) and apply uniformly.
- **Author self-labels are NOT in GT** (independence). If you want an author-vs-panel agreement
  number, score `author_selfcheck.csv` separately; never merge into `ground_truth.jsonl`.
- Freeze `ground_truth.jsonl` before any system run (already frozen; don't re-aggregate after seeing
  system outputs).

## Decisions to lock before coding
- [ ] D1: I_SOFT warned answer counts as emit=True (recommended yes).
- [ ] D2: baseline Match = binary answer/abstain (recommended) vs LLM 4-way.
- [ ] Sel.Acc denominator = all answered (recommended) vs answered∧GT=A.
