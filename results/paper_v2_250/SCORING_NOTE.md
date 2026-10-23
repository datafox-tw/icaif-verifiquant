# Scoring note — canonical vs superseded numbers in this run dir

## CoT single-shot silent-wrong split
`paper_results_summary.json` records CoT single-shot as **56 SW / 0 abstain / SWR 22.4%**
(raw aggregate: every non-numeric / prose refusal counted as silent-wrong).

The **canonical** Panel A numbers apply the paper's stated convention (informal prose
refusals = abstention; `paper/main_body_v2.tex` §Problem Formulation) plus the boolean-aware
scorer: **194 correct / 45 SW / 11 abstain / SWR 18.0%**, in
`results/cot_single_shot_flash/tier_breakdown.json` (medium 148/25/7 + hard 46/20/4).
This matches `paper/numbers.tex` (\cotSWRAll = 18.0%) and the bootstrap point estimate in
`results/stats_250q.json` (cot swr = 0.18). **Do not cite the 22.4% figure.**

## VerifiQuant canonical run
`results/vq_flash/` (2026-07-12, type-audited cards) is canonical: 234 / 8 / 8 = 93.6%.
`results/vq_flash_run1_prescalarfix/` (231 / 10 / 9 = 92.4%) is the earlier pre-fix run — superseded.

## Scorer consistency (re-scoring is a no-op)
The 250Q harness that scored VQ and CoT here is already boolean- and nearest-integer-aware:
VQ scores the only 3 boolean-gold cases (test-1842=True, test-2059=False, test-2125) and the
nearest-integer case (test-1890=138000) correctly under the final scorer's rules
(test-2125 abstains by the math-only AST policy). The 2026-07-29 boolean/rounding fixes
brought the *legacy R-PAL* scorer up to this harness; they do not change VQ/CoT counts.
