# SUPERSEDED scoring — do not cite these numbers

`pal_gemini_25_flash_summary.json` reports the PAL baseline under the **legacy scorer**:
accuracy **90.4% / SWR 8.0%** (226 correct / 20 SW / 4 abstain).

The paper (Panel A) uses the **unified VQ scorer** version instead:
accuracy **91.2% / SWR 7.2%** (228 correct / 18 SW / 4 abstain).

Canonical source: `../retrieval_pal_baseline/rpal_top4_gemini_25_flash_comparison.json`
(`system_metrics.plain_pal_unified_vq_scorer`). The unified scorer applies VQ's 1%
tolerance, percent/decimal normalization, boolean-aware parsing, and AST policy to every
Panel A system, making the SWR column apples-to-apples.

Rationale: `docs/2026-07-29_retrieval_pal_matched_baseline.md` (L84–85).
Reflected in `paper/numbers.tex` (\palAccAll = 91.2%, \palSWRAll = 7.2%).
