# Which gold to use for the messy-query set

**Do not score this set with `source_gold` from `messy_pilot_provenance.json`.**

`source_gold` is the answer to the *source* FinanceReasoning question that a topic was drawn
from. GPT-5.2 rewrote each topic into a new naturalistic query **with fresh numbers**, so that
value does not correspond to the query in `messy_pilot.jsonl`. (The 18 missing-input items were
then revised offline with an OpenAI Codex agent, seeded by the GPT-5.2 versions, before any label
was collected; `author_model` records them as `gpt-5.2_seed_then_codex_curated_v3`.)

Measured on the 34 items whose human majority label is `ANSWER`, `source_gold` disagrees with
the query's actual answer on **30 of 34**:

| item | true answer for the rewritten query | `source_gold` |
|---|---|---|
| mq-78 | 18 (percent change, nearest whole) | −451500 |
| mq-72 | 55.556 | 2 |
| mq-61 | 1.572 | 74 |
| mq-60 | 1814 | 245 |
| mq-64 | 558812 | 312620 |
| mq-22 | 9.576 | −3.52 |

Only four items agree, and three of those are coincidences inside the 1% comparison tolerance
(mq-19: 8.60 vs 8.64; mq-46: −31.52 vs −31.42; mq-38: both round to 1). A concrete example of
the drift: mq-05 asks for the change from **2020 to 2021** while its source question asked
**2018 to 2019**.

## What to use instead

- **Action labels** (`ground_truth.jsonl`) are the primary ground truth. They come from five
  finance-trained annotators who were blind to every system, and they are what the paper's
  Human-FNAR and Match numbers are computed from. Nothing here affects them.
- **Numeric answers** for the 34 `ANSWER` items are derived in
  `scripts/build_messy_answer_gold.py` and written to `answer_gold.json`. Every entry carries
  the formula inline, and the six entries where more than one reading is defensible are flagged
  `convention_dependent`.

`source_gold` is retained only to record which source topic each query came from.
