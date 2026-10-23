#!/usr/bin/env python3
"""
attribute_messy_shortfalls.py
-----------------------------
For each messy-query item the human panel labeled ANSWER but VerifiQuant (K=1)
declined, decide whether the 250-card library holds an applicable contract.

Why this exists
---------------
The paper says top-4 retrieval "never surfaces an applicable contract" on 24 of the
27 shortfalls. That wording is neutral between two different causes, and reviewers
read it both ways: the card exists but lexical retrieval missed it, or the library
holds no such card. This script separates the two.

How a verdict is reached
------------------------
A card is APPLICABLE when its declared purpose covers the asked quantity and its
code computes it from the inputs the query gives. A card that is only arithmetically
similar but declared for another quantity (FIFO for a LIFO ask, key-rate duration
with a hard-coded 1% shift for an effective-duration ask at 50 bp) is a near
neighbor, not applicable: selecting it is the force-fit the funnel exists to stop.

The verdicts below were made by reading all 250 card titles, descriptions and
code, and were adjudicated by the first author on 2026-10-14:

- `fic_article_687` is titled "Base Effect" but its declared description is a
  generic percent change between a current and a previous value, so it is
  applicable to the six year-over-year percent-change asks.
- Cards whose arithmetic coincides with a percent change but whose declared purpose
  is another quantity (Quarter over Quarter `2532`, EPS growth `1319`, CAGR `137`)
  are not counted as applicable for these six, because none of the six asks about a
  quarter, an EPS figure or a multi-period compound rate. The depth column therefore
  tracks `687` only. (If those cards were counted, k=8 or 16 would recover 2 of the
  7 instead of 1; the split 17 absent / 7 present does not change.)
- mq-62 (weighted average maturity) is present: the generic weighted-average card
  computes a balance-weighted mean of maturities from the given pools.
- mq-26 (effective duration from +/-50 bp prices) is absent: the only duration card
  hard-codes a 1% shift, so its code cannot use the inputs the query gives.

The depth column is recomputed here from the same store and retriever the run used
(no model calls), so it does not rest on anyone's judgment.

Run:  python3 scripts/attribute_messy_shortfalls.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
MQ = ROOT / "results/messy_query_pilot"
DB = ROOT / "results/paper_v2_250/fic/cards.db"
OUT = MQ / "shortfall_attribution.csv"

PCT_CHANGE = "fic_article_687"    # titled "Base Effect"; description and code are a generic percent change
WEIGHTED_AVG = "fic_article_2448"  # "Weighted Average" (2451 is a duplicate family)

# id -> (what is asked, verdict, applicable card, note, needs a domain check)
# verdict: present = applicable card exists, absent = none in the library
VERDICTS: dict[str, tuple[str, str, str, str, bool]] = {
    # ── applicable card exists, retrieval ranked it outside the top 4 ──────────
    "mq-05": ("percent change", "present", PCT_CHANGE, "", False),
    "mq-20": ("percent change", "present", PCT_CHANGE, "", False),
    "mq-30": ("percent change", "present", PCT_CHANGE, "", False),
    "mq-53": ("percent change", "present", PCT_CHANGE, "", False),
    "mq-75": ("percent change", "present", PCT_CHANGE, "", False),
    "mq-78": ("percent change", "present", PCT_CHANGE, "", False),
    "mq-62": ("weighted average maturity", "present", WEIGHTED_AVG,
              "named metric mapped onto the generic weighted-average card; top-4 held WALA only; adjudicated 2026-10-14", False),
    # ── no applicable card: elementary arithmetic the library does not hold ────
    "mq-11": ("part / whole share", "absent", "", "no generic proportion card", False),
    "mq-19": ("part / whole share", "absent", "", "no generic proportion card", False),
    "mq-34": ("part / whole share", "absent", "", "no generic proportion card", False),
    "mq-69": ("part / whole share", "absent", "", "no generic proportion card", False),
    "mq-70": ("part / whole share", "absent", "", "debt-to-capitalization card is a different ratio", False),
    "mq-38": ("ratio of two values", "absent", "", "", False),
    "mq-40": ("absolute difference", "absent", "", "", False),
    # ── no applicable card: named financial formula ────────────────────────────
    "mq-02": ("capital-gains tax on net gains", "absent", "", "carryover and Form 6781 cards do not net and tax", False),
    "mq-04": ("oil netback per barrel", "absent", "", "", False),
    "mq-08": ("absorption cost per unit", "absent", "", "", False),
    "mq-09": ("LIFO cost of goods sold", "absent", "", "only a FIFO card exists, and it was in the top 4", False),
    "mq-18": ("EV / EBITDA", "absent", "", "", False),
    "mq-23": ("dividend-adjusted forward price", "absent", "", "", False),
    "mq-26": ("effective duration", "absent", "",
              "key-rate duration card hard-codes a 1% shift, the query uses 50 bp; it was in the top 4; adjudicated 2026-10-14", False),
    "mq-32": ("debt-to-equity", "absent", "", "only debt-to-capitalization exists", False),
    "mq-64": ("lump-sum future value with an annual fee", "absent", "", "future-value cards carry no fee term", False),
    "mq-74": ("basis of gifted stock", "absent", "", "", False),
    # ── declined at a later gate, not a scope refusal ──────────────────────────
    "mq-27": ("average of high and low", "other", "", "selected Kijun-Sen, a near neighbor; an E-check stopped it", False),
    "mq-61": ("percent change, phrased as unsure", "other", PCT_CHANGE, "M exit with a clarification question", False),
    "mq-67": ("consolidated taxable income", "other", "", "selected the individual taxable-income card; F exit on unbound inputs", False),
}


def _jsonl(path: Path, key: str) -> dict[str, dict]:
    return {json.loads(line)[key]: json.loads(line) for line in path.open()}


def main() -> None:
    from verifiquant.card_store import SQLAlchemyArtifactStore

    store = SQLAlchemyArtifactStore(f"sqlite:///{DB}")
    truth = _jsonl(MQ / "ground_truth.jsonl", "id")
    queries = _jsonl(MQ / "messy_pilot.jsonl", "id")
    vq = _jsonl(MQ / "agreement/runs/vq_k1.jsonl", "case_id")

    shortfalls = [i for i, t in truth.items() if t["label"] == "A" and vq[i]["status"] != "success"]
    missing = sorted(set(shortfalls) ^ set(VERDICTS))
    if missing:
        raise SystemExit(f"verdict table and shortfall set disagree on: {missing}")

    def top(text: str, k: int) -> list[str]:
        return [r.get("fic_id") for r in store.retrieve_candidates(query=text, top_k=k)]

    rows = []
    for i in shortfalls:
        asked, verdict, card, note, check = VERDICTS[i]
        text = f"{queries[i]['question']}\n{queries[i].get('context', '')}"
        recorded = [c.get("fic_id") for c in (vq[i].get("selection_trace") or {}).get("retrieval_candidates", [])]
        if top(text, 4) != recorded:
            raise SystemExit(f"{i}: top-4 does not reproduce the recorded run, wrong store or retriever")
        # The retriever's candidate pool grows with top_k (pool = max(8k, 40)), so the order
        # at k=250 is not the order at k=4. The usable quantity is the depth the run would
        # have needed: the smallest k whose top-k contains the card.
        depth = next((k for k in range(4, 251) if card in top(text, k)), "") if card else ""
        rows.append({
            "id": i,
            "vq_exit": vq[i]["diagnostic_type"],
            "asked": asked,
            "verdict": verdict,
            "applicable_card": card,
            "depth_needed": depth,
            "top4": " | ".join(c.get("title", "") for c in (vq[i].get("selection_trace") or {}).get("retrieval_candidates", [])),
            "note": note,
            "check": "yes" if check else "",
        })

    with OUT.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    scope = [r for r in rows if r["vq_exit"] == "N"]
    present = [r for r in scope if r["verdict"] == "present"]
    ranks = sorted(int(r["depth_needed"]) for r in present)
    print(f"shortfalls {len(rows)}  |  N refusals {len(scope)}  |  later gates {len(rows) - len(scope)}")
    print(f"N refusals: applicable card present {len(present)}, absent {len(scope) - len(present)}")
    print(f"  retrieval depth each present card needed: {ranks}  (k=8 recovers {sum(r <= 8 for r in ranks)}, k=16 recovers {sum(r <= 16 for r in ranks)})")
    print(f"  resting on {PCT_CHANGE} alone: {sum(r['applicable_card'] == PCT_CHANGE for r in present)}")
    print("  by what was asked:", dict(Counter(r["asked"] for r in scope)))
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
