#!/usr/bin/env python3
"""
build_messy_answer_gold.py
--------------------------
Derive a numeric gold answer for each of the 34 messy-query items whose human
majority label is ANSWER, and score every system's answers against it.

Why this exists
---------------
`messy_pilot_provenance.json` carries a `source_gold` field, but that is the answer
to the *source* FinanceReasoning question. The author model rewrote each topic with
fresh values (mq-05 asks 2020->2021 where the source asked 2018->2019; mq-02's source
gold is 3750 where the rewritten inputs give 5274), so `source_gold` does not transfer
and must never be used to score this set.

An ANSWER label means the panel judged that the query is complete and admits exactly
one number, so a gold value is derivable from the query text alone. Each entry below
states the formula explicitly so a reader can check it without rerunning anything.

CONVENTION entries are the ones where more than one defensible reading exists. They
are listed separately so a domain reader can adjudicate them rather than having to
trust an author's silent choice.

Run:  python3 scripts/build_messy_answer_gold.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MQ = ROOT / "results/messy_query_pilot"

# ── mechanical: one reading only ─────────────────────────────────────────────
pct = lambda new, old: (new - old) / old * 100

MECHANICAL: dict[str, tuple[float, str]] = {
    "mq-02": ((42500 - 13200) * 0.18, "(42500-13200)*0.18  tax on net gains"),
    "mq-04": (82.50 - 9.25 - 18.40 - 7.60, "82.50-9.25-18.40-7.60  netback/bbl"),
    "mq-05": (pct(15620, 14275), "(15620-14275)/14275  net sales %chg"),
    "mq-08": ((31600 + 18900 + 6750 + 12250) / 620, "(DM+DL+VOH+FOH)/620  absorption cost/unit"),
    "mq-09": (50 * 67 + 35 * 62 + 10 * 58, "LIFO: 50@67 + 35@62 + 10@58 for 95 units"),
    "mq-11": (18.6 / 142.0 * 100, "18.6/142.0  share due within 12 months"),
    "mq-19": (44120 / 512880 * 100, "44120/512880  share due >5 years"),
    "mq-20": (pct(387.4, 412.6), "(387.4-412.6)/412.6  basic share count %chg"),
    "mq-22": (pct(52.18, 47.62), "(52.18-47.62)/47.62  Q3->Q4 high %chg"),
    "mq-27": ((27.84 + 19.16) / 2, "(27.84+19.16)/2  average of high and low"),
    "mq-30": (pct(915.4, 842.6), "(915.4-842.6)/842.6  goodwill %chg"),
    "mq-32": (942560 / 2413775 * 100, "942560/2413775  debt-to-equity as %"),
    "mq-34": (143 / 160 * 100, "143/160  US share of leased locations"),
    "mq-38": (round(95000 / 130000), "95000/130000 rounded to nearest whole"),
    "mq-40": (abs(31790 - 45215), "|31790-45215|  absolute difference"),
    "mq-46": (pct(18.9, 27.6), "(18.9-27.6)/27.6  unbilled receivables %chg"),
    "mq-53": (pct(1214680, 1089450), "(1214680-1089450)/1089450  long-lived assets %chg"),
    "mq-56": (pct(1037950, 842300), "(1037950-842300)/842300  revenue %chg"),
    "mq-60": (round(math.sqrt(2 * 48000 * 120 / 3.50)), "sqrt(2DS/H) EOQ, nearest unit"),
    "mq-61": (pct(4781, 4707), "(4781-4707)/4707  Transportation Solutions %chg"),
    "mq-62": ((1.25 * 45 + 3.50 * 120 + 5.25 * 300) / (1.25 + 3.50 + 5.25),
              "value-weighted average maturity in days"),
    "mq-67": (92500 + 38200 - 9600 + 2400 + 5150 - 3850,
              "combine all six items as the prompt instructs"),
    "mq-69": (276.5 / 425.0 * 100, "276.5/425.0  employee-related share"),
    "mq-70": (round(37.9 / 84.6 * 100), "37.9/84.6 to nearest whole percent"),
    "mq-72": (pct(14, 9), "(14-9)/9  additions for prior-year tax positions %chg"),
    "mq-75": (pct(24.10, 18.75), "(24.10-18.75)/18.75  hedging FV %chg"),
    "mq-77": (pct(41, 28), "(41-28)/28  excess-inventory hedge %chg"),
    "mq-78": (round(pct(214900, 182750)), "(214900-182750)/182750 to nearest whole percent"),
}

# ── convention-dependent: a domain reader should sign these off ──────────────
_fwd23 = (1.0228 ** 3 / 1.0172 ** 2 - 1) * 100
_pv_div = 0.60 * math.exp(-0.031 * 0.1667) + 0.75 * math.exp(-0.031 * 0.5)

CONVENTION: dict[str, tuple[float, str, str]] = {
    "mq-18": (
        (62 + 18 - 6) / 23,
        "EV/EBITDA = (mktcap + debt - cash)/EBITDA = 74/23",
        "the 35% tax rate is a distractor; an analyst quoting 'x EBITDA' means EV/EBITDA",
    ),
    "mq-23": (
        (92.40 - _pv_div) * math.exp(0.031 * 0.75),
        "F = (S - PV(divs)) * e^{rT}, discrete dividends discounted continuously",
        "the alternative is carrying dividends forward rather than discounting spot",
    ),
    "mq-26": (
        (107 - 103) / (2 * 105 * 0.005),
        "effective duration = (P- - P+) / (2 * P0 * dy)",
        "standard effective-duration definition; no real alternative",
    ),
    "mq-59": (
        _fwd23,
        "f(2,3) = (1+s3)^3 / (1+s2)^2 - 1, the 1y rate starting 2y forward",
        "compound (not simple) forward; VerifiQuant returned f(1,2) = 2.344 instead",
    ),
    "mq-64": (
        round(225000 * (1.063 * (1 - 0.0105)) ** 18),
        "FV = P0 * ((1+r)(1-f))^n, fee charged after interest each year-end",
        "the prompt fixes the fee timing explicitly, so this is close to mechanical",
    ),
    "mq-74": (
        19800,
        "gift dual-basis rule: sale price < FMV < donor basis, so loss basis = FMV",
        "if the question meant basis for GAIN it would be the donor's 24500",
    ),
}


def gold_table() -> dict[str, float]:
    g = {k: v[0] for k, v in MECHANICAL.items()}
    g.update({k: v[0] for k, v in CONVENTION.items()})
    return g


def main() -> None:
    gt = {json.loads(l)["id"]: json.loads(l)["label"]
          for l in open(MQ / "ground_truth.jsonl")}
    answerable = sorted(i for i in gt if gt[i] == "A")
    gold = gold_table()

    missing = [i for i in answerable if i not in gold]
    extra = [i for i in gold if i not in answerable]
    print(f"human-ANSWER items: {len(answerable)}   gold entries: {len(gold)}")
    if missing:
        print(f"  MISSING gold for: {missing}")
    if extra:
        print(f"  gold for non-ANSWER items (drop these): {extra}")

    print("\n-- mechanical --")
    for i in sorted(MECHANICAL):
        v, why = MECHANICAL[i]
        print(f"  {i}  {v:>14.4f}   {why}")
    print("\n-- convention-dependent (please adjudicate) --")
    for i in sorted(CONVENTION):
        v, why, note = CONVENTION[i]
        print(f"  {i}  {v:>14.4f}   {why}\n{'':>21}note: {note}")

    out = MQ / "answer_gold.json"
    out.write_text(json.dumps(
        {i: {"gold": gold[i],
             "formula": (MECHANICAL.get(i) or CONVENTION.get(i))[1],
             "convention_dependent": i in CONVENTION}
         for i in answerable}, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
