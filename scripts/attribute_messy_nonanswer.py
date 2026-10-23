#!/usr/bin/env python3
"""
attribute_messy_nonanswer.py
----------------------------
Does the 250-card library hold an applicable contract for each human-set item that is not
labeled ANSWER? Completes the item-level attribution of attribute_messy_shortfalls.py, which
covered the 24 ANSWER items that stop at N (advisor's follow-up B, 2026-10-17).

Same criterion: a card is APPLICABLE when its declared purpose covers the asked quantity and its
code computes it from the inputs the query gives. For REQUEST-INFO items, which omit one input by
design, the test is whether the card would compute the answer once that input is supplied. A card
whose arithmetic coincides but whose declared purpose is another quantity is a near neighbor, not
applicable. OUT-OF-SCOPE items ask for a judgment, not a number, so no contract answers them.

The verdicts were proposed on 2026-10-17 by reading the top-16 retrieval candidates and a keyword
search over all 250 card names and descriptions; they await the first author's confirmation.
Writes results/messy_query_pilot/nonanswer_attribution.csv.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MQ = ROOT / "results/messy_query_pilot"
PCT = "fic_article_687"  # Base Effect: generic percent change between a current and a previous value

# id -> (asked, verdict, applicable card, note); verdict: present | absent | not_a_computation
VERDICTS = {
    # ── REQUEST-INFO, stop at N ──────────────────────────────────────────────────────
    "mq-07": ("difference of two rollforward lines", "absent", "", "elementary difference"),
    "mq-10": ("aggregate intrinsic value of outstanding options", "absent", "",
              "near neighbor: intrinsic-value cards 1898 and 2824 price one option and take no option count"),
    "mq-13": ("sum of three year-end balances", "absent", "", "elementary sum"),
    "mq-16": ("quick ratio", "absent", "", "no quick-ratio card"),
    "mq-24": ("allowance as a share of gross receivables", "absent", "", "part / whole share"),
    "mq-29": ("change in EBITDA margin, in points", "absent", "", "margin then difference; Profit Margin card is net margin"),
    "mq-35": ("ending minus beginning balance", "absent", "", "elementary difference"),
    "mq-37": ("enterprise value", "absent", "", "no enterprise-value card"),
    "mq-44": ("GDP by the expenditure approach", "absent", "", "GDP card 1167 computes a growth rate, not the level"),
    "mq-55": ("total withdrawals under the 4% rule", "absent", "", "no safe-withdrawal card; Rule 72(t) is a different payout"),
    "mq-57": ("segment net revenue from an annual report", "absent", "", "a lookup, not a formula"),
    "mq-66": ("year with the largest provision", "absent", "", "elementary maximum"),
    "mq-68": ("growth in restructuring costs", "present", PCT, "percent change once the earlier amount is supplied"),
    # ── CLARIFY, stop at N ───────────────────────────────────────────────────────────
    "mq-14": ("'margin' from leased versus owned costs", "absent", "", "ratio of two totals; Profit Margin card is net margin"),
    "mq-28": ("10y key rate duration from +1bp bumps", "absent", "",
              "Key Rate Duration card 503 hard-codes a 1% shift and needs both bumps; same verdict as mq-26"),
    "mq-65": ("Europe's share of the store count", "absent", "", "part / whole share"),
    "mq-71": ("Intel's return relative to the index", "absent", "", "ratio of two cumulative values"),
    # ── OUT-OF-SCOPE, stop at N ──────────────────────────────────────────────────────
    **{i: ("a judgment or recommendation", "not_a_computation", "", "")
       for i in ("mq-12", "mq-15", "mq-21", "mq-42", "mq-45", "mq-47", "mq-48", "mq-50", "mq-51", "mq-63", "mq-76")},
    # ── REQUEST-INFO and CLARIFY where VerifiQuant selected a contract ─────────────────
    "mq-03": ("present value of five cash flows", "present", "fic_article_247", "selected card is applicable"),
    "mq-17": ("one-year holding-period return with coupon", "present", "fic_article_477", "selected card is applicable"),
    "mq-25": ("percentage drop from high to low price", "present", PCT, "selected Q/Q card 2532 is a near neighbor"),
    "mq-41": ("year-over-year growth of a balance", "present", PCT, "selected holding-period card 1063 is a near neighbor"),
    "mq-43": ("liabilities as a share of liabilities plus equity", "absent", "", "debt-to-capitalization card 2120 is a different ratio; same verdict as mq-70"),
    "mq-52": ("real return by the exact Fisher relation", "present", "fic_article_2290", "after-tax real return with a zero tax rate"),
    "mq-54": ("money-market yield on a 360-day basis", "present", "fic_article_950", "selected commercial-paper card 123 is a near neighbor"),
    "mq-01": ("growth of a tax-benefit balance", "present", PCT, "selected card 2659 is unrelated"),
    "mq-06": ("growth of a liability balance", "present", PCT, "selected annualized-return card 2331 is a near neighbor"),
    "mq-36": ("level annuity payment", "present", "fic_article_12", "selected card is applicable"),
    "mq-39": ("growth of a tax-benefit figure", "present", PCT, "selected card 313 is unrelated"),
    "mq-58": ("change in a rate-sensitivity figure", "present", PCT, "one reading of an ambiguous ask; selected EVE card 2738 is unrelated"),
    "mq-80": ("return on a note with monthly compounding", "present", "fic_article_431", "selected YTM card 30 is a near neighbor"),
}


def main() -> None:
    lab = {json.loads(l)["id"]: json.loads(l)["label"] for l in (MQ / "ground_truth.jsonl").open()}
    vq = {json.loads(l)["case_id"]: json.loads(l) for l in (MQ / "agreement/runs/vq_k1.jsonl").open()}
    rp = {json.loads(l)["case_id"]: json.loads(l) for l in (MQ / "agreement/runs/rpal_plus.jsonl").open()}

    def vq_exit(r):
        return "number" if r.get("status") == "success" else str(r.get("diagnostic_type"))

    def rp_exit(r):
        a = r.get("assessment")
        if isinstance(a, str):
            try:
                a = eval(a)  # noqa: S307 (stored as a Python repr)
            except Exception:
                a = None
        return "number" if str(r.get("output")) not in ("None", "") else str((a or {}).get("diagnostic"))

    expected_n = {i for i, l in lab.items() if l in ("R", "C", "O") and vq_exit(vq[i]) == "N"}
    expected_sel = {i for i, l in lab.items() if l in ("R", "C") and vq_exit(vq[i]) != "N"}
    missing = (expected_n | expected_sel) ^ set(VERDICTS)
    if missing:
        raise SystemExit(f"verdict table and item set disagree on: {sorted(missing)}")
    rows = [{"id": i, "label": lab[i], "vq_exit": vq_exit(vq[i]), "rpal_plus_exit": rp_exit(rp[i]),
             "asked": VERDICTS[i][0], "verdict": VERDICTS[i][1], "applicable_card": VERDICTS[i][2], "note": VERDICTS[i][3]}
            for i in sorted(VERDICTS)]
    out = MQ / "nonanswer_attribution.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for label, name in (("R", "REQUEST-INFO"), ("C", "CLARIFY")):
        sel = [r for r in rows if r["label"] == label and r["verdict"] == "present"]
        f = lambda key, val: sum(r[key] == val for r in sel)  # noqa: E731
        print(f"{name} with an applicable contract: {len(sel)} | VQ F {f('vq_exit', 'F')}, number {f('vq_exit', 'number')}, "
              f"M {f('vq_exit', 'M')}, N {f('vq_exit', 'N')} | R-PAL+ F {f('rpal_plus_exit', 'F')}, number {f('rpal_plus_exit', 'number')}, M {f('rpal_plus_exit', 'M')}")
    n_rows = [r for r in rows if r["vq_exit"] == "N"]
    wrong = sum(r["verdict"] == "present" for r in n_rows)
    print(f"non-ANSWER items stopping at N: {len(n_rows)}, of which an applicable contract exists for {wrong}")
    print("wrote", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
