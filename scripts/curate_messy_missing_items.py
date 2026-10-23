#!/usr/bin/env python3
"""Apply the audited v3 partial-context replacements to the 18 REQUEST-INFO items.

This is an offline, deterministic fallback for the external author call.  It preserves
all IDs, personas, source topics, and the 62 non-missing items.  Each replacement has
at least two visible numeric values and exactly one documented indispensable omission.
The omission audit is written only to provenance, never to the labeling sheet or solver
input.
"""
from __future__ import annotations

import json
from pathlib import Path

from build_messy_query_pilot import _validate_missing, _write_dataset


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/messy_query_pilot"


REPLACEMENTS = {
    "mq-03": {
        "question": "Can you discount these five year-end project cash flows and give me their present value, rounded to the nearest dollar?",
        "context": "Projected cash flows are $180,000 in Year 1, $215,000 in Year 2, $240,000 in Year 3, $265,000 in Year 4, and $300,000 in Year 5. Each cash flow arrives at year-end.",
        "omitted_input": "annual discount rate",
        "why_required": "Present value cannot be determined from the cash-flow amounts and timing without a discount rate.",
    },
    "mq-07": {
        "question": "What is the difference, in millions, between deferral of revenue and recognition of deferred revenue for this fiscal year? Nearest whole million is fine.",
        "context": "The contract-liability rollforward for the year ended June 30, 2019 reports deferral of revenue of $148 million, foreign-currency adjustments of $3 million, and ending contract liabilities of $612 million.",
        "omitted_input": "recognition of deferred revenue",
        "why_required": "The requested difference directly requires the recognition amount, which cannot be inferred without the beginning balance and all other rollforward components.",
    },
    "mq-10": {
        "question": "For the shareholder-approved plan, what is the aggregate intrinsic value of the outstanding in-the-money options at year-end? Please report it in millions to three decimals.",
        "context": "The plan has 1.8 million outstanding in-the-money options, and the company's year-end share price was $42.60 per share.",
        "omitted_input": "weighted-average exercise price",
        "why_required": "Aggregate intrinsic value requires the spread between market price and exercise price multiplied by the option count.",
    },
    "mq-13": {
        "question": "Can you add up the ending unrecognized-tax-benefit balances for 2017, 2018, and 2019 and give me the total in millions?",
        "context": "The footnote reports ending balances of $84 million for 2017 and $91 million for 2018. It separately reports $6 million of settlements during 2019.",
        "omitted_input": "2019 ending balance of unrecognized tax benefits",
        "why_required": "A three-year sum needs the 2019 ending balance; the settlements line is not a substitute for that balance.",
    },
    "mq-16": {
        "question": "Can you compute the quick ratio as a percentage using cash, marketable securities, and accounts receivable?",
        "context": "At quarter-end, cash and cash equivalents were $86 million, marketable securities were $34 million, and accounts receivable were $120 million.",
        "omitted_input": "current liabilities",
        "why_required": "The quick-assets numerator is available, but the ratio cannot be computed without current liabilities as the denominator.",
    },
    "mq-17": {
        "question": "What is the one-year roll-down holding-period return, including coupon income, if I buy this bond now and sell it at the expected price in a year? Give me the percentage to two decimals.",
        "context": "The bond can be purchased for $98.40 today and is expected to sell for $100.10 after a one-year holding period if rates are unchanged.",
        "omitted_input": "coupon cash received during the holding period",
        "why_required": "Holding-period return includes both the price change and coupon income, so prices alone are insufficient.",
    },
    "mq-24": {
        "question": "For fiscal 2014, what was the allowance for doubtful accounts as a percentage of gross accounts receivable? Report to three decimals.",
        "context": "The 2014 allowance for doubtful accounts was $18.6 million. The same balance-sheet note reports total current assets of $720 million.",
        "omitted_input": "gross accounts receivable for 2014",
        "why_required": "The percentage uses gross accounts receivable as its denominator; total current assets cannot replace it.",
    },
    "mq-25": {
        "question": "For the quarter ended March 31, 2017, what was the percentage drop from the high sale price to the low sale price? Round to three decimals.",
        "context": "The quarterly stock-price table lists a high sale price of $47.80 and trading volume of 2.3 million shares for that quarter.",
        "omitted_input": "low sale price for the quarter",
        "why_required": "The percentage drop requires both the high and low prices; trading volume does not determine the low.",
    },
    "mq-29": {
        "question": "How many percentage points did adjusted EBITDA margin increase or decrease from 2018 to 2019? Round to one decimal place.",
        "context": "Adjusted EBITDA margin was 14.7% in 2018, and 2019 revenue was $2.1 billion.",
        "omitted_input": "2019 adjusted EBITDA margin",
        "why_required": "A margin change needs the margin in both years; revenue alone does not determine the 2019 margin.",
    },
    "mq-35": {
        "question": "What was the net change in unrecognized tax benefits during 2008, measured as ending balance minus beginning balance? Nearest whole million, please.",
        "context": "The beginning balance on January 1, 2008 was $61 million, and additions for current-year tax positions were $9 million during 2008.",
        "omitted_input": "ending balance on December 31, 2008",
        "why_required": "The requested subtraction directly requires the ending balance; one rollforward component cannot determine it.",
    },
    "mq-37": {
        "question": "Using market cap plus debt, preferred stock, and minority interest, minus cash, what is the company's enterprise value? Round to the nearest $1 million.",
        "context": "Market capitalization is $8.6 billion, total debt is $2.1 billion, cash is $740 million, and preferred stock is $120 million.",
        "omitted_input": "minority interest",
        "why_required": "The stated enterprise-value definition includes minority interest, which is not derivable from the other capital-structure figures.",
    },
    "mq-41": {
        "question": "What was the 2010 growth rate in the balance held in money market funds compared with 2009? Give it as a percentage to three decimals.",
        "context": "Money market funds totaled $42 million in 2009. Total cash and cash equivalents across all instruments were $110 million in 2010.",
        "omitted_input": "2010 balance in money market funds",
        "why_required": "The growth rate needs the same money-market-fund balance in both years; total cash equivalents is a broader measure.",
    },
    "mq-43": {
        "question": "What percentage of total liabilities plus shareholders' equity is represented by total liabilities? Report to three decimals.",
        "context": "The balance sheet reports total liabilities of $680 million, cash of $145 million, and current debt of $90 million.",
        "omitted_input": "total shareholders' equity",
        "why_required": "The common-size denominator is liabilities plus equity, and equity cannot be recovered from the provided liability components.",
    },
    "mq-44": {
        "question": "Using the expenditure approach, what was GDP last year in USD trillions? Round to two decimals.",
        "context": "Last year's consumption was $12.4 trillion, investment was $3.1 trillion, government spending was $4.0 trillion, and exports were $2.6 trillion.",
        "omitted_input": "imports",
        "why_required": "The expenditure identity subtracts imports, so GDP cannot be computed from C, I, G, and exports alone.",
    },
    "mq-52": {
        "question": "Using the exact Fisher relation, can you calculate the mutual fund's real return for this 12-month period and give the percentage to two decimals?",
        "context": "The reported nominal return, already net of fund expenses, was 8.4% for the 12-month period. The fund's expense ratio was 0.6%.",
        "omitted_input": "inflation rate for the period",
        "why_required": "Real return adjusts the nominal return for inflation; the expense ratio does not measure inflation.",
    },
    "mq-54": {
        "question": "Using a 360-day money-market basis, what is the annualized yield on this short-term government bond? Give it as a percentage to two decimals.",
        "context": "The bond has a face value of $10,000 and was purchased for $9,750.",
        "omitted_input": "days to maturity",
        "why_required": "Annualizing the holding-period discount requires the number of days to maturity.",
    },
    "mq-55": {
        "question": "Using the 4% rule with the same inflation adjustment each year, what is the total amount withdrawn over the next five years? Round to the nearest dollar.",
        "context": "The starting portfolio is $1.2 million, so the Year 1 withdrawal is $48,000. The planning horizon is 5 years.",
        "omitted_input": "annual inflation adjustment rate",
        "why_required": "Years 2 through 5 depend on the annual inflation rate, which cannot be inferred from the initial withdrawal.",
    },
    "mq-66": {
        "question": "Which year from 2016 through 2018 had the largest provision for credit losses, and what was the amount in millions?",
        "context": "Provision for credit losses was $38 million in 2016 and $44 million in 2017. Net loans outstanding at the end of 2018 were $3.2 billion.",
        "omitted_input": "2018 provision for credit losses",
        "why_required": "Comparing all three years requires the 2018 provision; net loans outstanding is a different measure.",
    },
}


def main() -> None:
    rows = [json.loads(line) for line in (OUT / "messy_pilot.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    provenance = json.loads((OUT / "messy_pilot_provenance.json").read_text(encoding="utf-8"))
    prov_by_id = {row["id"]: row for row in provenance}
    actual = {row["id"] for row in rows if row.get("intended") == "missing"}
    if actual != set(REPLACEMENTS):
        raise ValueError(f"replacement IDs do not match intended=missing IDs: {actual ^ set(REPLACEMENTS)}")

    for row in rows:
        if row["id"] not in REPLACEMENTS:
            continue
        replacement = REPLACEMENTS[row["id"]]
        errors, number_count = _validate_missing(replacement)
        if errors:
            raise ValueError(f"{row['id']} failed validation: {errors}")
        row["question"] = replacement["question"]
        row["context"] = replacement["context"]
        row["author_model"] = "gpt-5.2_seed_then_codex_curated_v3"
        prov_by_id[row["id"]]["authoring_audit"] = {
            "omitted_input": replacement["omitted_input"],
            "why_required": replacement["why_required"],
            "present_numeric_count": number_count,
            "generation_attempts": 0,
            "regenerated_with_prompt": "partial_context_single_key_omission_v3",
            "method": "offline_manual_curation_after_empty_context_audit",
        }

    _write_dataset(rows, [prov_by_id[row["id"]] for row in rows], OUT)
    print(f"curated {len(REPLACEMENTS)} partial-context missing-data items -> {OUT}")


if __name__ == "__main__":
    main()
