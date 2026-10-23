from __future__ import annotations

import json
import re
from typing import Dict, List, Tuple


TAXONOMY: Dict[str, List[str]] = {
    "accounting_and_financial_reporting": [
        "balance_sheets",
        "cash_flow_statements",
        "depreciation_and_amortization",
        "income_statements",
    ],
    "corporate_finance": [
        "capital_structure",
        "div_policies",
        "eps",
        "misc",
        "wacc",
    ],
    "crypto_finance": [
        "defi",
        "global_events_impact",
        "regulations",
        "sentiments",
        "tokenomics",
        "transaction_analysis",
        "whale_detection",
    ],
    "finance_regulation": [
        "aml",
        "basel",
        "compliance",
        "kyc",
        "security",
    ],
    "financial_markets": [
        "bond_pricing",
        "forex_trading",
        "option_pricing",
        "stock_market_analysis",
    ],
    "financial_ratios": [
        "effratio",
        "levratio",
        "liqratio",
        "profitratio",
    ],
    "fintech": [
        "digital_banking",
        "payment_technologies",
        "robo_advisors",
    ],
    "investment_analysis": [
        "ci",
        "irr",
        "npv",
        "rar",
        "roi",
    ],
    "mergers_and_acquisitions": [
        "deal_structure",
        "due_diligence",
        "post_merger_integration",
        "synergies_and_cost_savings",
        "valuation_methods",
    ],
    "personal_finance": [
        "budgeting",
        "loanrepay",
        "personalinvest",
        "saveretire",
        "taxcalc",
    ],
    "risk_management": [
        "regulatory_compliance",
        "risk_appetite",
        "risk_metrics",
        "scenario_planning",
        "sensitivity_analysis",
        "stress_testing",
        "var",
    ],
    "sustainable_finance": [
        "carbon_credits",
        "esg_investing",
        "green_bonds",
        "social_impact_investing",
        "sustainability_reporting",
    ],
    "others":["others"],
}


def taxonomy_json(indent: int = 2) -> str:
    return json.dumps(TAXONOMY, ensure_ascii=False, indent=indent, sort_keys=True)


def is_valid_domain_topic(domain: str, topic: str) -> bool:
    return domain in TAXONOMY and topic in TAXONOMY[domain]


def normalize_taxonomy_label(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", str(value or "").strip().lower()).strip("_")
    return cleaned


def is_valid_domain(domain: str) -> bool:
    return normalize_taxonomy_label(domain) in TAXONOMY


def is_known_topic(domain: str, topic: str) -> bool:
    d = normalize_taxonomy_label(domain)
    t = normalize_taxonomy_label(topic)
    return d in TAXONOMY and t in TAXONOMY[d]


def validate_domain_topic(
    domain: str,
    topic: str,
    *,
    allow_new_topic: bool = True,
) -> Tuple[str, str, bool]:
    """
    Validate taxonomy with the  policy:
    - domain must come from existing taxonomy domains
    - topic can be a known topic or a new extension under that domain
    """
    d = normalize_taxonomy_label(domain)
    t = normalize_taxonomy_label(topic)
    if not d or d not in TAXONOMY:
        raise ValueError(f"Invalid domain '{domain}'. Domain must be one of taxonomy domains.")
    if not t:
        raise ValueError("Topic cannot be empty.")

    is_new_topic = t not in TAXONOMY[d]
    if is_new_topic and not allow_new_topic:
        raise ValueError(
            f"Invalid topic '{topic}' for domain '{d}'. Topic must come from taxonomy when allow_new_topic=False."
        )
    return d, t, is_new_topic
