# Drop the 5 annotators' filled label files here

One CSV per annotator, named with their name, e.g. `alice.csv`, `bob.csv`, ...
Each file = the labeling sheet with the `label` (A/R/C/O, optional `?`) and `reason`
columns filled. Extra columns (persona, question, context) are fine and ignored.

Then run:

    PYTHONPATH=. .venv/bin/python scripts/aggregate_messy_labels.py

Produces (one level up, in messy_query_pilot/):
  - ground_truth.jsonl       (majority label, >=3/5)
  - disagreement_pool.csv    (2-2-1 / no-majority items — adjudicate these by hand)
  - agreement_report.json    (Fleiss' kappa, majority rate, per-rater coverage)
