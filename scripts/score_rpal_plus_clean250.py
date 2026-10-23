"""
score_rpal_plus_clean250.py
---------------------------
Score the R-PAL+ (prompted) run on the clean 250-question set, using the *same*
scoring functions as scripts/analyze_governance_eval.py so the numbers are directly
comparable to plain R-PAL and VerifiQuant in Panel A / Figure 2.

Usage:
    PYTHONPATH=. python3 scripts/score_rpal_plus_clean250.py \
        [--run results/governance_eval/prompted_clean250_run1.jsonl]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_governance_eval import (  # noqa: E402
    load_jsonl, clean_labels_from_rpal, clean_summary,
)

DEFAULT_RUN = ROOT / "results/governance_eval/prompted_clean250_run1.jsonl"
QUESTIONS = ROOT / "results/paper_v2_250/questions_250.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--questions", type=Path, default=QUESTIONS)
    args = ap.parse_args()
    # Accept relative paths: resolve against cwd so .relative_to(ROOT) below is safe.
    args.run = args.run.resolve()
    args.questions = args.questions.resolve()

    clean_rows = load_jsonl(args.questions)
    questions = {str(r["question_id"]): r for r in clean_rows}
    rows = load_jsonl(args.run)

    labels = clean_labels_from_rpal(rows, questions, use_stored_verdict=False)
    summary = clean_summary(labels)
    summary["run"] = str(args.run.relative_to(ROOT))
    summary["scored_rows"] = len(rows)

    missing = set(questions) - set(labels)
    if missing:
        print(f"WARNING: {len(missing)} of {len(questions)} questions have no scored row "
              f"(run incomplete?): e.g. {sorted(missing)[:5]}", file=sys.stderr)

    print(json.dumps(summary, indent=2))
    print("\n-- for Figure 2 / Panel A --")
    print(f"  clean coverage      : {100*summary['coverage']:.1f}%")
    print(f"  clean accuracy      : {100*summary['accuracy']:.1f}%")
    print(f"  clean SWR           : {100*summary['swr']:.1f}%")
    print(f"  selective accuracy  : {100*summary['selective_accuracy']:.1f}%")

    out = args.run.with_name(args.run.stem + "_clean_scored.json")
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
