#!/usr/bin/env python3
"""OOD formula-family evaluation (ICAIF reviewer attack #1, setting c).

Question is OOD if its formula family (function_id) has NO card in the contract library.
We reuse the natural split: the paper_v1 53-card library vs the 250Q families (199 are absent).

Thesis test: when the library LACKS the family, does VerifiQuant ABSTAIN (good — honest "no
verified contract") or FORCE-FIT a wrong card and emit a confident verified-looking number
(bad — silent-wrong)? We also run the ID (covered) families as a sanity control (VQ should
still answer those).

This is single-shot (K=1) — abstention happens on the first pass; multi-turn oracle cannot
conjure a missing card. Config is recorded so a later K-variation is a cheap re-run.

Usage:
  PYTHONPATH=. GEMINI_API_KEY=… python3 scripts/run_ood_family_eval.py \
      --limit-ood 60 --limit-id 20 --seed 0 --out-dir results/ood_eval

Camera-ready reruns (E13, E15) add ``--top-k 4`` (the July runs used the default 3),
``--out-name``/``--resume`` and an API guard (scripts/api_guard.py): API-level failures
are retried and, if they persist, stop the run (exit 3) instead of being recorded as an
abstention. Every other exception is still recorded as an error, as before.
"""
from __future__ import annotations
import argparse, json, os, random, sqlite3, sys, time
from datetime import datetime

from api_guard import EXIT_API_EXHAUSTED, ApiExhausted, client_with_timeout, diagnose_with_api_retry

from verifiquant.card_store import SQLAlchemyArtifactStore  # noqa: F401 (import sanity)
from verifiquant.pipeline.run_error_classification_pipeline import (
    ErrorClassificationAPI, create_genai_client_from_env,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB_DB = os.path.join(ROOT, "results/paper_v1/fic/cards.db")
Q_SRC = os.path.join(ROOT, "results/paper_v2_250/questions_250.jsonl")

ABSTAIN_TYPES = {"M", "N"}                         # honest refusal on OOD
ABSTAIN_STATUSES = {"refusal", "error", "needs_clarification", "alert"}


def _num(x):
    try:
        return float(str(x).replace(",", "").replace("%", "").strip())
    except Exception:
        return None


def _correct(out_val, gold, tol=0.01):
    a, b = _num(out_val), _num(gold)
    if a is None or b is None:
        return False
    return abs(a - b) <= max(1e-6, abs(b) * tol)


def classify(rep, gold):
    """Return one of: abstain | correct | silent_wrong."""
    status = str(rep.get("status", ""))
    dtype = str(rep.get("diagnostic_type", "") or "")[:1].upper()
    if status == "success":
        return "correct" if _correct(rep.get("output_value"), gold) else "silent_wrong"
    # non-success: an abstention/refusal (good behaviour on OOD)
    if dtype in ABSTAIN_TYPES or status in ABSTAIN_STATUSES:
        return "abstain"
    return "abstain"  # any non-committed outcome = did not force an answer


# Faithful CoT single-shot baseline: chain-of-thought, ALLOWED to abstain (so we fairly measure
# whether a plain CoT solver refuses on OOD families or hallucinates a confident number).
BASELINE_COT_PROMPT = (
    "You are a financial reasoning assistant. Think step by step, then give the final numeric answer.\n"
    "If the problem genuinely cannot be solved from the given information, or requires a formula you\n"
    "are not confident about, respond with CANNOT SOLVE instead of guessing.\n"
    "End with exactly one line: 'Final answer: <number>' or 'Final answer: CANNOT SOLVE'."
)


def run_baseline(client, model, q):
    prompt = (BASELINE_COT_PROMPT
              + f"\n\nQuestion:\n{q.get('question','')}\n\nContext:\n{q.get('context','') or '(none)'}")
    resp = client.models.generate_content(model=model, contents=prompt)
    return resp.text or ""


def classify_baseline(answer_text, gold):
    """abstain | correct | silent_wrong for the CoT baseline."""
    t = str(answer_text or "")
    tail = t.split("Final answer:")[-1] if "Final answer:" in t else t
    if "CANNOT SOLVE" in t.upper():
        return "abstain", None
    val = _num(tail.strip().splitlines()[0] if tail.strip() else "")
    if val is None:
        return "abstain", None
    return ("correct" if _correct(val, gold) else "silent_wrong"), val


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lib-db", default=LIB_DB)
    ap.add_argument("--questions", default=Q_SRC)
    ap.add_argument("--limit-ood", type=int, default=60)
    ap.add_argument("--limit-id", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results/ood_eval"))
    ap.add_argument("--with-baseline", action="store_true", default=True,
                    help="also run a CoT single-shot baseline on the same questions")
    ap.add_argument("--no-baseline", dest="with_baseline", action="store_false")
    ap.add_argument("--out-name", default="", help="fixed file stem instead of a timestamp")
    ap.add_argument("--resume", action="store_true", help="skip case_ids already in --out-name")
    ap.add_argument("--http-timeout-s", type=int, default=120)
    args = ap.parse_args()

    lib = {r[0].replace("fic_article_", "")
           for r in sqlite3.connect(args.lib_db).execute("SELECT fic_id FROM core_cards")}
    qs = [json.loads(l) for l in open(args.questions) if l.strip()]
    def fam(q): return str(q.get("function_id", "")).replace("article-", "")
    ood = [q for q in qs if fam(q) not in lib]
    idd = [q for q in qs if fam(q) in lib]

    rng = random.Random(args.seed)
    rng.shuffle(ood); rng.shuffle(idd)
    ood = ood[: args.limit_ood]
    idd = idd[: args.limit_id]

    client = client_with_timeout(args.http_timeout_s)
    api = ErrorClassificationAPI.from_db(
        db_url=f"sqlite:///{args.lib_db}", client=client,
        selector_model=args.model, extractor_model=args.model, judge_model=args.model,
        top_k=args.top_k,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    ts = args.out_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_jsonl = os.path.join(args.out_dir, f"ood_results_{ts}.jsonl")
    out_summary = os.path.join(args.out_dir, f"ood_summary_{ts}.json")
    done = set()
    if args.resume and os.path.exists(out_jsonl):
        done = {json.loads(l)["case_id"] for l in open(out_jsonl) if l.strip()}

    blank = lambda: {"abstain": 0, "correct": 0, "silent_wrong": 0, "n": 0}
    counts = {"OOD": blank(), "ID": blank()}
    bcounts = {"OOD": blank(), "ID": blank()}  # CoT baseline
    t0 = time.time()
    with open(out_jsonl, "a" if args.resume else "w") as fo:
        for split, items in (("OOD", ood), ("ID", idd)):
            for i, q in enumerate(items, 1):
                row = {"case_id": q.get("question_id") or fam(q),
                       "question": q.get("question", ""), "context": q.get("context", "")}
                if row["case_id"] in done:
                    continue
                gold = q.get("ground_truth")
                try:
                    rep = diagnose_with_api_retry(
                        lambda: api.diagnose_row(row, top_k=args.top_k, debug_sanity=False),
                        label=f"{split}/{row['case_id']}")
                except ApiExhausted as exc:
                    print(f"API_EXHAUSTED {exc}", flush=True)
                    sys.exit(EXIT_API_EXHAUSTED)
                except Exception as exc:
                    rep = {"status": "error", "diagnostic_type": "Unknown", "message": str(exc)}
                verdict = classify(rep, gold)
                counts[split][verdict] += 1
                counts[split]["n"] += 1

                rec = {
                    "case_id": row["case_id"], "split": split, "family": fam(q), "title": q.get("article_title"),
                    "gold": gold, "verdict": verdict,
                    "status": rep.get("status"), "diagnostic_type": rep.get("diagnostic_type"),
                    "chosen_fic": rep.get("fic_id"), "output_value": rep.get("output_value"),
                }
                bverdict = bval = None
                if args.with_baseline:
                    try:
                        bans = run_baseline(client, args.model, q)
                        bverdict, bval = classify_baseline(bans, gold)
                    except Exception as exc:
                        bverdict, bval = "error", str(exc)[:80]
                    if bverdict in bcounts[split]:
                        bcounts[split][bverdict] += 1
                    bcounts[split]["n"] += 1
                    rec["baseline_verdict"] = bverdict
                    rec["baseline_value"] = bval

                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fo.flush()  # incremental so long runs are monitorable / resumable-by-inspection
                print(f"[{split} {i}/{len(items)}] {fam(q):9} VQ={verdict:12} "
                      f"CoT={bverdict} fic={rep.get('fic_id')}")

    # Recount from the file so a resumed run summarises every row, not only this session's.
    counts = {"OOD": blank(), "ID": blank()}
    bcounts = {"OOD": blank(), "ID": blank()}
    for line in open(out_jsonl):
        if not line.strip():
            continue
        r = json.loads(line)
        counts[r["split"]][r["verdict"]] += 1
        counts[r["split"]]["n"] += 1
        if args.with_baseline and "baseline_verdict" in r:
            if r["baseline_verdict"] in bcounts[r["split"]]:
                bcounts[r["split"]][r["baseline_verdict"]] += 1
            bcounts[r["split"]]["n"] += 1

    def rate(c, split, k):
        n = c[split]["n"] or 1
        return c[split][k] / n

    summary = {
        "ran_at": ts, "elapsed_seconds": round(time.time() - t0, 1),
        "config": {
            "lib_db": args.lib_db, "questions": args.questions,
            "split_rule": "family (function_id) present in library cards.db",
            "limit_ood": args.limit_ood, "limit_id": args.limit_id, "seed": args.seed,
            "top_k": args.top_k, "model": args.model, "k_rounds": 1,
            "total_ood_available": len(ood), "total_id_available": len(idd),
        },
        "VQ": {
            "OOD": {**counts["OOD"],
                    "abstain_rate": rate(counts, "OOD", "abstain"),
                    "silent_wrong_rate": rate(counts, "OOD", "silent_wrong"),
                    "force_fit_correct_rate": rate(counts, "OOD", "correct")},
            "ID": {**counts["ID"],
                   "accuracy": rate(counts, "ID", "correct"),
                   "silent_wrong_rate": rate(counts, "ID", "silent_wrong"),
                   "abstain_rate": rate(counts, "ID", "abstain")},
        },
        "CoT_baseline": {
            "OOD": {**bcounts["OOD"],
                    "abstain_rate": rate(bcounts, "OOD", "abstain"),
                    "silent_wrong_rate": rate(bcounts, "OOD", "silent_wrong"),
                    "correct_rate": rate(bcounts, "OOD", "correct")},
            "ID": {**bcounts["ID"],
                   "accuracy": rate(bcounts, "ID", "correct"),
                   "silent_wrong_rate": rate(bcounts, "ID", "silent_wrong"),
                   "abstain_rate": rate(bcounts, "ID", "abstain")},
        } if args.with_baseline else None,
    }
    json.dump(summary, open(out_summary, "w"), indent=2, ensure_ascii=False)
    print("\n=== SUMMARY ===")
    print(json.dumps({k: summary[k] for k in ("VQ", "CoT_baseline")}, indent=2))
    print("results:", out_jsonl)
    print("summary:", out_summary)


if __name__ == "__main__":
    main()
