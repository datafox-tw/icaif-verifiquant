#!/usr/bin/env python3
"""Self-consistency@k baseline (Wave 1-D).

k sampled CoT answers per question (temperature 1.0); majority vote by numeric
clustering; vote share = confidence. Thresholding the share yields REAL
risk-coverage operating points for the baseline — the realizable counterpart to
the perfect-rejector upper bound in the paper's Figure (fig:riskcov).

Usage:
  PYTHONPATH=. GEMINI_API_KEY=... .venv/bin/python scripts/run_sc_baseline.py \
      [--k 5] [--model gemini-2.5-flash] [--limit 0]
"""
from __future__ import annotations
import argparse, json, os, time
from datetime import datetime

from verifiquant.pipeline.run_error_classification_pipeline import (
    create_genai_client_from_env,
)
try:
    from google.genai import types as genai_types
except Exception:  # pragma: no cover
    genai_types = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q250 = os.path.join(ROOT, "results/paper_v2_250/questions_250.jsonl")

PROMPT = ("You are a financial reasoning assistant. Think step by step, then give the final "
          "numeric answer.\nEnd with exactly one line: 'Final answer: <number>'.\n\n"
          "Question:\n{question}\n\nContext:\n{context}\n")


def _num(x):
    try:
        return float(str(x).replace(",", "").replace("%", "").replace("$", "").strip())
    except (TypeError, ValueError):
        return None


def parse_answer(text: str):
    tail = text.split("Final answer:")[-1] if "Final answer:" in (text or "") else (text or "")
    for tok in tail.strip().splitlines()[:1]:
        return _num(tok)
    return None


def cluster_vote(vals):
    """Cluster numeric answers within 1% tolerance; return (winner, share, n_valid)."""
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, 0.0, 0
    clusters: list[list[float]] = []
    for v in vals:
        for c in clusters:
            if abs(v - c[0]) <= max(1e-9, abs(c[0]) * 0.01):
                c.append(v); break
        else:
            clusters.append([v])
    best = max(clusters, key=len)
    return best[0], len(best) / len(vals), len(vals)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--questions", default=Q250)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results/sc_baseline"))
    args = ap.parse_args()

    qs = [json.loads(l) for l in open(args.questions) if l.strip()]
    if args.limit:
        qs = qs[: args.limit]
    client = create_genai_client_from_env()
    cfg = genai_types.GenerateContentConfig(temperature=1.0) if genai_types else None
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"sc{args.k}_{args.model.replace('.', '').replace('-', '_')}"
    out_path = os.path.join(args.out_dir, f"{tag}.jsonl")
    t0 = time.time()
    rows_out = []

    with open(out_path, "w") as fo:
        for i, q in enumerate(qs, 1):
            prompt = PROMPT.format(question=q.get("question", ""), context=q.get("context", "") or "(none)")
            answers = []
            for _ in range(args.k):
                try:
                    r = client.models.generate_content(model=args.model, contents=prompt, config=cfg)
                    answers.append(parse_answer(r.text or ""))
                except Exception:
                    answers.append(None)
            winner, share, n_valid = cluster_vote(answers)
            gold = _num(q.get("ground_truth"))
            correct = (winner is not None and gold is not None
                       and abs(winner - gold) <= max(1e-6, abs(gold) * 0.01))
            rec = {"case_id": q.get("question_id"), "gold": q.get("ground_truth"),
                   "answers": answers, "winner": winner, "share": share,
                   "n_valid": n_valid, "correct": correct}
            rows_out.append(rec)
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fo.flush()
            print(f"[{tag} {i}/{len(qs)}] {q.get('question_id')} share={share:.2f} "
                  f"{'OK' if correct else 'wrong' if winner is not None else 'no-answer'}")

    # risk-coverage points by vote-share threshold
    points = []
    for thr in (0.0, 0.4, 0.6, 0.8, 1.0):
        answered = [r for r in rows_out if r["winner"] is not None and r["share"] >= thr]
        cov = len(answered) / len(rows_out) if rows_out else 0
        sel = (sum(r["correct"] for r in answered) / len(answered)) if answered else None
        sw = sum(1 for r in answered if not r["correct"]) / len(rows_out) if rows_out else 0
        points.append({"threshold": thr, "coverage": cov, "selective_acc": sel, "swr": sw})
    summary = {"tag": tag, "model": args.model, "k": args.k, "n": len(rows_out),
               "risk_coverage_points": points,
               "elapsed_s": round(time.time() - t0, 1),
               "ran_at": datetime.now().isoformat(timespec="seconds")}
    json.dump(summary, open(os.path.join(args.out_dir, f"{tag}_summary.json"), "w"), indent=2)
    print(json.dumps(summary["risk_coverage_points"], indent=1))


if __name__ == "__main__":
    main()
