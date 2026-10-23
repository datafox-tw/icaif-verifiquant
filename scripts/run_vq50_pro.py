#!/usr/bin/env python3
"""VQ-Pro strict-binding ablation runner (Wave 1-B prep).

Runs the canonical 50Q (medium) through single-shot (K=1) VerifiQuant diagnosis with a
chosen model against the pristine 250-card library. Run twice — once with
VERIFIQUANT_STRICT_BINDING unset, once =1 — then compare per-case with
scripts/compare_vq50_runs.py. Scores are ternary + unit-normalized.

Usage:
  PYTHONPATH=. GEMINI_API_KEY=... [VERIFIQUANT_STRICT_BINDING=1] \
    .venv/bin/python scripts/run_vq50_pro.py --tag pro_on --model gemini-2.5-pro

Also the gate-ladder runner (scripts/run_gate_ladder.sh). Retrieval depth was hard-coded
to 3 until 2026-10-15; ``--top-k`` (default 3, the old behaviour) now sets it, and the
camera-ready rerun (E14) passes 4. ``--resume`` and the API guard (scripts/api_guard.py)
work as in run_ood_family_eval.py: persistent API failures stop the run (exit 3) instead
of being recorded as abstentions.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from datetime import datetime

from api_guard import EXIT_API_EXHAUSTED, ApiExhausted, client_with_timeout, diagnose_with_api_retry

from verifiquant.pipeline.run_error_classification_pipeline import (
    ErrorClassificationAPI, create_genai_client_from_env,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q50 = os.path.join(ROOT, "results/paper_v1/questions_50.jsonl")
LIB = os.path.join(ROOT, "results/libsweep/lib_100/cards.db")


def _num(x):
    try:
        return float(str(x).replace(",", "").replace("%", "").strip())
    except (TypeError, ValueError):
        return None


def verdict(rep, gold):
    if rep.get("status") != "success":
        return "abstain", False
    a, b = _num(rep.get("output_value")), _num(gold)
    if a is None or b is None:
        return "silent_wrong", False
    if abs(a - b) <= max(1e-6, abs(b) * 0.01):
        return "correct", False
    near = any(abs(a * f - b) <= max(1e-9, abs(b) * 0.01) for f in (100.0, 0.01))
    return "silent_wrong", near


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default="gemini-2.5-pro")
    ap.add_argument("--questions", default=Q50)
    ap.add_argument("--lib-db", default=LIB)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results/vq50_pro"))
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--http-timeout-s", type=int, default=120)
    args = ap.parse_args()

    qs = [json.loads(l) for l in open(args.questions) if l.strip()]
    if args.limit:
        qs = qs[: args.limit]

    api = ErrorClassificationAPI.from_db(
        db_url=f"sqlite:///{args.lib_db}", client=client_with_timeout(args.http_timeout_s),
        selector_model=args.model, extractor_model=args.model, judge_model=args.model,
        top_k=args.top_k,
    )
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.tag}.jsonl")
    flag = os.environ.get("VERIFIQUANT_STRICT_BINDING", "")
    t0 = time.time()
    counts = {"correct": 0, "silent_wrong": 0, "abstain": 0, "near_miss": 0, "strict_fired": 0}

    done = set()
    if args.resume and os.path.exists(out_path):
        done = {json.loads(l)["case_id"] for l in open(out_path) if l.strip()}
    with open(out_path, "a" if args.resume else "w") as fo:
        for i, q in enumerate(qs, 1):
            row = {"case_id": q.get("question_id") or q.get("function_id"),
                   "question": q.get("question", ""), "context": q.get("context", "")}
            if row["case_id"] in done:
                continue
            try:
                rep = diagnose_with_api_retry(
                    lambda: api.diagnose_row(row, top_k=args.top_k, debug_sanity=False),
                    label=f"{args.tag}/{row['case_id']}")
            except ApiExhausted as exc:
                print(f"API_EXHAUSTED {exc}", flush=True)
                sys.exit(EXIT_API_EXHAUSTED)
            except Exception as exc:
                rep = {"status": "error", "diagnostic_type": "Unknown", "message": str(exc)}
            v, near = verdict(rep, q.get("ground_truth"))
            counts[v] += 1
            counts["near_miss"] += bool(near)
            notes = (rep.get("extraction_trace") or {}).get("coercion_notes", []) or []
            fired = [n for n in notes if "strict-binding" in n]
            counts["strict_fired"] += bool(fired)
            fo.write(json.dumps({
                "case_id": row["case_id"], "gold": q.get("ground_truth"),
                "verdict": v, "scale_near_miss": near,
                "status": rep.get("status"), "diagnostic_type": rep.get("diagnostic_type"),
                "fic_id": rep.get("fic_id"), "output_value": rep.get("output_value"),
                "provided_inputs": rep.get("provided_inputs"),
                "strict_notes": fired,
                "missing": (rep.get("extraction_trace") or {}).get("missing_fields"),
            }, ensure_ascii=False) + "\n")
            fo.flush()
            print(f"[{args.tag} {i}/{len(qs)}] {row['case_id']} {v}"
                  + (" [STRICT]" if fired else ""))

    # Recount from the file so a resumed run summarises every row, not only this session's.
    counts = {"correct": 0, "silent_wrong": 0, "abstain": 0, "near_miss": 0, "strict_fired": 0}
    for line in open(out_path):
        if line.strip():
            r = json.loads(line)
            counts[r["verdict"]] += 1
            counts["near_miss"] += bool(r.get("scale_near_miss"))
            counts["strict_fired"] += bool(r.get("strict_notes"))
    summary = {"tag": args.tag, "model": args.model, "lib_db": args.lib_db, "top_k": args.top_k,
               "strict_binding_flag": flag, "n": len(qs), **counts,
               "elapsed_s": round(time.time() - t0, 1),
               "ran_at": datetime.now().isoformat(timespec="seconds")}
    json.dump(summary, open(os.path.join(args.out_dir, f"{args.tag}_summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
