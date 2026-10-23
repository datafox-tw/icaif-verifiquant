#!/usr/bin/env python3
"""PAL/PoT-style execution baseline (Wave 1-C).

The LLM writes a self-contained Python solve() function; we execute it in a
subprocess (10 s timeout, import denylist) and score the printed number. This
isolates what deterministic *execution alone* buys, without contracts or gates —
the ablation the reviewer asked for ("is the gain from not letting the LLM do
arithmetic, or from the contracts/gates?").

Usage:
  PYTHONPATH=. GEMINI_API_KEY=... .venv/bin/python scripts/run_pal_baseline.py \
      [--model gemini-2.5-flash] [--limit 0]
"""
from __future__ import annotations
import argparse, ast, json, os, re, subprocess, sys, time
from datetime import datetime

from verifiquant.pipeline.run_error_classification_pipeline import create_genai_client_from_env

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q250 = os.path.join(ROOT, "results/paper_v2_250/questions_250.jsonl")

PROMPT = """You are a financial reasoning assistant that solves problems by writing Python.
Write a single self-contained Python function solve() that computes the numeric answer.
Rules:
- Only the Python standard library `math` may be imported; no other imports.
- No input(), no file or network access.
- solve() must return a single number (int or float).
- If the problem cannot be solved from the given information, return the string "CANNOT SOLVE".
Return ONLY the code, in one ```python block.

Question:
{question}

Context:
{context}
"""

FORBIDDEN_CALLS = {"open", "exec", "eval", "__import__"}


def deny_reason(code: str) -> str:
    """Return an AST-grounded sandbox rejection reason, or an empty string.

    The original regex scanned docstrings and comments, so harmless prose such
    as ``from the context`` was misclassified as a forbidden import.  Parsing
    the program first keeps the stated policy (only ``math`` imports and no
    dangerous builtins) without turning English prose into lost coverage.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ""  # The execution path will classify invalid syntax as exec_fail.
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            forbidden = [alias.name for alias in node.names if alias.name != "math"]
            if forbidden:
                return f"forbidden_import:{','.join(forbidden)}"
        elif isinstance(node, ast.ImportFrom):
            if node.module != "math":
                return f"forbidden_import_from:{node.module or '<relative>'}"
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALLS:
                return f"forbidden_call:{node.func.id}"
    return ""


def extract_code(text: str) -> str | None:
    m = re.search(r"```(?:python)?\s*(.*?)```", text or "", re.DOTALL)
    return m.group(1).strip() if m else None


def run_code(code: str) -> str | None:
    prog = code + "\n\nprint(solve())\n"
    try:
        r = subprocess.run([sys.executable, "-I", "-c", prog],
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip().splitlines()[-1] if r.stdout.strip() else None
    except Exception:
        return None


def _num(x):
    if isinstance(x, bool):
        return float(x)
    normalized = str(x).strip().lower()
    if normalized == "true":
        return 1.0
    if normalized == "false":
        return 0.0
    try:
        return float(str(x).replace(",", "").replace("%", "").strip())
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--questions", default=Q250)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results/pal_baseline"))
    args = ap.parse_args()

    qs = [json.loads(l) for l in open(args.questions) if l.strip()]
    if args.limit:
        qs = qs[: args.limit]
    client = create_genai_client_from_env()
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"pal_{args.model.replace('.', '').replace('-', '_')}"
    out_path = os.path.join(args.out_dir, f"{tag}.jsonl")
    counts = {"correct": 0, "silent_wrong": 0, "abstain": 0, "near_miss": 0,
              "code_fail": 0, "denied": 0}
    t0 = time.time()

    with open(out_path, "w") as fo:
        for i, q in enumerate(qs, 1):
            prompt = PROMPT.format(question=q.get("question", ""), context=q.get("context", "") or "(none)")
            out_val, reason = None, ""
            try:
                resp = client.models.generate_content(model=args.model, contents=prompt)
                code = extract_code(resp.text or "")
            except Exception as exc:
                code, reason = None, f"api:{exc}"[:80]
            if code is None:
                counts["code_fail"] += 1
                v = "abstain"
            elif deny_reason(code):
                counts["denied"] += 1
                v = "abstain"; reason = "denylist"
            else:
                out = run_code(code)
                if out is None:
                    counts["code_fail"] += 1
                    v = "abstain"; reason = "exec_fail"
                elif "CANNOT SOLVE" in out.upper():
                    v = "abstain"; reason = "declined"
                else:
                    out_val = out
                    a, b = _num(out), _num(q.get("ground_truth"))
                    if a is not None and b is not None and abs(a - b) <= max(1e-6, abs(b) * 0.01):
                        v = "correct"
                    else:
                        v = "silent_wrong"
                        if a is not None and b is not None and any(
                                abs(a * f - b) <= max(1e-9, abs(b) * 0.01) for f in (100.0, 0.01)):
                            counts["near_miss"] += 1
            counts[v] += 1
            fo.write(json.dumps({"case_id": q.get("question_id"), "gold": q.get("ground_truth"),
                                 "verdict": v, "output": out_val, "reason": reason},
                                ensure_ascii=False) + "\n")
            fo.flush()
            print(f"[{tag} {i}/{len(qs)}] {q.get('question_id')} {v} {reason}")

    summary = {"tag": tag, "model": args.model, "n": len(qs), **counts,
               "accuracy": counts["correct"] / len(qs),
               "swr": counts["silent_wrong"] / len(qs),
               "swr_unit_normalized": (counts["silent_wrong"] - counts["near_miss"]) / len(qs),
               "elapsed_s": round(time.time() - t0, 1),
               "ran_at": datetime.now().isoformat(timespec="seconds")}
    json.dump(summary, open(os.path.join(args.out_dir, f"{tag}_summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
