#!/usr/bin/env python3
"""Re-run VerifiQuant K=1 on the 80-trap confirmatory set (camera-ready E10).

The decision path is the same function the original July run used:
``_framework_oracle_loop`` from run_framework_guided_self_improve_pipeline with
``max_turns=1``, so the oracle is never called and nothing sees the gold. This
wrapper only adds what the original CLI lacked for a long run:

- one output row per case, flushed as it finishes (the CLI wrote at the end);
- a case-level retry on API exceptions (the CLI crashed the whole run);
- ``--resume`` and a manifest with the git commit and input hashes.

Output rows have the same shape as ``governance_confirmatory/vq_k1_traps.jsonl``,
so ``analyze_governance_eval.normalize_vq`` scores them unchanged.

Note on depth: the July file ``vq_k1_traps.jsonl`` has exactly 3 candidates on
all 80 rows (the CLI default ``--top-k 3``), while R-PAL+ and the clean 250 run
used 4. Pass ``--top-k`` explicitly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verifiquant.pipeline.run_error_classification_pipeline import (  # noqa: E402
    ErrorClassificationAPI,
    create_genai_client_from_env,
)
from verifiquant.pipeline.run_framework_guided_self_improve_pipeline import (  # noqa: E402
    _framework_oracle_loop,
)

G = ROOT / "results/governance_confirmatory"
DEFAULT_DB = ROOT / "results/paper_v2_250/fic/cards.db"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return ""


def _client_with_timeout(timeout_s: int) -> Any:
    """create_genai_client_from_env plus an HTTP timeout.

    Without one, a request on a connection that died (for example after a
    network switch) blocks forever; with one it raises and the caller retries.
    """
    import os
    from google import genai
    from google.genai import types as genai_types

    create_genai_client_from_env()  # same key check and error as every other runner
    return genai.Client(
        api_key=os.environ["GEMINI_API_KEY"],
        http_options=genai_types.HttpOptions(timeout=timeout_s * 1000),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=G / "trap_set.jsonl")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--m-min-top-score", type=float, default=0.05)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--http-timeout-s", type=int, default=120)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--shard", default="", help="i/n: keep rows whose index mod n == i")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.dataset.open(encoding="utf-8") if line.strip()]
    if args.shard:
        shard_i, shard_n = (int(x) for x in args.shard.split("/"))
        rows = [r for idx, r in enumerate(rows) if idx % shard_n == shard_i]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{args.run_name}.jsonl"
    manifest_path = args.out_dir / f"{args.run_name}_manifest.json"
    if out_path.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {out_path}; use --resume")

    done: set[str] = set()
    if args.resume and out_path.exists():
        done = {str(json.loads(line)["case_id"]) for line in out_path.open(encoding="utf-8") if line.strip()}

    manifest = {
        "protocol": "vq_k1_trap_rerun_v1",
        "experiment": "camera-ready E10",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "model": args.model,
        "http_timeout_s": args.http_timeout_s,
        "max_turns": 1,
        "top_k": args.top_k,
        "shard": args.shard,
        "m_min_top_score": args.m_min_top_score,
        "dataset": {"path": str(args.dataset.resolve()), "sha256": _sha256_file(args.dataset)},
        "card_db": {"path": str(args.db.resolve()), "sha256": _sha256_file(args.db)},
        "decision_path": "run_framework_guided_self_improve_pipeline._framework_oracle_loop(max_turns=1)",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    client = _client_with_timeout(args.http_timeout_s)
    api = ErrorClassificationAPI.from_db(
        db_url=f"sqlite:///{args.db.resolve()}",
        client=client,
        selector_model=args.model,
        extractor_model=args.model,
        judge_model=args.model,
        top_k=args.top_k,
        m_min_top_score=args.m_min_top_score,
    )

    started = time.time()
    failed: List[str] = []
    with out_path.open("a" if args.resume else "w", encoding="utf-8") as fh:
        for index, row in enumerate(rows, 1):
            cid = str(row.get("case_id", row.get("question_id")))
            if cid in done:
                continue
            fw: Dict[str, Any] | None = None
            errors: List[str] = []
            for attempt in range(args.max_retries + 1):
                try:
                    fw = _framework_oracle_loop(
                        api=api, client=client, oracle_model=args.model, row=row, max_turns=1,
                    )
                    break
                except Exception as exc:  # API 5xx / quota / malformed JSON
                    errors.append(f"{type(exc).__name__}: {str(exc)[:200]}")
                    time.sleep(min(2 ** attempt * 2, 60))
            if fw is None:
                failed.append(cid)
                print(f"[{args.run_name} {index}/{len(rows)}] {cid} FAILED {errors[-1]}", flush=True)
                continue
            record = {
                "case_id": cid,
                "source_sample_id": row.get("source_sample_id", row.get("question_id")),
                "framework_guided": fw,
                "retry_errors": errors,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            diag = fw["history"][0]["diagnostic"]
            print(f"[{args.run_name} {index}/{len(rows)}] {cid} {diag.get('status')} "
                  f"{diag.get('diagnostic_type')}", flush=True)

    manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["elapsed_s"] = round(time.time() - started, 1)
    manifest["failed_case_ids"] = failed
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run": args.run_name, "failed": failed, "elapsed_s": manifest["elapsed_s"]}), flush=True)


if __name__ == "__main__":
    main()
