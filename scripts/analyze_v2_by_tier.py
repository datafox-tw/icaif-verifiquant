"""
analyze_v2_by_tier.py
---------------------
paper_v2_250 成果整理：把 CoT / VQ 的 output.jsonl 按 medium/hard tier 分層,
產出三元分類 (Correct / Silent Wrong / Abstain) + Coverage / Sel.Acc / SWR。

同時支援 paper_v1 (50Q) 作為驗證：跑 v1 應重現 master record 的已知數字。

用法：
    python3 scripts/analyze_v2_by_tier.py \
        --questions results/paper_v2_250/questions_250.jsonl \
        --hard-min 4.15 \
        --run results/paper_v2_250/results/cot_single_shot_flash/output.jsonl cot \
        --run results/paper_v2_250/results/vq_flash/output.jsonl vq

三元分類規則（與 run_paper_experiments.aggregate_results 對齊）：
- cot:  final_is_correct=True → correct；有 final_answer 且不對 → silent_wrong；
        無 final_answer → abstain（CoT 正常不 abstain,此欄應為 0,>0 時列警告）
- vq :  final_is_correct=True → correct；final_status=success 且不對 → silent_wrong；
        其他（refusal/error/alert/needs_clarification）→ abstain
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def classify(row: dict, kind: str) -> str:
    payload = row.get("cot_self_improve") if kind == "cot" else row.get("framework_guided")
    payload = payload or {}
    if payload.get("final_is_correct") is True:
        return "correct"
    if kind == "cot":
        ans = payload.get("final_answer")
        return "silent_wrong" if ans not in (None, "",) else "abstain"
    status = str(payload.get("final_status", ""))
    return "silent_wrong" if status == "success" else "abstain"


def tier_of(difficulty: float, hard_min: float) -> str:
    return "hard" if difficulty >= hard_min else "medium"


def report(name: str, rows: list[dict], kind: str, qinfo: dict, hard_min: float) -> dict:
    buckets: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    missing = []
    for r in rows:
        cid = r.get("case_id") or r.get("source_sample_id")
        if cid not in qinfo:
            missing.append(cid)
            continue
        tier = tier_of(qinfo[cid], hard_min)
        buckets[tier][classify(r, kind)].append(cid)
        buckets["all"][classify(r, kind)].append(cid)

    out = {"run": name, "kind": kind, "n_rows": len(rows), "unmatched_ids": missing}
    print(f"\n=== {name} ({kind}, n={len(rows)}) ===")
    if missing:
        print(f"  ⚠️ {len(missing)} rows 的 case_id 不在 questions 檔: {missing[:5]}")
    hdr = f"  {'tier':<8} {'N':>4} {'correct':>8} {'SW':>4} {'abstain':>8} {'cov':>7} {'SelAcc':>8} {'SWR':>7}"
    print(hdr)
    for tier in ("medium", "hard", "all"):
        b = buckets.get(tier)
        if not b:
            continue
        c, sw, ab = len(b["correct"]), len(b["silent_wrong"]), len(b["abstain"])
        n = c + sw + ab
        cov = (c + sw) / n if n else 0
        sel = c / (c + sw) if (c + sw) else 0
        swr = sw / n if n else 0
        print(f"  {tier:<8} {n:>4} {c:>8} {sw:>4} {ab:>8} {cov:>7.1%} {sel:>8.1%} {swr:>7.1%}")
        out[tier] = {"n": n, "correct": c, "silent_wrong": sw, "abstain": ab,
                     "coverage": round(cov, 4), "selective_accuracy": round(sel, 4),
                     "swr": round(swr, 4),
                     "silent_wrong_ids": b["silent_wrong"], "abstain_ids": b["abstain"]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--hard-min", type=float, default=4.15,
                    help="difficulty >= 此值視為 hard tier（250Q 抽樣: medium max 4.1431, hard min 4.1589）")
    ap.add_argument("--run", nargs=2, action="append", metavar=("OUTPUT_JSONL", "KIND"),
                    required=True, help="KIND ∈ {cot, vq}，可重複")
    ap.add_argument("--json-out", help="彙整結果另存 JSON")
    args = ap.parse_args()

    qinfo = {q["question_id"]: float(q["difficulty"]) for q in load_jsonl(args.questions)}
    n_hard = sum(1 for d in qinfo.values() if d >= args.hard_min)
    print(f"questions: {len(qinfo)} (medium {len(qinfo)-n_hard} / hard {n_hard}, hard-min={args.hard_min})")

    results = []
    for path, kind in args.run:
        if not Path(path).exists():
            print(f"\n=== {path} — 不存在，跳過 ===")
            continue
        results.append(report(Path(path).parent.name, load_jsonl(path), kind, qinfo, args.hard_min))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(f"\nJSON → {args.json_out}")


if __name__ == "__main__":
    main()
