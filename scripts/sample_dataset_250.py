"""
sample_dataset_250.py
---------------------
Phase 0 of the scale-up (docs/plan_scale_up_experiments.md): build a 250Q set
= 180 medium + 70 hard, stratified by difficulty quartile, seed 42, as a strict
SUPERSET of the canonical paper_v1 50Q (so the old results stay a sub-analysis).

medium: force-include the 50 canonical IDs, then balance to 45/quartile (180 total).
hard:   stratified 70 from the filtered hard pool.

Usage:
    python3 scripts/sample_dataset_250.py
"""

import json
import random
import statistics
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MEDIUM = ROOT / "FinanceReasoning/data/FinanceReasoning/medium.json"
HARD = ROOT / "FinanceReasoning/data/FinanceReasoning/hard.json"
PAPER_V1_CFG = ROOT / "results/paper_v1/experiment_config.yaml"
RUN_DIR = ROOT / "results/paper_v2_250"
OUT_JSONL = RUN_DIR / "questions_250.jsonl"
OUT_CFG = RUN_DIR / "experiment_config.yaml"

N_MEDIUM, N_HARD, SEED = 180, 70, 42


def load_filtered(path):
    data = json.load(open(path, encoding="utf-8"))
    return [q for q in data if "function" in q and "ground_truth" in q and "difficulty" in q]


def quartiles(qs):
    qs = sorted(qs, key=lambda q: q["difficulty"])
    qsz = len(qs) // 4
    bins = [qs[i * qsz:(i + 1) * qsz] for i in range(4)]
    bins[3] = qs[3 * qsz:]
    return bins


def sample_medium_superset(med, force_ids, n, seed):
    rng = random.Random(seed)
    bins = quartiles(med)
    bin_of = {q["question_id"]: i for i, b in enumerate(bins) for q in b}
    forced = [q for q in med if q["question_id"] in force_ids]
    # target 45 per quartile; subtract forced already present in each
    per_q = [n // 4] * 4
    for r in range(n % 4):
        per_q[[1, 3, 0, 2][r]] += 1
    forced_per_q = [sum(1 for q in forced if bin_of[q["question_id"]] == i) for i in range(4)]
    selected = list(forced)
    for i, b in enumerate(bins):
        need = per_q[i] - forced_per_q[i]
        pool = [q for q in b if q["question_id"] not in force_ids]
        if need < 0:
            raise ValueError(f"medium Q{i+1}: forced {forced_per_q[i]} > target {per_q[i]}")
        if need > len(pool):
            raise ValueError(f"medium Q{i+1}: need {need} but only {len(pool)} available")
        selected += rng.sample(pool, need)
    rng.shuffle(selected)
    return selected


def stratified_sample(qs, n, seed):
    rng = random.Random(seed)
    bins = quartiles(qs)
    per_q = [n // 4] * 4
    for r in range(n % 4):
        per_q[[1, 3, 0, 2][r]] += 1
    out = []
    for i, (b, k) in enumerate(zip(bins, per_q)):
        if k > len(b):
            raise ValueError(f"hard Q{i+1}: need {k} but only {len(b)}")
        out += rng.sample(b, k)
    rng.shuffle(out)
    return out


def _stats(qs):
    d = [q["difficulty"] for q in qs]
    return {"n": len(qs), "min": round(min(d), 4), "max": round(max(d), 4),
            "mean": round(statistics.mean(d), 4), "median": round(statistics.median(d), 4)}


def main():
    import yaml
    med = load_filtered(MEDIUM)
    hard = load_filtered(HARD)
    orig = set(yaml.safe_load(open(PAPER_V1_CFG))["sampling"]["question_ids"])
    print(f"[pool] medium filtered={len(med)}  hard filtered={len(hard)}  canonical 50={len(orig)}")
    assert orig.issubset({q['question_id'] for q in med}), "canonical 50 not all in medium pool"

    med_sel = sample_medium_superset(med, orig, N_MEDIUM, SEED)
    hard_sel = stratified_sample(hard, N_HARD, SEED)
    for q in med_sel:
        q["level"] = q.get("level", "medium")
    for q in hard_sel:
        q["level"] = "hard"
    all_sel = med_sel + hard_sel
    rng = random.Random(SEED)
    rng.shuffle(all_sel)

    sel_ids = {q["question_id"] for q in med_sel}
    assert orig.issubset(sel_ids), "superset broken: some canonical IDs dropped"
    print(f"[medium] {_stats(med_sel)}  (superset of 50: {orig.issubset(sel_ids)})")
    print(f"[hard]   {_stats(hard_sel)}")
    print(f"[total]  {_stats(all_sel)}")

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for q in all_sel:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"[save] {len(all_sel)} -> {OUT_JSONL}")

    cfg = {
        "experiment": {
            "created_at": datetime.now().isoformat(), "version": "paper_v2_250",
            "description": "Scale-up 250Q (180 medium + 70 hard), superset of paper_v1 50Q, seed 42",
        },
        "sampling": {
            "source_medium": str(MEDIUM), "source_hard": str(HARD),
            "n_medium": N_MEDIUM, "n_hard": N_HARD, "seed": SEED,
            "method": "stratified_random_by_difficulty_quartile; medium force-includes canonical 50",
            "filter": "has_function AND has_ground_truth AND has_difficulty",
            "superset_of": "paper_v1 (50Q)",
            "difficulty_stats": {"medium": _stats(med_sel), "hard": _stats(hard_sel), "all": _stats(all_sel)},
            "question_ids": [q["question_id"] for q in all_sel],
            "function_ids": [q["function_id"] for q in all_sel],
            "canonical_50_ids": sorted(orig),
        },
        # This round: Flash full + CoT variants; no Pro scoring, no MAS (see plan).
        "pipelines": {
            "vq_flash": {"type": "verifiquant", "judge_model": "gemini-2.5-flash",
                         "selector_model": "gemini-2.5-flash", "extractor_model": "gemini-2.5-flash",
                         "oracle_model": "gemini-2.5-flash", "max_turns": 3, "top_k": 4,
                         "description": "VQ full, Flash 2.5, K=3 — 250Q"},
            "cot_single_shot_flash": {"type": "cot", "cot_model": "gemini-2.5-flash",
                                      "oracle_model": "gemini-2.5-flash", "max_turns": 1,
                                      "description": "CoT single-shot Flash"},
            "cot_basic_oracle_flash": {"type": "cot", "cot_model": "gemini-2.5-flash",
                                       "oracle_model": "gemini-2.5-flash", "max_turns": 3,
                                       "description": "CoT + blind oracle Flash, K=3"},
        },
        "output_jsonl": str(OUT_JSONL), "run_dir": str(RUN_DIR),
    }
    yaml.dump(cfg, open(OUT_CFG, "w"), allow_unicode=True, sort_keys=False)
    print(f"[save] config -> {OUT_CFG}")
    print("\nPhase 1 next:\n  python3 preprocessing/dataset_case_to_fic.py --input "
          f"{OUT_JSONL} ... (see plan)")


if __name__ == "__main__":
    main()
