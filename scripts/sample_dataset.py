"""
sample_dataset.py
-----------------
從 FinanceReasoning medium.json 以分層隨機抽樣（stratified by difficulty quartile）
產生固定的 50 題 JSONL，並輸出完整的 experiment_config.yaml。

用法：
    python3 scripts/sample_dataset.py \
        --source FinanceReasoning/data/FinanceReasoning/medium.json \
        --output-jsonl results/paper_v1/questions_50.jsonl \
        --output-config results/paper_v1/experiment_config.yaml \
        --n 50 \
        --seed 42

設計原則：
- 抽樣不依賴任何模型的 performance（model-agnostic）
- 固定 seed，任何人都能重現
- 按 difficulty 四分位分層，確保難度分布均勻
"""

import argparse
import json
import os
import random
import statistics
from datetime import datetime
from pathlib import Path


def load_and_filter(source_path: str) -> list[dict]:
    """載入 medium.json 並過濾出有 function + ground_truth 的題目。"""
    with open(source_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    qualified = [
        q for q in data
        if "function" in q and "ground_truth" in q and "difficulty" in q
    ]
    print(f"[sample] 可用題目：{len(qualified)} 題（from {len(data)} total）")
    return qualified


def stratified_sample(questions: list[dict], n: int, seed: int) -> list[dict]:
    """
    按 difficulty 分四個 quartile，從每個 quartile 等比例抽樣，合計 n 題。

    分配邏輯：
    - n=50 → Q1:12, Q2:13, Q3:12, Q4:13（harder quartiles 多 1 題）
    - 一般情況：先均分，餘數從 Q2/Q4 補
    """
    rng = random.Random(seed)
    questions_sorted = sorted(questions, key=lambda q: q["difficulty"])

    # 切成 4 個 quartile（等大小的 bins）
    total = len(questions_sorted)
    quartile_size = total // 4
    quartiles = [
        questions_sorted[i * quartile_size: (i + 1) * quartile_size]
        for i in range(4)
    ]
    # 最後一個 quartile 包含所有剩餘
    quartiles[3] = questions_sorted[3 * quartile_size:]

    # 計算每個 quartile 的抽樣數量
    base = n // 4
    remainder = n % 4  # 0–3
    # remainder 的部分分給 Q2, Q4（較難的 quartile 多一題）
    per_quartile = [base, base, base, base]
    for i in [1, 3, 0, 2][:remainder]:  # 先分給 Q2, Q4
        per_quartile[i] += 1

    sampled = []
    for i, (q_list, k) in enumerate(zip(quartiles, per_quartile)):
        if len(q_list) < k:
            raise ValueError(
                f"Quartile {i+1} 只有 {len(q_list)} 題，但需要抽 {k} 題。"
                "請減少 n 或改用更大的資料集。"
            )
        chosen = rng.sample(q_list, k)
        sampled.extend(chosen)
        diffs = [q["difficulty"] for q in q_list]
        print(
            f"  Q{i+1}: {len(q_list)} 題 "
            f"[{min(diffs):.3f}–{max(diffs):.3f}] "
            f"→ 抽 {k} 題"
        )

    # 打亂順序（避免 JSONL 按難度排列影響 pipeline 行為）
    rng.shuffle(sampled)
    return sampled


def save_jsonl(questions: list[dict], path: str) -> None:
    """存成 pipeline 所需的 JSONL 格式。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"[sample] 已儲存 {len(questions)} 題 → {path}")


def save_config(
    questions: list[dict],
    source_path: str,
    output_jsonl: str,
    n: int,
    seed: int,
    config_path: str,
) -> None:
    """輸出 experiment_config.yaml，記錄所有抽樣決策。"""
    import yaml  # type: ignore

    diffs = [q["difficulty"] for q in questions]
    config = {
        "experiment": {
            "created_at": datetime.now().isoformat(),
            "version": "paper_v1",
            "description": "FinNLP 2026 main experiment — 50Q stratified random from FinanceReasoning medium",
        },
        "sampling": {
            "source": source_path,
            "n": n,
            "seed": seed,
            "method": "stratified_random_by_difficulty_quartile",
            "filter": "has_function AND has_ground_truth",
            "difficulty_stats": {
                "min": round(min(diffs), 4),
                "max": round(max(diffs), 4),
                "mean": round(statistics.mean(diffs), 4),
                "median": round(statistics.median(diffs), 4),
            },
            "question_ids": [q["question_id"] for q in questions],
            "function_ids": [q["function_id"] for q in questions],
        },
        "pipelines": {
            # VQ 系列
            "vq_flash": {
                "type": "verifiquant",
                "judge_model": "gemini-2.5-flash",
                "selector_model": "gemini-2.5-flash",
                "extractor_model": "gemini-2.5-flash",
                "oracle_model": "gemini-2.5-flash",
                "max_turns": 3,
                "top_k": 4,
                "description": "VerifiQuant full pipeline, Flash 2.5, K=3 (oracle corrects ≤2x)",
            },
            "vq_pro": {
                "type": "verifiquant",
                "judge_model": "gemini-2.5-pro",
                "selector_model": "gemini-2.5-pro",
                "extractor_model": "gemini-2.5-pro",
                "oracle_model": "gemini-2.5-pro",
                "max_turns": 3,
                "top_k": 4,
                "description": "VerifiQuant full pipeline, Pro 2.5, K=3",
            },
            # CoT 系列
            "cot_single_shot_flash": {
                "type": "cot",
                "cot_model": "gemini-2.5-flash",
                "oracle_model": "gemini-2.5-flash",
                "max_turns": 1,
                "description": "CoT single-shot, Flash 2.5, no self-improve",
            },
            "cot_single_shot_pro": {
                "type": "cot",
                "cot_model": "gemini-2.5-pro",
                "oracle_model": "gemini-2.5-pro",
                "max_turns": 1,
                "description": "CoT single-shot, Pro 2.5, no self-improve",
            },
            "cot_basic_oracle_flash": {
                "type": "cot",
                "cot_model": "gemini-2.5-flash",
                "oracle_model": "gemini-2.5-flash",
                "max_turns": 3,
                "description": "CoT + basic oracle self-improve, Flash 2.5, ≤2 corrections",
            },
            "cot_vq_funnel_flash": {
                "type": "cot_vq_funnel",
                "cot_model": "gemini-2.5-flash",
                "oracle_model": "gemini-2.5-flash",
                "max_turns": 3,
                "description": "CoT + VQ 6-layer funnel concept in oracle prompt, Flash 2.5",
            },
            "cot_vq_funnel_pro": {
                "type": "cot_vq_funnel",
                "cot_model": "gemini-2.5-pro",
                "oracle_model": "gemini-2.5-pro",
                "max_turns": 3,
                "description": "CoT + VQ 6-layer funnel concept in oracle prompt, Pro 2.5",
            },
            # JP Morgan reimplementation
            "jpmorgan": {
                "type": "jpmorgan_reimpl",
                "description": "JP Morgan MAS reimplementation — run separately",
                "note": "See reimplementation-jpmorgan/ directory",
            },
        },
        "output_jsonl": output_jsonl,
        "run_dir": str(Path(config_path).parent),
    }

    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    print(f"[sample] 已儲存 experiment config → {config_path}")


def main():
    parser = argparse.ArgumentParser(description="Stratified random sampling from FinanceReasoning")
    parser.add_argument(
        "--source",
        default="FinanceReasoning/data/FinanceReasoning/medium.json",
        help="原始 medium.json 路徑",
    )
    parser.add_argument(
        "--output-jsonl",
        default="results/paper_v1/questions_50.jsonl",
        help="輸出 JSONL 路徑",
    )
    parser.add_argument(
        "--output-config",
        default="results/paper_v1/experiment_config.yaml",
        help="輸出 experiment config YAML 路徑",
    )
    parser.add_argument("--n", type=int, default=50, help="抽樣題數（預設 50）")
    parser.add_argument("--seed", type=int, default=42, help="Random seed（預設 42）")
    args = parser.parse_args()

    print(f"\n=== VerifiQuant Paper Experiment — Dataset Sampling ===")
    print(f"  Source  : {args.source}")
    print(f"  N       : {args.n}")
    print(f"  Seed    : {args.seed}")
    print(f"  Output  : {args.output_jsonl}\n")

    questions = load_and_filter(args.source)
    sampled = stratified_sample(questions, args.n, args.seed)

    diffs = [q["difficulty"] for q in sampled]
    print(f"\n[sample] 抽樣結果：{len(sampled)} 題")
    print(f"  difficulty: {min(diffs):.3f} – {max(diffs):.3f}  mean={statistics.mean(diffs):.3f}")

    save_jsonl(sampled, args.output_jsonl)

    try:
        save_config(
            sampled, args.source, args.output_jsonl,
            args.n, args.seed, args.output_config
        )
    except ImportError:
        print("[sample] 注意：yaml 套件未安裝（pip install pyyaml），略過 config 輸出。")
        # Fallback: save as JSON
        config_json = args.output_config.replace(".yaml", ".json")
        with open(config_json, "w") as f:
            json.dump({"seed": args.seed, "n": args.n, "question_ids": [q["question_id"] for q in sampled]}, f, indent=2)
        print(f"[sample] 已改存 JSON config → {config_json}")

    print("\n✅ 完成。下一步：")
    print(f"   python3 scripts/run_paper_experiments.py --config {args.output_config}")


if __name__ == "__main__":
    main()
