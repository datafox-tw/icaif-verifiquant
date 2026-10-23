"""
run_paper_experiments.py
------------------------
VerifiQuant FinNLP 2026 — 主實驗執行腳本

用法（跑全部）：
    python3 scripts/run_paper_experiments.py \
        --config results/paper_v1/experiment_config.yaml

用法（只跑特定 baseline）：
    python3 scripts/run_paper_experiments.py \
        --config results/paper_v1/experiment_config.yaml \
        --only vq_flash cot_single_shot_flash

實驗順序：
    1. Build FIC cards（dataset_case_to_fic.py）
    2. Build card store（build_card_store.py）
    3. 跑每個 pipeline baseline
    4. 彙整所有 summary → paper_results_summary.json

每個 baseline 的輸出存在：
    <run_dir>/results/<baseline_name>/output.jsonl
    <run_dir>/results/<baseline_name>/summary.json
    <run_dir>/results/<baseline_name>/run_log.json   ← 記錄 timing、model、參數

設計原則：
- 每個 baseline 獨立失敗，不影響其他
- 已有 output.jsonl 的 baseline 預設跳過（--force 強制重跑）
- 所有參數從 config YAML 讀，不硬編碼
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# ── 工具函式 ──────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    try:
        import yaml
        with open(config_path) as f:
            return yaml.safe_load(f)
    except ImportError:
        # fallback：如果 yaml 沒裝，嘗試 JSON（sample_dataset.py 的 fallback 輸出）
        json_path = config_path.replace(".yaml", ".json")
        with open(json_path) as f:
            return json.load(f)


def run_cmd(cmd: list[str], log_path: str, description: str) -> tuple[bool, float]:
    """執行一個 subprocess 命令，把 stdout/stderr 寫到 log_path。"""
    print(f"\n[run] {description}")
    print(f"  $ {' '.join(cmd)}")
    start = time.time()
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as logf:
        result = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - start
    status = "✅" if result.returncode == 0 else "❌"
    print(f"  {status} returncode={result.returncode}  ({elapsed:.1f}s)  log→{log_path}")
    return result.returncode == 0, elapsed


def save_run_log(path: str, baseline: str, cmd: list, success: bool, elapsed: float, params: dict):
    """儲存單一 baseline 的執行 metadata。"""
    log = {
        "baseline": baseline,
        "ran_at": datetime.now().isoformat(),
        "success": success,
        "elapsed_seconds": round(elapsed, 1),
        "command": cmd,
        "params": params,
    }
    with open(path, "w") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)


# ── Step 1: FIC Card Generation ───────────────────────────────────────────

def build_fic_cards(config: dict, run_dir: str, force: bool) -> bool:
    """Dataset → FIC core/retrieval/repair cards."""
    fic_dir = os.path.join(run_dir, "fic")
    core_out = os.path.join(fic_dir, "core.jsonl")
    retrieval_out = os.path.join(fic_dir, "retrieval.jsonl")
    repair_out = os.path.join(fic_dir, "repair.jsonl")

    if not force and os.path.exists(core_out):
        print(f"[fic] 已存在 core.jsonl，跳過（--force 強制重建）")
        return True

    Path(fic_dir).mkdir(parents=True, exist_ok=True)
    cmd = [
        "python3", "-m", "verifiquant.preprocessing.dataset_case_to_fic",
        "--input", config["output_jsonl"],
        "--functions-catalog-path", "verifiquant/data/functions-article-all.json",
        "--financial-docs-path", "verifiquant/data/financial_documents.json",
        "--core-output", core_out,
        "--retrieval-output", retrieval_out,
        "--repair-output", repair_out,
        "--duplicate-fic-policy", "suffix",
        "--on-validation-error", "save",
        "--validation-report", os.path.join(fic_dir, "validation_report.json"),
        "--seed-report-output", os.path.join(fic_dir, "seed_report.json"),
        "--skip-existing-core",
        "--checkpoint-every-record",   # crash-safe: saves after each card
    ]
    success, elapsed = run_cmd(cmd, os.path.join(fic_dir, "build_fic.log"), "Building FIC cards")
    return success


def build_card_store(run_dir: str, force: bool) -> str:
    """FIC cards → SQLite card store。回傳 db_url。"""
    db_path = os.path.join(run_dir, "fic", "cards.db")
    db_url = f"sqlite:///{os.path.abspath(db_path)}"

    if not force and os.path.exists(db_path):
        print(f"[cardstore] 已存在 cards.db，跳過（--force 強制重建）")
        return db_url

    fic_dir = os.path.join(run_dir, "fic")
    cmd = [
        "python3", "-m", "verifiquant.card_store",
        "--db-url", db_url,
        "--core", os.path.join(fic_dir, "core.jsonl"),
        "--retrieval", os.path.join(fic_dir, "retrieval.jsonl"),
        "--repair", os.path.join(fic_dir, "repair.jsonl"),
    ]
    run_cmd(cmd, os.path.join(fic_dir, "build_store.log"), "Building card store")
    return db_url


# ── Step 2: Pipeline Runners ──────────────────────────────────────────────

def run_vq(baseline_name: str, params: dict, config: dict, run_dir: str, db_url: str, force: bool):
    """跑 VerifiQuant error-classification + framework-guided pipeline。"""
    result_dir = os.path.join(run_dir, "results", baseline_name)
    output_jsonl = os.path.join(result_dir, "output.jsonl")
    summary_json = os.path.join(result_dir, "summary.json")

    if not force and os.path.exists(summary_json):
        print(f"[{baseline_name}] 已有 summary.json，跳過")
        return

    Path(result_dir).mkdir(parents=True, exist_ok=True)

    # Per-pipeline FIC card-version override. If a pipeline declares `db_path`
    # (relative to run_dir, e.g. "fic/cards_v3.db"), use it instead of the
    # globally-built default cards.db. This makes the canonical V3-card results
    # reproducible and pins each pipeline to an explicit card version.
    pipeline_db_url = db_url
    if params.get("db_path"):
        db_abs = os.path.abspath(os.path.join(run_dir, params["db_path"]))
        pipeline_db_url = f"sqlite:///{db_abs}"
        print(f"[{baseline_name}] 使用指定卡片版本 db_path={params['db_path']}")

    cmd = [
        "python3", "-m", "verifiquant.pipeline.run_framework_guided_self_improve_pipeline",
        "--input", config["output_jsonl"],
        "--db-url", pipeline_db_url,
        "--output", output_jsonl,
        "--summary-output", summary_json,
        "--max-turns", str(params.get("max_turns", 3)),
        "--top-k", str(params.get("top_k", 4)),
        "--selector-model", params.get("selector_model", "gemini-2.5-flash"),
        "--extractor-model", params.get("extractor_model", "gemini-2.5-flash"),
        "--judge-model", params.get("judge_model", "gemini-2.5-flash"),
        "--oracle-model", params.get("oracle_model", "gemini-2.5-flash"),
    ]

    success, elapsed = run_cmd(
        cmd,
        os.path.join(result_dir, "stdout.log"),
        f"VQ pipeline: {baseline_name}",
    )
    save_run_log(os.path.join(result_dir, "run_log.json"), baseline_name, cmd, success, elapsed, params)


def run_cot(baseline_name: str, params: dict, config: dict, run_dir: str, force: bool):
    """跑 CoT self-improve pipeline（single-shot 和 basic oracle 變體）。"""
    result_dir = os.path.join(run_dir, "results", baseline_name)
    output_jsonl = os.path.join(result_dir, "output.jsonl")
    summary_json = os.path.join(result_dir, "summary.json")

    if not force and os.path.exists(summary_json):
        print(f"[{baseline_name}] 已有 summary.json，跳過")
        return

    Path(result_dir).mkdir(parents=True, exist_ok=True)

    # 所有 CoT 變體（single-shot / basic oracle）均使用同一 script；
    # max_turns=1 → single-shot，max_turns>1 → self-improve with basic oracle prompt
    script = "verifiquant.pipeline.run_cot_self_improve_pipeline"

    cmd = [
        "python3", "-m", script,
        "--input", config["output_jsonl"],
        "--output", output_jsonl,
        "--summary-output", summary_json,
        "--max-turns", str(params.get("max_turns", 1)),
        "--cot-model", params.get("cot_model", "gemini-2.5-flash"),
        "--oracle-model", params.get("oracle_model", "gemini-2.5-flash"),
    ]

    success, elapsed = run_cmd(
        cmd,
        os.path.join(result_dir, "stdout.log"),
        f"CoT pipeline: {baseline_name}",
    )
    save_run_log(os.path.join(result_dir, "run_log.json"), baseline_name, cmd, success, elapsed, params)


# ── JP Morgan Runner ──────────────────────────────────────────────────────

# 注意：reimplement-jpmogan/ 是獨立目錄，要從其路徑執行
# input 格式：JSONL（與 verifiquant 的 questions_50.jsonl 相同 schema）
# output 格式：JSONL（每筆含 is_correct、pipeline_error 等欄位）

JPM_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reimplement-jpmogan")
# JP Morgan has its own venv with langgraph/langchain; use it instead of the system python
JPM_PYTHON = os.path.join(JPM_DIR, "venv", "bin", "python3")

def run_jpmorgan(baseline_name: str, params: dict, config: dict, run_dir: str, force: bool):
    """跑 JP Morgan MAS reimplementation。"""
    result_dir = os.path.join(run_dir, "results", baseline_name)
    output_jsonl = os.path.join(result_dir, "output.jsonl")
    summary_json = os.path.join(result_dir, "summary.json")

    if not force and os.path.exists(summary_json):
        print(f"[{baseline_name}] 已有 summary.json，跳過（--force 強制重跑）")
        return

    Path(result_dir).mkdir(parents=True, exist_ok=True)

    # JP Morgan pipeline 要從其目錄跑（有相對 import）
    # 使用絕對路徑的 input / output / summary
    input_abs = os.path.abspath(config["output_jsonl"])
    output_abs = os.path.abspath(output_jsonl)
    summary_abs = os.path.abspath(summary_json)

    # Use the JPM venv python so langgraph/langchain are available
    jpm_python = JPM_PYTHON if os.path.exists(JPM_PYTHON) else "python3"
    cmd = [
        jpm_python,
        os.path.join(JPM_DIR, "run_pipeline.py"),
        "--input", input_abs,
        "--output", output_abs,
        "--summary-output", summary_abs,
    ]

    # 切換到 JPM_DIR 跑，否則 relative imports 可能失敗
    # 繼承 GEMINI_API_KEY 從父進程（避免 JPM .env 過期問題）
    import dotenv as _dotenv  # type: ignore
    _dotenv.load_dotenv()     # load main project .env into os.environ first
    run_env = os.environ.copy()

    start = time.time()
    print(f"\n[run] JP Morgan MAS: {baseline_name}")
    print(f"  $ cd {JPM_DIR} && {' '.join(cmd)}")
    Path(output_abs).parent.mkdir(parents=True, exist_ok=True)
    log_path = os.path.join(result_dir, "stdout.log")
    with open(log_path, "w") as logf:
        result = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, text=True, cwd=JPM_DIR, env=run_env)
    elapsed = time.time() - start
    success = result.returncode == 0
    status = "✅" if success else "❌"
    print(f"  {status} returncode={result.returncode}  ({elapsed:.1f}s)  log→{log_path}")

    save_run_log(os.path.join(result_dir, "run_log.json"), baseline_name, cmd, success, elapsed, params)


# ── Step 3: Aggregation ───────────────────────────────────────────────────

def aggregate_results(run_dir: str) -> dict:
    """
    掃描所有 baseline 的 summary.json，彙整成一個 paper_results_summary.json。
    計算 correct / wrong / abstain / selective_accuracy / SWR。
    """
    results_root = os.path.join(run_dir, "results")
    aggregated = {}

    for baseline_dir in sorted(Path(results_root).iterdir()):
        if not baseline_dir.is_dir():
            continue
        summary_path = baseline_dir / "summary.json"
        run_log_path = baseline_dir / "run_log.json"
        if not summary_path.exists():
            aggregated[baseline_dir.name] = {"status": "missing_summary"}
            continue

        with open(summary_path) as f:
            summary = json.load(f)

        run_log = {}
        if run_log_path.exists():
            with open(run_log_path) as f:
                run_log = json.load(f)

        # 嘗試解析三元分類
        total = summary.get("total_cases", 50)
        correct = summary.get("correct_count", summary.get("correct", None))
        accuracy = summary.get("accuracy", None)

        # VQ-specific fields
        success_count = summary.get("success_count", None)
        recovery_count = summary.get("recovery_count", None)
        recovery_rate = summary.get("recovery_rate", None)
        correct_one_shot = summary.get("correct_in_one_shot", None)

        if correct is not None:
            # 盡量推算 silent_wrong 和 abstain
            # VQ: success_count = answered (correct + silent_wrong)
            #     total - success_count = abstain/error
            if success_count is not None:
                silent_wrong = success_count - correct
                abstain = total - success_count
            else:
                # CoT: 一般不 abstain
                silent_wrong = total - correct
                abstain = 0

            coverage = (correct + silent_wrong) / total if total else 0
            selective_acc = correct / (correct + silent_wrong) if (correct + silent_wrong) > 0 else 0
            swr = silent_wrong / total if total else 0

            aggregated[baseline_dir.name] = {
                "total": total,
                "correct": correct,
                "silent_wrong": silent_wrong,
                "abstain": abstain,
                "accuracy": round(accuracy, 4) if accuracy else round(correct / total, 4),
                "coverage": round(coverage, 4),
                "selective_accuracy": round(selective_acc, 4),
                "silent_wrong_rate": round(swr, 4),
                "recovery_count": recovery_count,
                "recovery_rate": recovery_rate,
                "correct_one_shot": correct_one_shot,
                "model": run_log.get("params", {}).get("judge_model") or run_log.get("params", {}).get("cot_model"),
                "ran_at": run_log.get("ran_at"),
                "elapsed_seconds": run_log.get("elapsed_seconds"),
                # CoT pipelines embed run_config (prompt templates + mode) in summary
                "run_config": summary.get("run_config"),
            }
        else:
            aggregated[baseline_dir.name] = {"status": "unknown_format", "raw": summary}

    # 存檔
    out_path = os.path.join(run_dir, "paper_results_summary.json")
    with open(out_path, "w") as f:
        json.dump(aggregated, f, indent=2, ensure_ascii=False)
    print(f"\n[aggregate] 彙整完成 → {out_path}")

    # 印出 quick table
    print("\n" + "=" * 80)
    print(f"{'Baseline':<35} {'Correct':>8} {'SW':>6} {'Abstain':>8} {'Sel.Acc':>9} {'SWR':>7}")
    print("-" * 80)
    for name, r in aggregated.items():
        if "correct" in r:
            print(
                f"{name:<35} {r['correct']:>7}/{r['total']:<2}"
                f" {r['silent_wrong']:>6}"
                f" {r['abstain']:>8}"
                f" {r['selective_accuracy']:>9.1%}"
                f" {r['silent_wrong_rate']:>7.1%}"
            )
        else:
            print(f"{name:<35}  {r.get('status', '?')}")
    print("=" * 80)
    return aggregated


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VerifiQuant Paper Experiment Runner")
    parser.add_argument("--config", required=True, help="experiment_config.yaml 路徑")
    parser.add_argument(
        "--only", nargs="+", default=None,
        help="只跑指定 baseline（e.g. --only vq_flash cot_single_shot_flash）",
    )
    parser.add_argument(
        "--skip-fic", action="store_true",
        help="跳過 FIC card generation（已有 fic/ 目錄時用）",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="強制重跑所有 step，即使已有輸出",
    )
    parser.add_argument(
        "--aggregate-only", action="store_true",
        help="只彙整已有結果，不跑任何 pipeline",
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  VerifiQuant Paper Experiment Runner")
    print(f"  Config: {args.config}")
    print(f"  Time  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    config = load_config(args.config)
    run_dir = config.get("run_dir", str(Path(args.config).parent))

    if args.aggregate_only:
        aggregate_results(run_dir)
        return

    # Step 1: Build FIC cards
    if not args.skip_fic:
        ok = build_fic_cards(config, run_dir, args.force)
        if not ok:
            print("❌ FIC card generation 失敗，中止。請查看 fic/build_fic.log")
            sys.exit(1)
        db_url = build_card_store(run_dir, args.force)
    else:
        db_path = os.path.join(run_dir, "fic", "cards.db")
        db_url = f"sqlite:///{os.path.abspath(db_path)}"
        print(f"[fic] --skip-fic：使用現有 {db_url}")

    # Step 2: 跑各 baseline
    pipelines = config.get("pipelines", {})
    only = set(args.only) if args.only else set(pipelines.keys())

    for name, params in pipelines.items():
        if name not in only:
            continue
        pipeline_type = params.get("type", "cot")

        if pipeline_type == "verifiquant":
            run_vq(name, params, config, run_dir, db_url, args.force)
        elif pipeline_type == "cot":
            run_cot(name, params, config, run_dir, args.force)
        elif pipeline_type == "jpmorgan_reimpl":
            run_jpmorgan(name, params, config, run_dir, args.force)
        else:
            print(f"\n[{name}] 未知 pipeline type: {pipeline_type}，略過。")

    # Step 3: 彙整
    aggregate_results(run_dir)
    print(f"\n✅ 全部完成。結果存在 {run_dir}/paper_results_summary.json")


if __name__ == "__main__":
    main()
