# VerifiQuant

**Governance-by-Execution for Financial LLM Workflows** · ACM ICAIF '26, Milan

Yu-Chi Ko, Mi-Yen Yeh, Shou-De Lin · National Taiwan University and Academia Sinica

[Paper (DOI)](https://doi.org/10.1145/3858063.3858688) ·
[Extended version with Appendices A–K](paper/verifiquant_icaif26_extended.pdf) ·
[Demo and replay of the recorded runs](https://www.verifiquant.com) ·
[Datasets](datasets/README.md)

LLMs in financial workflows fail most dangerously with a confident, well-formatted, wrong number.
VerifiQuant packages each formula family as a **Financial Inference Contract** (typed inputs, boundary
checks, declared conventions, locked code) and runs every query through an executable funnel before
anything is computed. The LLM only selects a contract, binds the text to its inputs, and flags which
declared conventions the text leaves open. Executable gates, not the model, decide whether a number may
leave, and every other path ends at a named exit:

| Exit | Meaning | Decided by |
|---|---|---|
| N | No contract covers the ask | retrieval score (code), selector (LLM) |
| M | Several quantities could be meant | selector (LLM) |
| F | A required input is missing | input gate (code) |
| E | An input breaks a declared boundary | boundary gate (code) |
| I | A declared convention is left open (HARD: ask, SOFT: answer with a warning) | ambiguity gate (LLM) |
| C | Every gate passes; the contract's own Python computes | code |

Within the declared library, the benefit is not higher accuracy on well-posed questions but safer
behavior when an input does not justify a number: on 80 traps built so that no valid number exists,
false numeric acceptance falls from 27.5% (the same checks described in a single prompt, R-PAL+) to
10.4% (3-run means, Gemini 2.5 Flash, same top-4 retrieved contracts). The price is coverage, not
accuracy. See the paper for the full results and their boundary.

## Repository map

| Path | What it is |
|---|---|
| [`verifiquant/`](verifiquant/) | The package. `card_store.py` (SQLite contract store, FTS5 BM25 retrieval), `contracts.py` (DiagnosticReport), `preprocessing/` (the three-stage offline contract build, E-check lint, the bounded-repair verifier in `stage_transform.py`), `pipeline/` (the funnel in `run_error_classification_pipeline.py`, the K-turn loop, the CoT baseline, the trap builders) |
| [`datasets/`](datasets/) | Every input and label the paper uses: 250 clean questions, the 250-contract library, 80 traps, the out-of-library split, coverage-sweep subsets, the human-labeled set, RepairBench mutants. Its [README](datasets/README.md) documents each set |
| [`results/`](results/) | The recorded runs behind every reported number (one folder per experiment, see below) |
| [`scripts/`](scripts/) | Dataset builders, experiment runners, analysis, figures, and `assert_paper_numbers.py` |
| [`paper/`](paper/) | Camera-ready LaTeX source (body and Appendices A–K), figures, and both PDFs |
| [`tests/`](tests/) | Offline checks (no API key needed) |

## Install

Python 3.11 or newer.

```bash
python3 -m venv .venv && source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

## Check it offline

No API key is needed for any of this.

```bash
python3 tests/test_offline_smoke.py
```

```bash
python3 tests/test_transform_poc.py
```

Re-derive the paper's numbers from the recorded runs. The script reads every number the camera-ready
source prints (tables, macros and key sentences in `paper/`) and recomputes it from `results/`:

```bash
python3 scripts/assert_paper_numbers.py
```

It ends with `RESULT: all 303 checks pass`.

## Run the funnel

Set `GEMINI_API_KEY` (every experiment used `gemini-2.5-flash`). The paper fixes retrieval depth at 4;
the code's default is 3, so pass it explicitly.

```python
from verifiquant.pipeline.run_error_classification_pipeline import (
    ErrorClassificationAPI, create_genai_client_from_env,
)

api = ErrorClassificationAPI.from_db(
    db_url="sqlite:///datasets/02_contract_library/cards.db",
    client=create_genai_client_from_env(),
    top_k=4,
)
report = api.diagnose(
    question="What is the Price Rate of Change (Price ROC) of the stock over the week?",
    context="At the beginning of the week the stock was priced at $2.5 per share. "
            "By the end of the week it was $165 per share.",
)
print(report["status"], report["diagnostic_type"], report.get("triggered_rule_ids"))
# alert E ['price_coherence_extreme_change']
```

`api.retrieve(question=..., context=..., top_k=4)` lists the candidate contracts without any model call.
For a whole file, `python3 -m verifiquant.pipeline.run_error_classification_pipeline --input Q.jsonl
--db-url sqlite:///datasets/02_contract_library/cards.db --top-k 4 --output out.jsonl` runs one turn
per question, and `verifiquant.pipeline.run_framework_guided_self_improve_pipeline` (`--max-turns 3`)
runs the K-turn clarification loop.

## Where each result comes from

Runner commands call the model and cost API credits. Analysis scripts run offline on `results/`.
Runs are LLM-dependent, so a rerun reproduces the distribution, not the exact rows. Where a runner
writes one, the `*_manifest.json` or `run_log.json` next to a result records the model, retrieval depth,
dataset hash and git commit it was made with.

| Paper result | Recorded in `results/` | Produced by |
|---|---|---|
| Clean set: VerifiQuant K=1/K=3, CoT, CoT + oracle | `paper_v2_250/results/` | `scripts/run_paper_experiments.py --config results/paper_v2_250/experiment_config.yaml` |
| Clean set: R-PAL, PAL | `retrieval_pal_baseline/`, `pal_baseline/` | `run_retrieval_pal_baseline.py`, `run_pal_baseline.py` |
| Clean set: R-PAL+ (3 runs) | `governance_eval/` | `run_retrieval_pal_governance_eval.py --mode prompted`, `score_rpal_plus_clean250.py` |
| Traps: VerifiQuant (3 runs, top-4) | `governance_confirmatory_1014/vq_k1_top4_run*.jsonl` | `run_vq_trap_rerun.py --top-k 4` |
| Traps: R-PAL, R-PAL+ | `governance_confirmatory/conf_{plain,prompted}_run*.jsonl` | `run_retrieval_pal_governance_eval.py --mode plain` / `--mode prompted` |
| Staged controls (App. A) | `governance_confirmatory_1014/staged_*` | `run_staged_rpal_governance_eval.py` |
| Scripted user (App. B) | `scripted_user_1015/` | `run_scripted_user_eval.py` |
| Gemini 2.5 Pro backbone (App. C) | `governance_confirmatory_1015_pro/` | `run_vq_trap_rerun.py` and `run_retrieval_pal_governance_eval.py` with `--model gemini-2.5-pro` |
| Out-of-library split | `ood_eval_top4/` (July top-3 run: `ood_eval/`) | `run_ood_family_eval.py --top-k 4` |
| Coverage sweep (App. I) | `libsweep_top4/` (July top-3 run and the subset libraries: `libsweep/`) | `build_subset_libraries.py`, then `run_libsweep.sh` |
| Gate ladder | `gate_ladder_top4/` (July top-3 run: `gate_ladder/`) | `run_gate_ladder.sh` |

The two shell scripts reproduce the July runs at the runners' default depth of 3. The camera-ready
figures use the 2026-10-15 top-4 reruns, made with the same runners plus `--top-k 4` and an `_top4`
output folder; `analyze_cameraready_topk_reruns.py` compares the two.
| Self-consistency reference | `sc_baseline/` | `run_sc_baseline.py` |
| Human-labeled set (RQ4, App. E) | `messy_query_pilot/` | `build_messy_query_pilot.py`, `aggregate_messy_labels.py`, `score_messy_external.py` |
| RepairBench (App. F) | `repairbench/` | `run_repairbench.py` |
| Safety–utility figure | `mixed_risk_coverage/` | `build_mixed_risk_coverage.py`, `plot_safety_utility.py` |
| Camera-ready rerun summaries | `cameraready_*_summary.json` | `analyze_cameraready_e10_e7.py`, `analyze_cameraready_e6_e12.py`, `analyze_cameraready_topk_reruns.py` |

`results/paper_v1/` holds the 50-question pilot's inputs; its 53-card library is also the
out-of-library split's 50-family library.

## Build a contract library

The contracts were generated offline from FinanceReasoning's reference functions and article text
(three LLM calls per formula family, then lint and smoke tests; no contract was hand-tuned). Fetch
FinanceReasoning at the pinned commit as described in [`datasets/README.md`](datasets/README.md), copy
`functions/functions-article-all.json` and `documents/financial_documents.json` into `verifiquant/data/`,
then run `python3 -m verifiquant.preprocessing.dataset_case_to_fic` (see `scripts/supervise_card_build.sh`
for the exact flags used for the 250-card library) and load the artifacts with
`python3 -m verifiquant.card_store`. The build prompts are printed in Appendix K.

## License

Code: MIT ([`LICENSE`](LICENSE)). Data we authored and the paper: CC BY 4.0
([`LICENSE-DATA.md`](LICENSE-DATA.md)). Fields taken from FinanceReasoning, which carries no license,
remain under the upstream terms.

## Citation

```bibtex
@inproceedings{ko2026verifiquant,
  author    = {Ko, Yu-Chi and Yeh, Mi-Yen and Lin, Shou-De},
  title     = {{VerifiQuant}: Governance-by-Execution for Financial {LLM} Workflows},
  booktitle = {Proceedings of the 7th ACM International Conference on AI in Finance (ICAIF '26)},
  year      = {2026},
  address   = {Milan, Italy},
  publisher = {ACM},
  doi       = {10.1145/3858063.3858688}
}
```

Contact: Yu-Chi Ko, r14946005@ntu.edu.tw
