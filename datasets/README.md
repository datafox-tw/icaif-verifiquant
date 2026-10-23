# VerifiQuant Datasets

This folder holds every dataset used in the ICAIF '26 paper *VerifiQuant: Governance-by-Execution for Financial LLM Workflows*. It covers the whole chain, from the FinanceReasoning source questions to each evaluation set and contract library the paper reports on.

Only inputs and labels are here. System outputs (model runs, scores) stay in the experiment record under `results/` (see [Where the results live](#where-the-results-live)).

| Folder | What it is | Size | Paper |
|---|---|---|---|
| [`00_source/`](00_source/) | Checksums of the FinanceReasoning files that everything is drawn from | 5 files | §4.1 |
| [`01_clean_250/`](01_clean_250/) | 250 well-posed questions (180 medium, 70 hard) | 250 | §4.1, RQ1, RQ3 |
| [`02_contract_library/`](02_contract_library/) | The 250-card Financial Inference Contract (FIC) library | 250 cards | §3, §4.1 |
| [`03_traps_80/`](03_traps_80/) | Balanced F/E/I/N trap set, plus the scripted user's replies | 80 traps, 26 replies | §4.3, RQ2, RQ3, §6 |
| [`04_out_of_library/`](04_out_of_library/) | Fixed out-of-library split of the 250 and its 50-family library | 199 OOD, 51 ID | §4.1, RQ3 |
| [`05_coverage_sweep/`](05_coverage_sweep/) | Nested library subsets at 10, 25, 50, 75 and 100% | 5 subsets | RQ3, App. I |
| [`06_human_set_80/`](06_human_set_80/) | Human-labeled external-validity set | 80 queries, 5 annotators | §4.5, RQ4, App. E |
| [`07_repair_mutants/`](07_repair_mutants/) | RepairBench patch mutants and verifier verdicts | 906 mutants | §3.3, App. F |
| [`pilot_50/`](pilot_50/) | 50-question pilot and its 50-trap set | 50 + 50 | App. J only |
| `MANIFEST.json` | sha256, size, row count and origin of every file | | |

Section numbers refer to the camera-ready paper. Appendix letters refer to its extended (arXiv) version. Paths such as `scripts/...`, `verifiquant/...` and `results/...` refer to the code repository, <https://github.com/datafox-tw/icaif-verifiquant>.

## How the datasets relate

```
FinanceReasoning (ACL 2025): 2,238 questions = easy 1,000 + medium 1,000 + hard 238
│                            908 of them are "function-linked" (carry a reference function)
│
├─ function-linked medium + hard ── stratified sample, seed 42 ──► 01_clean_250
│     │                                                            (superset of pilot_50)
│     ├─ + reference functions + article text ── 3-stage LLM build ──► 02_contract_library
│     │     ├─ deterministic trap operators ──────────────────────► 03_traps_80
│     │     │     └─ F and I_HARD traps ──────────────────────────► scripted_user_replies (26)
│     │     ├─ nested family subsets, seed 42 ────────────────────► 05_coverage_sweep
│     │     └─ mutated transform specs, seed 0 ───────────────────► 07_repair_mutants
│     └─ split by family membership in the pilot library ─────────► 04_out_of_library
│
├─ function-linked medium, stratified, seed 42 ──────────────────► pilot_50
│     └─ pilot card build (53 cards) ─────────────────────────────► 04_out_of_library/library_50fam
│
└─ easy + medium questions OUTSIDE the 250 ── rewritten by GPT-5.2 ► 06_human_set_80
                                              18 revised offline, labeled by 5 annotators
```

Two facts about this chain matter when you read any result:

1. **The clean set and the traps are in-library by construction.** Every one of the 250 questions has its own card in the library, because the library was built from those questions' reference functions. The cards were written with the source question in view (about a third repeat a value from it, mostly inside ambiguity hints), but the build never reads a gold answer. These two sets measure routing, binding and gating. They do not measure unseen-formula derivation.
2. **The human set is library-agnostic.** Its sources are drawn from outside the 250 (0 of 80 overlap), so it mixes in-library and out-of-library questions in a proportion nobody controlled.

## Getting started

Nothing here needs a model to read. Python's standard library is enough:

```python
import json, sqlite3

def load_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]

clean = load_jsonl("datasets/01_clean_250/questions_250.jsonl")
traps = load_jsonl("datasets/03_traps_80/trap_set.jsonl")
human = load_jsonl("datasets/06_human_set_80/messy_pilot.jsonl")
labels = {r["id"]: r["label"] for r in load_jsonl("datasets/06_human_set_80/ground_truth.jsonl")}

con = sqlite3.connect("datasets/02_contract_library/cards.db")
print(con.execute("SELECT fic_id, name FROM core_cards LIMIT 3").fetchall())
```

To run VerifiQuant itself against a library, install the repository's requirements and set `GEMINI_API_KEY`:

```python
from verifiquant.pipeline.run_error_classification_pipeline import (
    ErrorClassificationAPI, create_genai_client_from_env,
)

api = ErrorClassificationAPI.from_db(
    db_url="sqlite:///datasets/02_contract_library/cards.db",
    client=create_genai_client_from_env(),
    top_k=4,  # the paper fixes retrieval depth at 4
)
report = api.diagnose(question=clean[0]["question"], context=clean[0]["context"])
print(report["status"], report["diagnostic_type"], report.get("output_value"))
```

`api.retrieve(question=..., context=..., top_k=4)` lists the candidate contracts and works without an API key.

## 00_source: FinanceReasoning

FinanceReasoning (Tang et al., ACL 2025) is not redistributed here, because its repository carries no license file. Fetch it at the commit we used and check it against our checksums:

```bash
git clone https://github.com/BUPT-Reasoning/FinanceReasoning.git
```

```bash
git -C FinanceReasoning checkout d444bedf30f9d37e71df356bffe60aee2a994379
```

```bash
cd FinanceReasoning/data && shasum -a 256 -c ../../datasets/00_source/checksums.sha256
```

| File | Used for |
|---|---|
| `FinanceReasoning/{easy,medium,hard}.json` | Question pool for every set below |
| `functions/functions-article-all.json` | Function library given to the R-PAL baselines |
| `documents/financial_documents.json` | Article text attached to each card at build time |

Several files below embed FinanceReasoning fields (question, context, reference function, solution, gold answer). Their use is therefore also subject to the upstream terms.

FinanceReasoning has two id namespaces, and they are easy to confuse. `question_id` (for example `test-1831`) is the id in the 2,238-row union, and every file here uses it. `source` (for example `FinanceReasoning-test-512`) is the id inside the originating sub-collection, and its number is unrelated to `question_id`.

## 01_clean_250: clean in-library questions

`questions_250.jsonl` holds 250 rows in the FinanceReasoning schema:

| Field | Meaning |
|---|---|
| `question_id`, `source`, `level`, `difficulty` | Ids and difficulty (`level` is `medium` or `hard`) |
| `question`, `context` | What every system sees |
| `function_id`, `article_title`, `function` | The reference function the question is linked to (`article-N`) |
| `python_solution`, `ground_truth` | Per-question solution and gold answer |
| `statistics` | Upstream number, operator and code statistics |

**How it was built** (`scripts/sample_dataset_250.py`, config in `sampling_config.yaml`). The pool is every medium and hard question with a reference function, a gold answer and a difficulty. The script draws 180 medium and 70 hard, stratified by difficulty quartile, seed 42. The medium draw force-includes the 50 pilot questions, so `pilot_50` is a subset. All 250 have distinct `function_id` and distinct `source`.

**Scoring.** A committed number is correct within 1% relative tolerance after output-scale normalization. A refusal, a clarification request or a crash counts as abstention. A wrong committed number is a silent wrong.

## 02_contract_library: the 250-card FIC library

| File | Contents |
|---|---|
| `core.jsonl` | 250 cards: `inputs`, `execution` (the contract's Python code), `diagnostic_checks` (E-checks), `semantic_hints` (I-class ambiguities and their options), `output`, `selection_hints` |
| `retrieval.jsonl` | 250 retrieval records: keywords, negative keywords, selection hints |
| `repair.jsonl` | 1,960 repair rules keyed by `fic_id` and `diagnostic_check_id` |
| `cards.db` | The same three artifacts as the SQLite store the pipeline reads (tables `core_cards`, `retrieval_cards`, `repair_rules`, plus a full-text index) |
| `card_repairs.json` | The build-time lint repairs applied on 2026-07-12. No gold answer is involved |
| `validation_report.json` | Cross-artifact integrity check (0 kernel duplicates, 0 failed smoke tests) |

Card ids are `fic_article_N`, and card `fic_article_N` comes from reference function `article-N`. `cards.db` agrees with `core.jsonl` field by field. This is the type-audited library of 2026-07-12 that every paper number uses.

The cards come from the three-stage build in `verifiquant/preprocessing/` (core, retrieval, repair). Its prompts are printed in App. K. To rebuild a store from the JSONL artifacts, use `card_store.SQLAlchemyArtifactStore(...).ingest_artifacts(...)`, as `scripts/build_subset_libraries.py` does.

## 03_traps_80: confirmatory trap set

`trap_set.jsonl` holds 80 traps, 20 per operator, over 52 base questions from the 250, with at most two operators per base question.

| Operator | What it does | Expected handling | Mix |
|---|---|---|---|
| F | Replaces a required input with the literal token `[REDACTED]` | `refuse_or_ask` | balance 5, count 5, other 5, price 3, denominator 2 |
| E | Injects a value that violates one of the card's own E-checks | `abstain_or_repair` | negativity 8, magnitude 4, non-positive/zero 4, rate out of unit 4 |
| I | Deletes a declared disambiguator | I_HARD: `clarify` (6). I_SOFT: `proceed_with_warning` (14) | |
| N | Replaces the ask with an out-of-scope request | `graceful_refuse` | exotic 14, near-miss 6 |

Each row carries `trap_id`, `base_question_id`, `operator`, `family`, `expected_diagnostic`, `expected_behavior`, the perturbed `question` and `context`, the base question's `function`, `python_solution` and `ground_truth`, and a `grounding` record of exactly what the operator changed. For example, an F trap records `removed_field` and `removed_value`, and an E trap records the E-check `expression`, the `violating_value`, and `verified: true`.

**How it was built.** `verifiquant/pipeline/build_trap_set_v2.py` is deterministic and makes no model call. The label *is* the operator. Every E trap is verified to fire the card's own predicate on the injected value and not on the clean one. `trap_manifest.json` records the sha256 of the inputs (questions, canonical run, `cards.db`), and the packaging script checks them. Only 6 of 40 hard-ambiguity cases had a deletable disambiguator phrase, which caps I_HARD at 6. Report I_HARD and I_SOFT separately.

**Scoring.** The false numeric acceptance rate (FNAR) is the share of traps on which a system commits a number. A number on an I_SOFT trap does not count when the system labels it I_SOFT and attaches a warning. FNAR needs no gold answer. App. D discusses the F traps that remove an argument with a default value (for example `F-test-1263` and `F-test-1559` remove a share count that defaults to 100).

`scripted_user_replies.jsonl` holds the 26 replies of the scripted user (§6, App. B): one per F trap (`"The <field> is <value>."`, the exact removed span) and one per I_HARD trap (`"It is a call option."`, `"Use the FIFO method."`). The replies come from `scripts/run_scripted_user_eval.py`. The runner appends a reply to the context only when the first turn withheld a number. Twenty-five of the 26 were sent in at least one run, and each matches the reply recorded there.

## 04_out_of_library: fixed OOD split

| File | Contents |
|---|---|
| `ood_199.jsonl` | The 199 clean questions whose family has no card in `library_50fam` |
| `id_51.jsonl` | The other 51 (in-library), in `questions_250` order |
| `library_50fam/cards.db` | The library both systems retrieve from in this analysis (53 cards) |
| `library_50fam/fic_ids.json` | Its card ids |

`library_50fam` is the pilot's library: cards for the 50 pilot families plus three more (NPV, NPVGO, Discounted Payback Periods). The Discounted Payback card matches one further clean question (`test-1958`), so 51 questions are in-library. A question is in-library when `fic_article_<N>` for its `function_id` `article-<N>` is in the library. This is the rule in `scripts/run_ood_family_eval.py`.

## 05_coverage_sweep: library-coverage sweep

`subsets.json` lists the card ids in each nested subset of the 250-card library, with the resulting ID and OOD question counts:

| Coverage | Families | ID questions | OOD questions |
|---|---|---|---|
| 10% | 25 | 25 | 225 |
| 25% | 62 | 62 | 188 |
| 50% | 125 | 125 | 125 |
| 75% | 188 | 188 | 62 |
| 100% | 250 | 250 | 0 |

The subsets are one fixed `random.Random(42)` shuffle of the sorted card ids, so each is contained in the next. They equal the libraries the sweep ran on (`results/libsweep/lib_*/cards.db`). To rebuild the SQLite stores:

```bash
PYTHONPATH=. python3 scripts/build_subset_libraries.py --seed 42 --out-dir /tmp/libsweep
```

## 06_human_set_80: human-labeled external-validity set

| File | Contents |
|---|---|
| `messy_pilot.jsonl` | The 80 queries: `id`, `persona`, `intended`, `question`, `context`, `source_id`, `author_model` |
| `messy_pilot_provenance.json` | Per item: source question and `source_gold`. The 18 revised items also have an `authoring_audit` (`omitted_input`, `why_required`, `method`) |
| `messy_pilot_labeling.csv`, `.md` | The blank labeling sheet the annotators received |
| `LABELING_GUIDELINE.md`, `_zh.md` | Annotator instructions (English and Traditional Chinese) |
| `labels/A1.csv` … `A5.csv` | Each annotator's labels and one-line reasons (anonymized, reasons partly in Chinese) |
| `ground_truth.jsonl` | The frozen majority label: `id`, `label`, `votes`, `n_raters` |
| `agreement_report.json`, `disagreement_pool.csv` | Fleiss' κ = 0.835 (on the 79 items all five labeled), 80/80 items reach a majority, so the pool is empty |
| `answer_gold.json` | Numeric gold for the 34 `A` items, each with its formula. Six are flagged `convention_dependent` |
| `README_GOLD.md` | Which gold to use, and why `source_gold` must not be used |
| `unstratified_v1/` | The unstratified first draft (80 queries, never labeled). It shows why the final set is stratified |
| `pre_trim_82q/` | The 82-item annotation record (queries, provenance, five label files, majority, agreement) before `mq-81` and `mq-82` were removed |

**Labels.** Each annotator assigns one action by a top-down rule: `O` (out of scope) if the ask is not to compute one number, else `R` (request info) if a required value is missing, else `C` (clarify) if an interpretation choice would change the number, else `A` (answer). The majority (at least 3 of 5) gives 34 A, 20 R, 10 C and 16 O. The labels were frozen before any system was scored.

**How it was built.**

1. Source questions come from FinanceReasoning easy and medium, excluding the 250 (`scripts/build_messy_query_pilot.py`).
2. GPT-5.2 rewrote each one under a persona and one of four hidden intents: clean 27, missing 18, ambiguous 18, judgment 17. The author never saw our contracts, taxonomy, gold answers or system outputs.
3. All 18 missing-intent drafts had left the context empty, so we revised them offline with an OpenAI Codex agent, seeded by the GPT-5.2 versions. Each revision omits exactly one indispensable number (`scripts/curate_messy_missing_items.py`), and a person read every one. These 18 have `author_model = gpt-5.2_seed_then_codex_curated_v3`. Unlike the other 62, they are not blind to our contracts.
4. Five finance-trained annotators labeled the items, blind to the source, the intent and every system output. `scripts/aggregate_messy_labels.py` produces the majority and the agreement report.
5. `mq-81` and `mq-82` were appended to the sheets by hand, outside the stratified design. We removed them on that structural criterion (`scripts/trim_messy_set_to_80.py`). Both had majority `A`, so no Human-FNAR numerator changes.

**Scoring.** Human-FNAR is the share of the 46 non-`A` items on which a system commits a number. A warned number counts as committed here, which is stricter than the trap metric. Match is the share of the 80 items where a system's action equals the majority label. It is four-way for systems that expose a reason for declining, and answer-versus-abstain for CoT and plain R-PAL. `scripts/score_messy_external.py` computes both. Its `sel_acc` field scores against `source_gold` and is retired (see the warning below). Answer coverage is the share of the 34 `A` items a system answers.

> **Do not score answers with `source_gold`.** It is the answer to the *source* question. The rewrites use fresh numbers, so it disagrees with the true answer on 30 of the 34 `A` items. Use `ground_truth.jsonl` for actions and `answer_gold.json` for numbers.

## 07_repair_mutants: RepairBench

`repairbench_mutants.jsonl` holds one row per transform spec (`kind: valid`, 179 rows) and one per mutant (`kind: mutant`, 906 rows). A mutant row carries `class`, `fic`, `hint`, the mutated `expr`, and the verifier's verdicts `k1` (one sampled binding) and `k5` (five sampled bindings), plus `equivalent` and the rejection `reason`.

| Class | Mutation | Mutants |
|---|---|---|
| `op_swap` | First `*`↔`/` or `+`↔`-` swapped | 171 |
| `coeff_perturb` | First numeric constant doubled | 164 |
| `var_swap` | One input name replaced by another | 34 |
| `identity` | Expression replaced by `result` | 179 |
| `call_inject` | Wrapped in `min(expr, 1e9)` | 179 |
| `oversize` | Padded past `max_expr_nodes` | 179 |

Of the 906 mutants, 47 pass at five bindings. 27 of those are arithmetically equivalent to the original and are excluded, which leaves 20 true false accepts. The verifier therefore rejects 859 of 879 non-equivalent mutants (97.7%). The run makes no model call and reproduces byte for byte:

```bash
PYTHONPATH=. python3 scripts/run_repairbench.py --core datasets/02_contract_library/core.jsonl --seed 0 --out-dir /tmp/repairbench
```

## pilot_50: the pilot

`questions_50.jsonl` holds the 50 medium questions of the first pilot (`scripts/sample_dataset.py`, seed 42). `trap_set_50.jsonl` holds the 50-trap pilot built over them by the earlier generator `verifiquant/pipeline/build_trap_set.py`: 10 each of F, E, I, N and M. M yields no reliable quantitative trap, so 40 are evaluable. The 80-trap set supersedes it. The paper uses it only for the Heston out-of-scope case in App. J.

## Running the paper's evaluations on these files

Each runner below defaults to the paths under `results/`, and the flags shown point it at this folder instead. All of them need `GEMINI_API_KEY`, and all use Gemini 2.5 Flash by default.

| Evaluation | Command |
|---|---|
| VerifiQuant on traps (K=1) | `PYTHONPATH=. python3 scripts/run_vq_trap_rerun.py --dataset datasets/03_traps_80/trap_set.jsonl --db datasets/02_contract_library/cards.db --top-k 4 --out-dir OUT --run-name vq_traps` |
| R-PAL on the clean set | `PYTHONPATH=. python3 scripts/run_retrieval_pal_baseline.py --questions datasets/01_clean_250/questions_250.jsonl --db datasets/02_contract_library/cards.db --top-k 4 --out-dir OUT` |
| R-PAL / R-PAL+ on traps | `PYTHONPATH=. python3 scripts/run_retrieval_pal_governance_eval.py --mode prompted --dataset datasets/03_traps_80/trap_set.jsonl --db datasets/02_contract_library/cards.db --top-k 4 --out-dir OUT` (`--mode plain` for R-PAL) |
| Out-of-library split | `PYTHONPATH=. python3 scripts/run_ood_family_eval.py --lib-db datasets/04_out_of_library/library_50fam/cards.db --questions datasets/01_clean_250/questions_250.jsonl --limit-ood 999 --limit-id 999 --top-k 4 --no-baseline --out-dir OUT` |
| Coverage sweep point | Same as above with `--lib-db` set to a store rebuilt from `05_coverage_sweep` |

The R-PAL runners read the function library from `FinanceReasoning/data/functions/functions-article-all.json` (override with `--function-library`).

## Where the results live

| Result | Directory under `results/` |
|---|---|
| Clean set, VerifiQuant and CoT | `paper_v2_250/results/` |
| Clean set, R-PAL / PAL / R-PAL+ | `retrieval_pal_baseline/`, `pal_baseline/`, `governance_eval/` |
| Traps, VerifiQuant (top-4, three runs) | `governance_confirmatory_1014/` |
| Traps, R-PAL and R-PAL+ | `governance_confirmatory/` |
| Staged controls | `governance_confirmatory_1014/` |
| Scripted user | `scripted_user_1015/` |
| Gemini 2.5 Pro backbone | `governance_confirmatory_1015_pro/` |
| Out-of-library split | `ood_eval_top4/` (July top-3 run in `ood_eval/`) |
| Coverage sweep | `libsweep_top4/` (July top-3 run in `libsweep/`) |
| Human set, system outputs | `messy_query_pilot/agreement/runs/` |
| RepairBench | `repairbench/` |

`scripts/assert_paper_numbers.py` checks the paper's reported numbers against these directories.

## Integrity and rebuilding

`MANIFEST.json` lists every file with its sha256, size, row count and origin. Most files are byte-for-byte copies of the run artifacts. Five are derived by the packaging script, and each is checked against the record it replaces.

```bash
.venv/bin/python scripts/package_datasets.py --force
```

Before it writes anything, the script asserts every count stated above. It also checks the input hashes against the trap manifest and fails if any set has drifted. It never touches this README. Three copied provenance files (`01_clean_250/sampling_config.yaml`, `05_coverage_sweep/build_manifest.json`, `07_repair_mutants/repairbench_summary.json`) recorded absolute paths on the build machine. The script strips the repository prefix from them, so every path reads as relative to the repository root, and `MANIFEST.json` marks these files. With `--zip PATH` it also writes the folder to one zip with fixed timestamps, so a rebuild from the same commit gives identical bytes.

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

@inproceedings{tang2025financereasoning,
  author    = {Tang, Zichen and E, Haihong and Ma, Ziyan and He, Haoyang and Liu, Jiacheng and Yang, Zhongjun and Rong, Zihua and Li, Rongjin and Ji, Kun and Huang, Qing and Hu, Xinyang and Liu, Yang and Zheng, Qianhe},
  title     = {{FinanceReasoning}: Benchmarking Financial Numerical Reasoning More Credible, Comprehensive and Challenging},
  booktitle = {Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages     = {15721--15749},
  year      = {2025}
}
```

## License

Not yet set. Choose one for the files we authored (contract library, traps, human-set queries and labels) before public release. Files that embed FinanceReasoning content also fall under the upstream terms.

The human-set items are synthetic and contain no personal information. Annotators took part voluntarily and are anonymized as A1 to A5. Our institution determined that ethics-committee review was not required.
