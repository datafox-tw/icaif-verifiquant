"""Offline smoke test: no API key, no network.

Checks that the package imports, that the released 250-contract library loads and retrieves
the same top candidate the paper's recorded run did, and that the loan contract's declared
annuity-due transform passes the verifier and gives the number quoted on the poster.

Run: python3 tests/test_offline_smoke.py   (or: python3 -m pytest tests/)
"""
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from verifiquant.card_store import SQLAlchemyArtifactStore  # noqa: E402
from verifiquant.preprocessing.stage_transform import (  # noqa: E402
    _run_fic_code,
    apply_result_postprocess,
    get_transform_spec_for_choice,
    verify_transform,
)

LIBRARY = ROOT / "datasets/02_contract_library/cards.db"
TRAPS = ROOT / "datasets/03_traps_80/trap_set.jsonl"

MODULES = [
    "verifiquant.card_store",
    "verifiquant.contracts",
    "verifiquant.taxonomy",
    "verifiquant.preprocessing.fic_generation_pipeline",
    "verifiquant.preprocessing.stage_transform",
    "verifiquant.pipeline.run_error_classification_pipeline",
    "verifiquant.pipeline.run_framework_guided_self_improve_pipeline",
    "verifiquant.pipeline.run_cot_self_improve_pipeline",
    "verifiquant.pipeline.build_trap_set_v2",
]


def store() -> SQLAlchemyArtifactStore:
    return SQLAlchemyArtifactStore(f"sqlite:///{LIBRARY}")


def test_imports() -> None:
    for name in MODULES:
        importlib.import_module(name)


def test_library_loads() -> None:
    assert len(store().load_core_by_id()) == 250


def test_retrieval_matches_recorded_run() -> None:
    # Trap E-test-1080: the recorded VerifiQuant run retrieved fic_article_1441 (Rate of Change) first.
    trap = next(json.loads(l) for l in TRAPS.open(encoding="utf-8") if '"E-test-1080"' in l)
    rows = store().retrieve_candidates(query=f"{trap['question']}\n{trap['context']}", top_k=4)
    assert [r["fic_id"] for r in rows][0] == "fic_article_1441", rows


def test_loan_transform_verifies() -> None:
    s = store()
    core = s.load_core_by_id()["fic_article_2164"]
    rule = next(r for r in s.load_repair_rules()
                if r["fic_id"] == "fic_article_2164" and r["rule_id"] == "i_payment_timing_soft")
    spec = get_transform_spec_for_choice(rule, "beginning_of_period")
    inputs = {"principal": 250000, "annual_interest_rate": 0.06, "loan_term_years": 30}
    result = verify_transform(spec, core, [inputs])
    assert result.passed, result.error
    old = _run_fic_code(core["execution"]["code"], core["execution"]["entrypoint"], inputs)
    new = apply_result_postprocess(old, spec, inputs)
    assert (round(old, 2), round(new, 2)) == (1498.88, 1491.42), (old, new)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all offline checks pass")
