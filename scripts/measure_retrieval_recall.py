"""
measure_retrieval_recall.py
---------------------------
Offline retrieval diagnostics. No LLM calls, no system reruns, no reported number changes.

Answers the review's top-k question: is retrieval@4 leaving recall on the table
versus larger k?

Gold linkage, and a trap to avoid. `core_cards.source_meta_json` carries BOTH
`question_id` and `source`, and they are different namespaces:

  * `question_id` (e.g. test-1831) is the function-linked benchmark question. Its
    `function_id` matches the card's `function_id` exactly, so this is the correct
    key for a gold-contract lookup.
  * `source` (e.g. FinanceReasoning-test-512) is a higher-level, more abstract
    reference. Those ids often sit in the easy split and carry no `function_id`.

The messy set's `source_id` lives in the `source` namespace, not the `question_id`
one. Comparing it against `question_id` yields a spurious zero overlap. Comparing
it against `source` yields 13/80, but source overlap is a poor proxy for whether a
card's formula actually applies, and VerifiQuant's answer rate does not separate on
it (15.4% vs 13.4%, n=13 vs 67). We therefore do NOT use this script to attribute
the messy-set shortfall to library coverage; it measures retriever quality only, on
the clean set where the gold contract is in-library by construction.

Query construction matches the runs: "question + newline + context".
"""
from __future__ import annotations
import json, sqlite3, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from verifiquant.card_store import SQLAlchemyArtifactStore  # noqa: E402

DB = ROOT / "results/paper_v2_250/fic/cards.db"
Q250 = ROOT / "results/paper_v2_250/questions_250.jsonl"
MESSY = ROOT / "results/messy_query_pilot/messy_pilot.jsonl"
PROV = ROOT / "results/messy_query_pilot/messy_pilot_provenance.json"
FR = ROOT / "FinanceReasoning/data/FinanceReasoning"
KS = (1, 2, 4, 8, 16, 32)


def library_maps():
    con = sqlite3.connect(DB)
    fn2fic, fic_fns = {}, set()
    for fic_id, sm in con.execute("select fic_id, source_meta_json from core_cards"):
        meta = json.loads(sm or "{}")
        f = meta.get("function_id")
        if f:
            fn2fic[f] = fic_id
            fic_fns.add(f)
    return fn2fic, fic_fns


def recall_at_k(store, items, fn2fic, label):
    """items: list of (query_text, gold_function_id)."""
    maxk = max(KS)
    hits = {k: 0 for k in KS}
    ranks = []
    scored = 0
    for query, gold_fn in items:
        gold_fic = fn2fic.get(gold_fn)
        if gold_fic is None:
            continue
        scored += 1
        cands = store.retrieve_candidates(query=query, top_k=maxk)
        ids = [c.get("fic_id") for c in cands]
        pos = ids.index(gold_fic) + 1 if gold_fic in ids else None
        ranks.append(pos)
        for k in KS:
            if pos is not None and pos <= k:
                hits[k] += 1
    print(f"\n=== {label}  (n={scored}) ===")
    if not scored:
        print("  no scorable items")
        return
    for k in KS:
        print(f"  recall@{k:<2} = {hits[k]/scored:6.1%}   ({hits[k]}/{scored})")
    found = [r for r in ranks if r]
    print(f"  gold never in top-{maxk}: {scored-len(found)}/{scored}")
    if found:
        med = sorted(found)[len(found) // 2]
        print(f"  median rank when found: {med}")


def main():
    fn2fic, lib_fns = library_maps()
    store = SQLAlchemyArtifactStore(f"sqlite:///{DB}")

    # --- 1. clean 250Q: gold is in-library by construction ---
    clean = []
    for line in open(Q250):
        r = json.loads(line)
        q = f"{r['question']}\n{r.get('context','')}"
        clean.append((q, r["function_id"]))
    recall_at_k(store, clean, fn2fic, "CLEAN 250Q (gold in-library by construction)")

    print("\n=== Note on the messy set ===")
    print("  The messy set carries `source_id`, which is in the `source` namespace,")
    print("  not the function-linked `question_id` namespace, so it has no reliable")
    print("  gold-contract key. Source-level overlap with the library is 13/80, but")
    print("  VerifiQuant's answer rate does not separate on it (15.4% vs 13.4%,")
    print("  n=13 vs 67), so we make no library-coverage claim from that overlap.")


if __name__ == "__main__":
    main()
