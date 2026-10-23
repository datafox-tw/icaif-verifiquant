#!/usr/bin/env python3
"""Wave 1 prep: build stratified subset card libraries for the library-size sweep.

From the pristine 250-card artifacts (paper_v2_250/fic/*.jsonl) build cards.db subsets
at the requested fractions (seeded family sampling), via card_store.ingest_artifacts so
the retrieval FTS stays consistent. Output: <out-dir>/lib_<pct>/cards.db.

The sweep then runs run_ood_family_eval.py against each subset:
  PYTHONPATH=. python3 scripts/run_ood_family_eval.py \
      --lib-db results/libsweep/lib_25/cards.db \
      --limit-ood 999 --limit-id 999 --no-baseline

Usage: PYTHONPATH=. python3 scripts/build_subset_libraries.py [--fractions 10,25,50,75,100]
"""
from __future__ import annotations
import argparse, json, os, random

from verifiquant.card_store import SQLAlchemyArtifactStore

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIC = os.path.join(ROOT, "results/paper_v2_250/fic")


def load_jsonl(name):
    p = os.path.join(FIC, name)
    return [json.loads(l) for l in open(p) if l.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fractions", default="10,25,50,75,100")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results/libsweep"))
    args = ap.parse_args()

    core = load_jsonl("core.jsonl")
    retrieval = load_jsonl("retrieval.jsonl")
    repair = load_jsonl("repair.jsonl")
    fams = sorted({c["fic_id"] for c in core})
    rng = random.Random(args.seed)
    order = fams[:]
    rng.shuffle(order)  # one fixed shuffle → nested subsets (10% ⊂ 25% ⊂ … ⊂ 100%)

    manifest = {"seed": args.seed, "total_families": len(fams), "subsets": {}}
    for frac in [int(x) for x in args.fractions.split(",")]:
        k = max(1, round(len(order) * frac / 100))
        chosen = set(order[:k])
        out = os.path.join(args.out_dir, f"lib_{frac}")
        os.makedirs(out, exist_ok=True)
        db = os.path.join(out, "cards.db")
        if os.path.exists(db):
            os.remove(db)
        store = SQLAlchemyArtifactStore(f"sqlite:///{db}")
        store.ingest_artifacts(
            core_cards=[c for c in core if c["fic_id"] in chosen],
            retrieval_cards=[r for r in retrieval if r.get("fic_id") in chosen],
            repair_rules=[r for r in repair if r.get("fic_id") in chosen],
            validate_relations=False,
        )
        manifest["subsets"][frac] = {"families": k, "db": db}
        print(f"lib_{frac}: {k} families -> {db}")

    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("manifest written; subsets are nested (10% ⊂ 25% ⊂ … ⊂ 100%).")


if __name__ == "__main__":
    main()
