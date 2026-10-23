#!/usr/bin/env python3
"""
package_datasets.py
-------------------
Assemble datasets/, the release copy of every dataset the ICAIF '26 camera-ready paper uses,
from the experiment record under results/. Nothing here calls a model.

    datasets/
      00_source/            checksums of the FinanceReasoning files everything is drawn from
      01_clean_250/         the 250 clean in-library questions (180 medium + 70 hard, seed 42)
      02_contract_library/  the 250-card FIC library (JSONL artifacts + the SQLite store)
      03_traps_80/          the balanced 80-trap confirmatory set, plus the 26 scripted replies
      04_out_of_library/    the fixed 199 OOD / 51 ID split and its 50-family library
      05_coverage_sweep/    the nested library subsets at 10/25/50/75/100%
      06_human_set_80/      the human-labeled set: queries, provenance, 5 label files, majority
      07_repair_mutants/    the 906 RepairBench mutants and their verifier verdicts
      pilot_50/             the 50-question pilot and its 50-trap set (appendix case only)
      MANIFEST.json         sha256, size, row count and origin of every file

Most files are byte-for-byte copies of the run artifacts the paper's numbers come from. Five are
derived here, and each derivation is checked against the record it replaces:

  - 00_source/checksums.sha256          sha256 of the local FinanceReasoning files
  - 03_traps_80/scripted_user_replies   the scripted user's one reply per F / I_HARD trap,
                                        from run_scripted_user_eval.user_reply; must equal
                                        every reply recorded in runs/scripted_user_1015/
  - 04_out_of_library/id_51.jsonl       the 250 minus the 199 OOD questions
  - 04_out_of_library/library_50fam/fic_ids.json
  - 05_coverage_sweep/subsets.json      the fic_ids in each runs/libsweep/lib_*/cards.db; must
                                        equal the seed-42 nested shuffle of
                                        scripts/build_subset_libraries.py

Every count the paper states about these sets is asserted before anything is written, and the
input hashes are checked against the trap-set manifest. The script fails instead of packaging a
set that drifted. Three copied provenance files record absolute paths on the build machine; the
repository prefix is stripped from them, and MANIFEST.json says which. datasets/README.md is
written by hand and is never touched.

--zip writes the finished folder to one zip (for the ACM supplementary-material upload), under a
wrapper folder named after the zip, with fixed timestamps so a rebuild gives identical bytes.

Usage: .venv/bin/python scripts/package_datasets.py [--force] [--zip taps_upload/<tag>/icaif26-13-datasets.zip]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sqlite3
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results"
FIC = RUNS / "paper_v2_250/fic"
HUMAN = RUNS / "messy_query_pilot"
SRC = ROOT / "FinanceReasoning/data"
OUT = ROOT / "datasets"

LOCAL_PREFIX = f"{ROOT}/".encode()
TEXT_SUFFIXES = {".json", ".jsonl", ".yaml", ".md", ".csv", ".txt"}
ZIP_DATE = (2026, 10, 18, 0, 0, 0)

MANAGED = ["00_source", "01_clean_250", "02_contract_library", "03_traps_80", "04_out_of_library",
           "05_coverage_sweep", "06_human_set_80", "07_repair_mutants", "pilot_50", "MANIFEST.json"]

SOURCE_FILES = [
    "FinanceReasoning/easy.json",
    "FinanceReasoning/medium.json",
    "FinanceReasoning/hard.json",
    "functions/functions-article-all.json",
    "documents/financial_documents.json",
]

COPIES = {
    "01_clean_250/questions_250.jsonl": RUNS / "paper_v2_250/questions_250.jsonl",
    "01_clean_250/sampling_config.yaml": RUNS / "paper_v2_250/experiment_config.yaml",
    "02_contract_library/core.jsonl": FIC / "core.jsonl",
    "02_contract_library/retrieval.jsonl": FIC / "retrieval.jsonl",
    "02_contract_library/repair.jsonl": FIC / "repair.jsonl",
    "02_contract_library/cards.db": FIC / "cards.db",
    "02_contract_library/card_repairs.json": FIC / "card_repairs.json",
    "02_contract_library/validation_report.json": FIC / "validation_report.json",
    "03_traps_80/trap_set.jsonl": RUNS / "governance_confirmatory/trap_set.jsonl",
    "03_traps_80/trap_manifest.json": RUNS / "governance_confirmatory/trap_manifest.json",
    "04_out_of_library/ood_199.jsonl": RUNS / "ood_eval/ood199_questions.jsonl",
    "04_out_of_library/library_50fam/cards.db": RUNS / "paper_v1/fic/cards.db",
    "05_coverage_sweep/build_manifest.json": RUNS / "libsweep/manifest.json",
    "06_human_set_80/messy_pilot.jsonl": HUMAN / "messy_pilot.jsonl",
    "06_human_set_80/messy_pilot_provenance.json": HUMAN / "messy_pilot_provenance.json",
    "06_human_set_80/messy_pilot_labeling.csv": HUMAN / "messy_pilot_labeling.csv",
    "06_human_set_80/messy_pilot_labeling.md": HUMAN / "messy_pilot_labeling.md",
    "06_human_set_80/LABELING_GUIDELINE.md": HUMAN / "LABELING_GUIDELINE.md",
    "06_human_set_80/LABELING_GUIDELINE_zh.md": HUMAN / "LABELING_GUIDELINE_zh.md",
    "06_human_set_80/ground_truth.jsonl": HUMAN / "ground_truth.jsonl",
    "06_human_set_80/agreement_report.json": HUMAN / "agreement_report.json",
    "06_human_set_80/disagreement_pool.csv": HUMAN / "disagreement_pool.csv",
    "06_human_set_80/answer_gold.json": HUMAN / "answer_gold.json",
    "06_human_set_80/README_GOLD.md": HUMAN / "README_GOLD.md",
    **{f"06_human_set_80/labels/A{i}.csv": HUMAN / f"labels/A{i}.csv" for i in range(1, 6)},
    "06_human_set_80/unstratified_v1/messy_pilot.jsonl": HUMAN / "natural_v1/messy_pilot.jsonl",
    "06_human_set_80/unstratified_v1/messy_pilot_provenance.json": HUMAN / "natural_v1/messy_pilot_provenance.json",
    "06_human_set_80/pre_trim_82q/README.md": HUMAN / "_pre_trim_82q/README.md",
    "06_human_set_80/pre_trim_82q/messy_pilot.jsonl": HUMAN / "_pre_trim_82q/messy_pilot.jsonl",
    "06_human_set_80/pre_trim_82q/messy_pilot_provenance.json": HUMAN / "_pre_trim_82q/messy_pilot_provenance.json",
    "06_human_set_80/pre_trim_82q/ground_truth.jsonl": HUMAN / "_pre_trim_82q/ground_truth.jsonl",
    "06_human_set_80/pre_trim_82q/agreement_report.json": HUMAN / "_pre_trim_82q/agreement_report.json",
    **{f"06_human_set_80/pre_trim_82q/labels/A{i}.csv": HUMAN / f"_pre_trim_82q/labels/A{i}.csv" for i in range(1, 6)},
    "07_repair_mutants/repairbench_mutants.jsonl": RUNS / "repairbench/repairbench_20260719_184634.jsonl",
    "07_repair_mutants/repairbench_summary.json": RUNS / "repairbench/repairbench_summary_20260719_184634.json",
    "pilot_50/questions_50.jsonl": RUNS / "paper_v1/questions_50.jsonl",
    "pilot_50/sampling_config.yaml": RUNS / "paper_v1/experiment_config.yaml",
    "pilot_50/trap_set_50.jsonl": RUNS / "paper_v1/trap/trap_set.jsonl",
    "pilot_50/trap_manifest_50.json": RUNS / "paper_v1/trap/trap_manifest.json",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fic_ids(db: Path) -> list[str]:
    return [r[0] for r in sqlite3.connect(db).execute("SELECT fic_id FROM core_cards ORDER BY fic_id")]


def family(question: dict) -> str:
    """Same rule as scripts/run_ood_family_eval.py: article-N is in-library iff fic_article_N is."""
    return str(question.get("function_id", "")).replace("article-", "")


def split_by_library(questions: list[dict], library: list[str]) -> tuple[list[dict], list[dict]]:
    lib = {f.replace("fic_article_", "") for f in library}
    return ([q for q in questions if family(q) in lib], [q for q in questions if family(q) not in lib])


def rows_of(path: Path) -> int | None:
    if path.suffix == ".jsonl":
        return sum(1 for line in path.open(encoding="utf-8") if line.strip())
    if path.suffix == ".csv":
        return sum(1 for _ in csv.DictReader(path.open(encoding="utf-8")))
    if path.suffix == ".db":
        return sqlite3.connect(path).execute("SELECT COUNT(*) FROM core_cards").fetchone()[0]
    return None


# ── checks ──────────────────────────────────────────────────────────────────────────────────────

def check_clean_and_library(q250: list[dict]) -> None:
    assert len(q250) == 250, len(q250)
    assert Counter(q["level"] for q in q250) == {"medium": 180, "hard": 70}
    assert all(q.get("function_id") for q in q250), "a clean question has no function_id"
    assert len({q["function_id"] for q in q250}) == 250 and len({q["source"] for q in q250}) == 250

    core, retrieval, repair = (jsonl(FIC / f) for f in ("core.jsonl", "retrieval.jsonl", "repair.jsonl"))
    assert (len(core), len(retrieval), len(repair)) == (250, 250, 1960)
    con = sqlite3.connect(FIC / "cards.db")
    cols = [r[1] for r in con.execute("PRAGMA table_info(core_cards)")]
    db = {r[0]: dict(zip(cols, r)) for r in con.execute("SELECT * FROM core_cards")}
    assert set(db) == {c["fic_id"] for c in core}, "cards.db and core.jsonl hold different cards"
    for card in core:
        for field in ("inputs", "execution", "diagnostic_checks", "semantic_hints", "output", "selection_hints"):
            assert json.loads(db[card["fic_id"]][f"{field}_json"] or "null") == card.get(field), \
                f"cards.db disagrees with core.jsonl on {card['fic_id']}.{field}"
    assert con.execute("SELECT COUNT(*) FROM repair_rules").fetchone()[0] == 1960
    # every clean question has its own card: the evaluation is in-library by construction
    in_lib, out_lib = split_by_library(q250, list(db))
    assert len(in_lib) == 250 and not out_lib


def check_traps(q250: list[dict]) -> list[dict]:
    manifest = json.loads((RUNS / "governance_confirmatory/trap_manifest.json").read_text())
    for key, path in (("questions", RUNS / "paper_v2_250/questions_250.jsonl"), ("cards_db", FIC / "cards.db")):
        assert sha256(path) == manifest["sources"][key]["sha256"], f"{key} changed since the traps were built"
    traps_path = RUNS / "governance_confirmatory/trap_set.jsonl"
    assert sha256(traps_path) == manifest["trap_set_sha256"], "trap_set.jsonl changed since its manifest"
    traps = jsonl(traps_path)
    assert len(traps) == 80 and Counter(t["operator"] for t in traps) == {"F": 20, "E": 20, "I": 20, "N": 20}
    assert Counter(t["expected_diagnostic"] for t in traps if t["operator"] == "I") == {"I_HARD": 6, "I_SOFT": 14}
    bases = Counter(t["base_question_id"] for t in traps)
    assert len(bases) == 52 and max(bases.values()) <= 2
    assert set(bases) <= {q["question_id"] for q in q250}
    assert not any(t["needs_review"] for t in traps)
    return traps


def scripted_replies(traps: list[dict], q250: list[dict]) -> list[dict]:
    sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
    from run_scripted_user_eval import user_reply  # imports the runner's deps; no model call
    base = {q["question_id"]: q for q in q250}
    rows = []
    for t in traps:
        if not (t["operator"] == "F" or t["expected_diagnostic"] == "I_HARD"):
            continue
        g = t["grounding"]
        rows.append({
            "trap_id": t["trap_id"],
            "operator": t["operator"],
            "expected_diagnostic": t["expected_diagnostic"],
            "removed": g["removed_field"] if t["operator"] == "F" else g["removed_phrase"],
            "reply": user_reply(t, base[t["base_question_id"]]),
        })
    assert len(rows) == 26 and Counter(r["operator"] for r in rows) == {"F": 20, "I": 6}
    recorded: dict[str, set[str]] = {}
    for path in sorted((RUNS / "scripted_user_1015").glob("*_run[123].jsonl")):
        for r in jsonl(path):
            if r.get("reply"):
                recorded.setdefault(r["trap_id"], set()).add(r["reply"])
    mine = {r["trap_id"]: r["reply"] for r in rows}
    assert recorded and all(v == {mine[k]} for k, v in recorded.items()), "scripted replies drifted from E6"
    return rows


def check_ood(q250: list[dict]) -> tuple[list[dict], list[str]]:
    ood = jsonl(RUNS / "ood_eval/ood199_questions.jsonl")
    library = fic_ids(RUNS / "paper_v1/fic/cards.db")
    in_lib, out_lib = split_by_library(q250, library)
    assert len(library) == 53, len(library)
    assert {q["question_id"] for q in out_lib} == {q["question_id"] for q in ood}, "OOD split drifted"
    assert len(ood) == 199 and len(in_lib) == 51
    pilot = {q["question_id"] for q in jsonl(RUNS / "paper_v1/questions_50.jsonl")}
    assert pilot <= {q["question_id"] for q in in_lib}, "a pilot question is not in-library"
    return in_lib, library


def sweep_subsets(q250: list[dict]) -> dict:
    families = sorted(c["fic_id"] for c in jsonl(FIC / "core.jsonl"))
    order = families[:]
    random.Random(42).shuffle(order)  # scripts/build_subset_libraries.py, --seed 42
    subsets, prev = {}, set()
    expected = {10: (25, 25), 25: (62, 62), 50: (125, 125), 75: (188, 188), 100: (250, 250)}
    for frac, (n_fam, n_id) in expected.items():
        on_disk = fic_ids(RUNS / f"libsweep/lib_{frac}/cards.db")
        assert sorted(order[: max(1, round(len(order) * frac / 100))]) == on_disk, f"lib_{frac} is not the seed-42 subset"
        assert len(on_disk) == n_fam and prev <= set(on_disk)
        in_lib, out_lib = split_by_library(q250, on_disk)
        assert len(in_lib) == n_id
        subsets[str(frac)] = {"n_families": n_fam, "n_id_questions": len(in_lib),
                              "n_ood_questions": len(out_lib), "fic_ids": on_disk}
        prev = set(on_disk)
    return {"seed": 42, "rule": "one fixed random.Random(42) shuffle of the 250 sorted fic_ids; the first "
                                "round(250*pct/100) form each subset, so 10% ⊂ 25% ⊂ 50% ⊂ 75% ⊂ 100%",
            "split_rule": "a question is ID iff fic_article_<N> for its function_id article-<N> is in the subset",
            "subsets": subsets}


def check_human(q250: list[dict]) -> None:
    items = jsonl(HUMAN / "messy_pilot.jsonl")
    assert len(items) == 80
    assert Counter(i["intended"] for i in items) == {"clean": 27, "missing": 18, "ambiguous": 18, "judgment": 17}
    assert Counter(i["author_model"] for i in items) == {"gpt-5.2": 62, "gpt-5.2_seed_then_codex_curated_v3": 18}
    assert all(i["author_model"] == "gpt-5.2" or i["intended"] == "missing" for i in items)
    assert not {i["source_id"] for i in items} & {q["question_id"] for q in q250}, "a human-set source is in the 250"
    provenance = json.loads((HUMAN / "messy_pilot_provenance.json").read_text())
    assert [p["id"] for p in provenance] == [i["id"] for i in items]

    gt = jsonl(HUMAN / "ground_truth.jsonl")
    assert [g["id"] for g in gt] == [i["id"] for i in items]
    assert Counter(g["label"] for g in gt) == {"A": 34, "R": 20, "C": 10, "O": 16}
    for k in range(1, 6):
        sheet = list(csv.DictReader((HUMAN / f"labels/A{k}.csv").open(encoding="utf-8")))
        assert [r["id"] for r in sheet] == [i["id"] for i in items], f"A{k}.csv rows differ from the 80"
    gold = json.loads((HUMAN / "answer_gold.json").read_text())
    assert set(gold) == {g["id"] for g in gt if g["label"] == "A"}, "answer gold does not cover exactly the 34 A items"

    pre = jsonl(HUMAN / "_pre_trim_82q/messy_pilot.jsonl")
    assert len(pre) == 82 and {p["id"] for p in pre} - {i["id"] for i in items} == {"mq-81", "mq-82"}
    assert len(jsonl(HUMAN / "natural_v1/messy_pilot.jsonl")) == 80


def check_mutants() -> None:
    rows = jsonl(RUNS / "repairbench/repairbench_20260719_184634.jsonl")
    summary = json.loads((RUNS / "repairbench/repairbench_summary_20260719_184634.json").read_text())
    mutants = [r for r in rows if r["kind"] == "mutant"]
    passed = [r for r in mutants if r["k5"]]
    equivalent = sum(1 for r in passed if r["equivalent"])
    assert sum(1 for r in rows if r["kind"] == "valid") == 179 == summary["usable"]
    assert (len(mutants), len(passed), equivalent) == (906, 47, 27)
    assert len(passed) - equivalent == summary["true_false_accept_k5_total"] == 20
    assert summary["core"].endswith("paper_v2_250/fic/core.jsonl") and summary["seed"] == 0


def check_pilot(q250: list[dict]) -> None:
    pilot = jsonl(RUNS / "paper_v1/questions_50.jsonl")
    assert len(pilot) == 50 and {q["question_id"] for q in pilot} <= {q["question_id"] for q in q250}
    traps = jsonl(RUNS / "paper_v1/trap/trap_set.jsonl")
    assert Counter(t["operator"] for t in traps) == {o: 10 for o in "FEINM"}


# ── assembly ────────────────────────────────────────────────────────────────────────────────────

def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="replace the managed parts of an existing datasets/")
    ap.add_argument("--zip", type=Path, help="also write datasets/ to this zip, under a folder named after it")
    args = ap.parse_args()

    existing = [name for name in MANAGED if (OUT / name).exists()]
    if existing and not args.force:
        sys.exit(f"datasets/ already holds {existing}; rerun with --force to rebuild them")

    missing = [str(p.relative_to(ROOT)) for p in COPIES.values() if not p.exists()]
    missing += [f"FinanceReasoning/data/{f}" for f in SOURCE_FILES if not (SRC / f).exists()]
    assert not missing, f"missing inputs: {missing}"

    q250 = jsonl(RUNS / "paper_v2_250/questions_250.jsonl")
    check_clean_and_library(q250)
    traps = check_traps(q250)
    replies = scripted_replies(traps, q250)
    id51, library50 = check_ood(q250)
    sweep = sweep_subsets(q250)
    check_human(q250)
    check_mutants()
    check_pilot(q250)
    print("all checks passed")

    for name in existing:
        target = OUT / name
        shutil.rmtree(target) if target.is_dir() else target.unlink()

    origin: dict[str, str] = {}
    for rel, src in COPIES.items():
        dest = OUT / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        origin[rel] = str(src.relative_to(ROOT))
        data = src.read_bytes()
        if src.suffix in TEXT_SUFFIXES and LOCAL_PREFIX in data:
            # a few provenance files record absolute paths on the build machine; keep them repo-relative
            data = data.replace(LOCAL_PREFIX, b"")
            origin[rel] += " (absolute repository prefix stripped)"
        dest.write_bytes(data)
    leaked = [str(p.relative_to(OUT)) for p in OUT.rglob("*")
              if p.is_file() and p.suffix in TEXT_SUFFIXES and str(ROOT).encode() in p.read_bytes()]
    assert not leaked, f"absolute local paths left in {leaked}"

    sums = [(sha256(SRC / f), f) for f in SOURCE_FILES]
    (OUT / "00_source").mkdir(parents=True, exist_ok=True)
    (OUT / "00_source/checksums.sha256").write_text("".join(f"{h}  {f}\n" for h, f in sums))
    origin["00_source/checksums.sha256"] = "derived: sha256 of FinanceReasoning/data/* (files not redistributed)"

    write_jsonl(OUT / "03_traps_80/scripted_user_replies.jsonl", replies)
    origin["03_traps_80/scripted_user_replies.jsonl"] = \
        "derived: scripts/run_scripted_user_eval.py:user_reply over the 20 F + 6 I_HARD traps"
    write_jsonl(OUT / "04_out_of_library/id_51.jsonl", id51)
    origin["04_out_of_library/id_51.jsonl"] = "derived: questions_250 whose family has a card in library_50fam"
    write_json(OUT / "04_out_of_library/library_50fam/fic_ids.json", library50)
    origin["04_out_of_library/library_50fam/fic_ids.json"] = "derived: fic_id column of library_50fam/cards.db"
    write_json(OUT / "05_coverage_sweep/subsets.json", sweep)
    origin["05_coverage_sweep/subsets.json"] = \
        "derived: fic_ids of results/libsweep/lib_*/cards.db (checked against the seed-42 shuffle)"

    files = []
    for path in sorted(p for p in OUT.rglob("*") if p.is_file()):
        rel = str(path.relative_to(OUT))
        if rel in ("README.md", "MANIFEST.json") or path.name == ".DS_Store":
            continue
        files.append({"path": rel, "sha256": sha256(path), "bytes": path.stat().st_size,
                      "rows": rows_of(path), "origin": origin[rel]})
    write_json(OUT / "MANIFEST.json", {
        "git_commit": git_commit(),
        "builder": "scripts/package_datasets.py",
        "source": {"repository": "https://github.com/BUPT-Reasoning/FinanceReasoning",
                   "commit": "d444bedf30f9d37e71df356bffe60aee2a994379",
                   "files": [{"path": f"data/{f}", "sha256": h} for h, f in sums]},
        "files": files,
    })
    total = sum(f["bytes"] for f in files)
    print(f"wrote {len(files)} files ({total / 1e6:.1f} MB) to {OUT.relative_to(ROOT)}/")
    if args.zip:
        write_zip(args.zip, [f["path"] for f in files] + ["MANIFEST.json", "README.md"])


def write_zip(zpath: Path, members: list[str]) -> None:
    """Zip datasets/ under one wrapper folder, with fixed timestamps so a rebuild gives the same bytes."""
    zpath = zpath.resolve()
    zpath.parent.mkdir(parents=True, exist_ok=True)
    top = zpath.stem
    with zipfile.ZipFile(zpath, "w") as z:
        for rel in sorted(members):
            info = zipfile.ZipInfo(f"{top}/{rel}", date_time=ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, (OUT / rel).read_bytes(), compresslevel=9)
    with zipfile.ZipFile(zpath) as z:
        assert z.testzip() is None
        names = z.namelist()
        assert len(names) == len(members) and all(n.startswith(f"{top}/") for n in names)
        for rel in members:
            assert z.read(f"{top}/{rel}") == (OUT / rel).read_bytes(), f"{rel} differs inside the zip"
    shown = zpath.relative_to(ROOT) if zpath.is_relative_to(ROOT) else zpath
    print(f"zip: {shown} ({zpath.stat().st_size / 1e6:.1f} MB, {len(names)} files under {top}/)")


if __name__ == "__main__":
    main()
