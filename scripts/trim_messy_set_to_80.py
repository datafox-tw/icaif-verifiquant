"""
trim_messy_set_to_80.py
-----------------------
One-shot, idempotent maintenance script: drop the two off-design items (mq-81, mq-82)
from the messy-query external-validity set so the released set is the 80 items produced
by the stratified v3 authoring run.

Why these two and only these two: every in-design item carries an `intended` condition
field in messy_pilot_provenance.json (clean / missing / ambiguous / judgment). mq-81 and
mq-82 have no `intended` field, no source_context from the stratified author call, and do
not appear in natural_v1/messy_pilot.jsonl (the 80-item authored set). They were appended
to the annotator sheets by hand and are outside the sampling design. The criterion is
label-independent: it never looks at an annotation or a system output.

Originals are archived under `_pre_trim_82q/` before anything is rewritten.

Run:  python3 scripts/trim_messy_set_to_80.py
Then: python3 scripts/aggregate_messy_labels.py
      python3 scripts/score_messy_external.py
"""
from __future__ import annotations

import csv
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "results/messy_query_pilot"
ARCHIVE = DIR / "_pre_trim_82q"
DROP = {"mq-81", "mq-82"}

# every file that is keyed by item id
JSONL_BY_ID = [
    DIR / "messy_pilot.jsonl",
    DIR / "ground_truth.jsonl",
]
JSONL_BY_CASE_ID = [
    DIR / "agreement/runs/cot_single_shot.jsonl",
    DIR / "agreement/runs/rpal_plain.jsonl",
    DIR / "agreement/runs/rpal_plus.jsonl",
    DIR / "agreement/runs/vq_k1.jsonl",
]
CSV_BY_ID = [
    DIR / "messy_pilot_labeling.csv",
    DIR / "labels/A1.csv",
    DIR / "labels/A2.csv",
    DIR / "labels/A3.csv",
    DIR / "labels/A4.csv",
    DIR / "labels/A5.csv",
]


def archive() -> None:
    if ARCHIVE.exists():
        print(f"[skip] archive already exists: {ARCHIVE}")
        return
    ARCHIVE.mkdir(parents=True)
    for rel in [
        "messy_pilot.jsonl",
        "messy_pilot_provenance.json",
        "messy_pilot_labeling.csv",
        "messy_pilot_labeling.md",
        "ground_truth.jsonl",
        "agreement_report.json",
        "disagreement_pool.csv",
        "labels",
        "agreement",
    ]:
        src = DIR / rel
        if not src.exists():
            continue
        dst = ARCHIVE / rel
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    (ARCHIVE / "README.md").write_text(
        "# Pre-trim snapshot (82 items)\n\n"
        "Verbatim copy of the messy-query external-validity artifacts as they stood when\n"
        "annotation ran, before `mq-81` and `mq-82` were removed. Those two items were\n"
        "appended to the annotator sheets by hand and are outside the stratified v3\n"
        "authoring design: they carry no `intended` condition in the provenance file and do\n"
        "not appear in `natural_v1/messy_pilot.jsonl`. The exclusion criterion is structural\n"
        "and label-independent. Retained here so the full annotation record stays auditable;\n"
        "the released and reported set is the 80 in-design items.\n\n"
        "See `scripts/trim_messy_set_to_80.py`.\n",
        encoding="utf-8",
    )
    print(f"[archive] wrote {ARCHIVE}")


def trim_jsonl(path: Path, key: str) -> None:
    if not path.exists():
        print(f"[skip] missing {path}")
        return
    kept, dropped = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if json.loads(line)[key] in DROP:
            dropped += 1
            continue
        kept.append(line)
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    print(f"[jsonl ] {path.name}: kept {len(kept)}, dropped {dropped}")


def trim_csv(path: Path) -> None:
    if not path.exists():
        print(f"[skip] missing {path}")
        return
    # utf-8-sig round-trips the BOM some sheets carry; keep whatever the file had
    raw = path.read_bytes()
    enc = "utf-8-sig" if raw.startswith(b"\xef\xbb\xbf") else "utf-8"
    with open(path, encoding=enc, newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    kept = [r for r in body if not (r and r[0].strip() in DROP)]
    with open(path, "w", encoding=enc, newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(kept)
    print(f"[csv   ] {path.name}: kept {len(kept)}, dropped {len(body) - len(kept)}")


def trim_provenance() -> None:
    path = DIR / "messy_pilot_provenance.json"
    recs = json.loads(path.read_text(encoding="utf-8"))
    kept = [r for r in recs if r["id"] not in DROP]
    path.write_text(json.dumps(kept, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[json  ] {path.name}: kept {len(kept)}, dropped {len(recs) - len(kept)}")


def trim_labeling_md() -> None:
    """The sheet has a checklist table (| mq-NN | | |) then one block per item."""
    path = DIR / "messy_pilot_labeling.md"
    text = path.read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines()
             if not re.match(r"^\|\s*(mq-81|mq-82)\s*\|", ln)]
    text = "\n".join(lines)
    # blocks start at "**mq-NN**"; cut from the first dropped id to EOF (they are last)
    m = re.search(r"\n\*\*mq-81\*\*", text)
    if m:
        text = text[: m.start()].rstrip() + "\n"
    path.write_text(text, encoding="utf-8")
    print(f"[md    ] {path.name}: trimmed")


def verify() -> None:
    ids = [json.loads(l)["id"] for l in (DIR / "messy_pilot.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(ids) == 80, f"dataset has {len(ids)} items, expected 80"
    assert not (set(ids) & DROP)
    prov = json.loads((DIR / "messy_pilot_provenance.json").read_text(encoding="utf-8"))
    assert len(prov) == 80
    assert all(r.get("intended") for r in prov), "an in-design item is missing `intended`"
    from collections import Counter
    print(f"[verify] 80 items, conditions = {dict(Counter(r['intended'] for r in prov))}")


def main() -> None:
    archive()
    for p in JSONL_BY_ID:
        trim_jsonl(p, "id")
    for p in JSONL_BY_CASE_ID:
        trim_jsonl(p, "case_id")
    for p in CSV_BY_ID:
        trim_csv(p)
    trim_provenance()
    trim_labeling_md()
    verify()


if __name__ == "__main__":
    main()
