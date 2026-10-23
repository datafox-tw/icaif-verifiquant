"""
aggregate_messy_labels.py
-------------------------
Aggregate the finance-student annotations for the messy-query external-validity set.

Input : a folder of per-annotator CSVs (one file per person), each with at least
        columns `id` and `label`; `reason` optional. Label cells may be full words
        (ANSWER/REQUEST-INFO/CLARIFY/OUT-OF-SCOPE) or single letters (A/R/C/O),
        optionally suffixed with '?'. Blank / lone '?' = no decision.
        Default folder: results/messy_query_pilot/labels/

Output (offline, no model calls):
  - ground_truth.jsonl        : id, majority label (>=ceil(n/2)+ decision), votes, n_raters
  - disagreement_pool.csv     : items with no majority (you adjudicate these)
  - agreement_report.json     : per-rater coverage, Fleiss' kappa, majority-rate

Freeze `ground_truth.jsonl` (plus your adjudications of the pool) BEFORE running any
system, then score VQ/R-PAL/CoT against it.
"""
from __future__ import annotations
import csv, glob, json, math, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "results/messy_query_pilot"
CATS = ["A", "R", "C", "O"]
_MAP = {
    "A": "A", "ANSWER": "A",
    "R": "R", "REQUEST-INFO": "R", "REQUESTINFO": "R", "REQUEST_INFO": "R",
    "C": "C", "CLARIFY": "C",
    "O": "O", "OUT-OF-SCOPE": "O", "OUTOFSCOPE": "O", "OOS": "O",
}

def norm(cell: str):
    s = (cell or "").strip().upper().rstrip("?").strip()
    s = s.replace(" ", "")
    return _MAP.get(s)  # None if blank/unrecognized

def load_annotators(folder: Path):
    """rater_name -> {id: label}."""
    out = {}
    for fp in sorted(glob.glob(str(folder / "*.csv"))):
        name = Path(fp).stem
        with open(fp, encoding="utf-8-sig") as f:
            rdr = csv.DictReader(f)
            cols = {c.lower(): c for c in (rdr.fieldnames or [])}
            idc = cols.get("id"); lc = next((cols[c] for c in cols if c.startswith("label")), None)
            if not idc or not lc:
                print(f"  ! skip {name}: needs 'id' and 'label' columns", file=sys.stderr)
                continue
            m = {}
            for row in rdr:
                lab = norm(row.get(lc))
                if row.get(idc) and lab:
                    m[row[idc].strip()] = lab
            out[name] = m
    return out

def fleiss_kappa(rows):
    """rows: list of per-item category counts (only items with the same total n)."""
    from collections import Counter
    by_n = defaultdict(list)
    for c in rows:
        by_n[sum(c.values())].append(c)
    # use the modal rater-count group (usually all raters present)
    n = max(by_n, key=lambda k: len(by_n[k])) if by_n else 0
    grp = by_n.get(n, [])
    N = len(grp)
    if N == 0 or n < 2:
        return None, N, n
    p = {cat: sum(c.get(cat, 0) for c in grp) / (N * n) for cat in CATS}
    Pbar = sum((sum(c.get(cat, 0) ** 2 for cat in CATS) - n) / (n * (n - 1)) for c in grp) / N
    Pe = sum(v * v for v in p.values())
    kappa = (Pbar - Pe) / (1 - Pe) if (1 - Pe) else None
    return kappa, N, n

def main(folder=DIR / "labels"):
    ann = load_annotators(Path(folder))
    if not ann:
        print(f"No annotator CSVs found in {folder}. Put one CSV per person there.", file=sys.stderr)
        sys.exit(1)
    ids = sorted({i for m in ann.values() for i in m})
    per_item_counts, gt, pool = [], [], []
    for qid in ids:
        votes = [m[qid] for m in ann.values() if qid in m]
        cnt = Counter(votes)
        per_item_counts.append({c: cnt.get(c, 0) for c in CATS})
        top, ntop = (cnt.most_common(1)[0] if cnt else (None, 0))
        n = len(votes)
        if n and ntop >= math.floor(n / 2) + 1:          # strict majority of present raters
            gt.append({"id": qid, "label": top, "votes": dict(cnt), "n_raters": n})
        else:
            pool.append({"id": qid, "votes": dict(cnt), "n_raters": n})

    kappa, Nk, nk = fleiss_kappa(per_item_counts)
    (Path(folder).parent / "ground_truth.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in gt), encoding="utf-8")
    with open(Path(folder).parent / "disagreement_pool.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["id", "votes", "n_raters", "adjudicated_label", "reason"])
        for r in pool:
            w.writerow([r["id"], json.dumps(r["votes"], ensure_ascii=False), r["n_raters"], "", ""])
    report = {
        "n_annotators": len(ann), "annotators": {k: len(v) for k, v in ann.items()},
        "n_items": len(ids), "majority_resolved": len(gt), "disagreement_pool": len(pool),
        "majority_rate": round(len(gt) / len(ids), 3) if ids else None,
        "fleiss_kappa": (round(kappa, 3) if kappa is not None else None),
        "kappa_computed_on": {"n_items": Nk, "raters_per_item": nk},
        "majority_label_distribution": dict(Counter(r["label"] for r in gt)),
    }
    (Path(folder).parent / "agreement_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n-> ground_truth.jsonl ({len(gt)}), disagreement_pool.csv ({len(pool)}) — adjudicate the pool, then freeze.")

if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DIR / "labels")
