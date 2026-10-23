#!/usr/bin/env python3
"""
plot_safety_utility.py
----------------------
Final #2 figure: ratio-free safety-vs-utility operating points on the
clean + trap deployment distribution. Reads the adapter JSON and writes a
single-column figure to paper/figures/.

x = clean coverage (utility retained), y = trap FNAR (false numeric acceptance).
More governance (plain R-PAL -> prompted R-PAL -> VerifiQuant) trades clean
coverage for a lower false-acceptance rate; the ideal corner is bottom-right.

``--vq july`` (default) redraws the submitted figure; ``--vq top4x3`` draws the
camera-ready one, whose VerifiQuant trap point is the mean of three top-4 runs.
``--with-staged`` (top4x3 only) adds the staged controls S1 and S2 of camera-ready E7 as
reference points, read from governance_confirmatory_1014/cameraready_e10_e7_summary.json
(3-run means on both axes), and writes safety_utility_250q_top4_staged.
``--compact`` draws the same figure flatter for the 9-page camera-ready (shorter axis labels,
no em-dash) and adds ``_compact`` to the file name, so the archived drafts keep their figure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
_ap = argparse.ArgumentParser()
_ap.add_argument("--vq", choices=("july", "top4x3"), default="july")
_ap.add_argument("--with-staged", action="store_true")
_ap.add_argument("--compact", action="store_true")
ARGS = _ap.parse_args()
if ARGS.with_staged and ARGS.vq != "top4x3":
    _ap.error("--with-staged needs --vq top4x3")
_SUFFIX = "" if ARGS.vq == "july" else "_top4"
_FIG_SUFFIX = _SUFFIX + ("_staged" if ARGS.with_staged else "") + ("_compact" if ARGS.compact else "")
DATA = ROOT / f"results/mixed_risk_coverage/mixed_risk_coverage{_SUFFIX}.json"
OUT = ROOT / f"paper/figures/safety_utility_250q{_FIG_SUFFIX}"

d = json.load(open(DATA))["variant3_safety_vs_utility"]

COL = {"vq_k1": "#2563eb", "rpal_prompted": "#6b7280", "rpal_plain": "#dc2626"}
LAB = {"vq_k1": "VerifiQuant (K=1)", "rpal_prompted": "R-PAL$^{+}$ (taxonomy prompt)",
       "rpal_plain": "R-PAL (plain)"}
MRK = {"vq_k1": "o", "rpal_prompted": "D", "rpal_plain": "^"}
order = ["rpal_plain", "rpal_prompted", "vq_k1"]

fig, ax = plt.subplots(figsize=(4.7, 2.75) if ARGS.compact else (4.7, 3.7))

# All three systems now share the same 250Q clean denominator, so every point sits
# on the frontier and none needs a separate marker convention.
comparable = ["rpal_plain", "rpal_prompted", "vq_k1"]
xs = [d[k]["clean_coverage"] * 100 for k in comparable]
ys = [d[k]["trap_fnar"] * 100 for k in comparable]
ax.plot(xs, ys, "--", color="#9ca3af", lw=1.2, zorder=1)

for k in order:
    x, y = d[k]["clean_coverage"] * 100, d[k]["trap_fnar"] * 100
    ax.scatter(x, y, s=150, color=COL[k], marker=MRK[k], edgecolor="white", lw=1.6,
               zorder=3, label=LAB[k])

# per-point annotations (offset to avoid overlap). Half-up rounding, as the paper's
# macros are typeset; plain float formatting would print 0.1125 as 11.2.
def _pct(x: float) -> str:
    return f"{int(x * 1000 + 0.5 + 1e-9) / 10:.1f}%"


ann = {"rpal_plain": (6, 6), "rpal_prompted": (8, 6), "vq_k1": (8, -12) if ARGS.with_staged else (8, 6)}
if ARGS.compact:  # flatter axes: labels move off the legend and the ideal marker
    ann["vq_k1"] = (-12, -14)
disp = {k: f"{_pct(d[k]['clean_coverage'])} / {_pct(d[k]['trap_fnar'])}" for k in order}
for k in order:
    x, y = d[k]["clean_coverage"] * 100, d[k]["trap_fnar"] * 100
    ax.annotate(disp[k], (x, y), textcoords="offset points",
                xytext=ann[k], fontsize=8, color=COL[k])

if ARGS.with_staged:
    st = json.load(open(ROOT / "results/governance_confirmatory_1014/cameraready_e10_e7_summary.json"))
    staged = {
        "s1": (100 * st["clean250"]["s1_mean"]["coverage"], 100 * st["systems"]["s1"]["overall"]["fnar_mean"],
               "Staged", "#d97706", "s", (-70, -3) if ARGS.compact else (-30, 13)),
        "s2": (100 * st["clean250"]["s2_mean"]["coverage"], 100 * st["systems"]["s2"]["overall"]["fnar_mean"],
               "Staged+Spec", "#059669", "P", (-22, -15)),
    }
    path = [(d["rpal_prompted"]["clean_coverage"] * 100, d["rpal_prompted"]["trap_fnar"] * 100),
            staged["s1"][:2], staged["s2"][:2]]
    ax.plot([p[0] for p in path], [p[1] for p in path], ":", color="#d97706", lw=1.1, zorder=1)
    for k, (x, y, label, color, marker, off) in staged.items():
        ax.scatter(x, y, s=120, color=color, marker=marker, edgecolor="white", lw=1.4, zorder=3, label=label)
        ax.annotate(f"{_pct(x / 100)} / {_pct(y / 100)}", (x, y), textcoords="offset points",
                    xytext=off, fontsize=8, color=color)

# ideal corner
ax.scatter([100], [0], marker="*", s=210, color="#16a34a", zorder=2)
ax.annotate("ideal", (100, 0), textcoords="offset points", xytext=(-34, -11) if ARGS.compact else (-34, 6),
            fontsize=8.5, color="#16a34a")

if ARGS.compact:
    ax.set_xlabel("Clean coverage (% answered, utility retained)")
    ax.set_ylabel("Trap FNAR (%)\n↓ lower is safer")
else:
    ax.set_xlabel("Clean coverage (% answered — utility retained)")
    ax.set_ylabel("Trap FNAR (% false numeric acceptance)\n↓ lower is safer")
ax.set_xlim((54 if ARGS.compact else 56) if ARGS.with_staged else 82, 101.5)
ax.set_ylim(-7 if ARGS.compact else -3, 55)
ax.grid(alpha=0.25)
if ARGS.compact:
    ax.legend(fontsize=8, loc="upper left", frameon=True, markerscale=0.7, labelspacing=0.25, borderpad=0.35)
else:
    ax.legend(fontsize=8, loc="upper left", frameon=True)
# Denominators stay in the figure, at a readable size, because the LaTeX caption is
# kept short for the page budget. An earlier 7pt version of this line was unreadable.
_runs_note = ("R-PAL$^{+}$ = 3-run mean" if ARGS.vq == "july"
              else "trap values are 3-run means except plain R-PAL" if ARGS.with_staged
              else "R-PAL$^{+}$ and VerifiQuant traps = 3-run means")
ax.text(0.5, -0.36 if ARGS.compact else -0.26, (f"250 clean questions  |  80 traps  |  {_runs_note}" if ARGS.compact else f"clean 250Q  |  traps 80Q  |  {_runs_note}"),
        transform=ax.transAxes, ha="center", fontsize=8.5, color="#4b5563")

plt.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}.{ext}", dpi=200, bbox_inches="tight")
    print("wrote", f"{OUT}.{ext}")
