#!/usr/bin/env python3
"""
export_prompts_appendix.py
--------------------------
Write paper/appendix_prompts.tex: every prompt behind a reported number, printed verbatim.

Each prompt is produced by the code that ran the experiments, not copied by hand. The model
call is replaced by a stub that records the prompt, and run-time fields are filled with
<<PLACEHOLDER>> strings. The templates are unchanged since the reported runs:
  - VerifiQuant selector, binder and critic: identical at f7fd111 (2026-07-12) and today.
  - CoT, oracle, PAL, R-PAL, R-PAL+, self-consistency: unchanged since each script's first commit.
  - Staged controls: every run manifest stores the stage frame, stage specs and taxonomy hash,
    and all 36 manifests match the current code.

  - Library construction: the core-stage prompt is taken from commit 708a7bd (2026-05-23), the
    version in the repository when 176 of the 250 cards were built on 2026-06-15; the retrieval and
    repair prompts are unchanged since 2026-04-05.
  - Human-set authoring: AUTHOR_PROMPT and PERSONAS are unchanged since the set was generated; the
    missing-input intent is the 07-30 version whose outputs were revised offline (appendix E).

Usage: .venv/bin/python scripts/export_prompts_appendix.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT)]
OUT = ROOT / "paper/appendix_prompts.tex"

# Non-ASCII characters are printed in ASCII so pdfLaTeX's listings can typeset them.
ASCII = {"\u2192": "->", "\u2190": "<-", "\u2194": "<->", "\u2014": "--", "\u2013": "-", "\u201c": '"', "\u201d": '"',
         "\u2018": "'", "\u2019": "'", "\u2265": ">=", "\u2264": "<=", "\u00d7": "x", "\u2026": "...",
         "\u2260": "!=", "\u00b7": "*", "\u2248": "~"}


def capture(module: Any, attr: str, call: Callable[[], Any]) -> str:
    """Run `call` with module.<attr> (the LLM call) replaced by a stub; return the prompt."""
    seen: List[str] = []
    original = getattr(module, attr)

    def stub(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        seen.append(kwargs.get("prompt") if "prompt" in kwargs else args[2])
        return {}

    setattr(module, attr, stub)
    try:
        try:
            call()
        except Exception:
            pass  # the stub's empty reply may break post-processing; the prompt is already recorded
    finally:
        setattr(module, attr, original)
    assert len(seen) == 1, f"expected one model call, saw {len(seen)}"
    return seen[0]


def swap_json(prompt: str, value: Any, label: str) -> str:
    dumped = json.dumps(value, ensure_ascii=False, indent=2)
    assert dumped in prompt, f"sentinel JSON for {label} not found"
    return prompt.replace(dumped, f"<<{label}>>")


def fields(schema: Any) -> str:
    """Field list of a google.genai response schema, with enum values."""
    out = []
    for name, sub in (schema.properties or {}).items():
        enum = getattr(sub, "enum", None)
        out.append(f"\\texttt{{{name}}}" + (f" ({', '.join(enum)})" if enum else ""))
    return ", ".join(out).replace("_", "\\_")


def listing(text: str) -> str:
    for k, v in ASCII.items():
        text = text.replace(k, v)
    bad = sorted({c for c in text if ord(c) > 126})
    assert not bad, f"unmapped non-ASCII characters: {bad}"
    assert "\\end{lstlisting}" not in text
    return "\\begin{lstlisting}\n" + text.strip("\n") + "\n\\end{lstlisting}\n"


def runtime_prompts() -> List[str]:
    from verifiquant.pipeline import run_error_classification_pipeline as vq
    q, c = "<<QUESTION>>", "<<CONTEXT>>"
    card = {k: f"<<{k.upper()}>>" for k in ("fic_id", "title", "topic", "summary", "selection_hints",
                                             "applicable_when", "not_applicable_when", "scope_boundaries")}
    sel = capture(vq, "_llm_json", lambda: vq._select_card_with_llm(
        None, "model", q, c, [vq.RetrievalCandidate(retrieval=card, score=0.0)]))
    sel = sel.replace("Candidate 1 | fic_id=<<FIC_ID>> | score=0.000",
                      "Candidate <<RANK>> | fic_id=<<FIC_ID>> | score=<<RETRIEVAL SCORE>>")
    req = [{"name": "<<INPUT SCHEMA>>"}]
    ext = swap_json(capture(vq, "_llm_json", lambda: vq._extract_inputs_with_llm(
        None, "model", q, c, {"inputs": req})), req, "CONTRACT INPUT SCHEMA (JSON: name, type, unit, description, aliases)")
    hints, bound = [{"id": "<<HINT>>"}], {"<<INPUT>>": "<<VALUE>>"}
    crit = capture(vq, "_llm_json", lambda: vq._critic_check_with_llm(None, "model", q, c, hints, bound))
    crit = swap_json(swap_json(crit, hints, "DECLARED SEMANTIC HINTS (JSON)"), bound, "BOUND INPUTS (JSON)")
    crit_open = capture(vq, "_llm_json", lambda: vq._critic_check_with_llm(None, "model", q, c, [], bound))
    open_rules = crit_open.split("- Mode: open\n", 1)[1].split("\nQuestion:", 1)[0]
    return [
        "\\subsection{VerifiQuant at run time}\n"
        "The model is called in exactly three places. Each call requests JSON through a response schema, whose fields are listed after the prompt.\n",
        "\\paragraph{Contract selection (N and M)} One block per retrieved candidate, top 4, in rank order.\n" + listing(sel)
        + f"Output fields: {fields(vq._schema_selection())}.\n",
        "\\paragraph{Input binding (F and E read the bound values)}\\leavevmode\n" + listing(ext)
        + f"Output fields: {fields(vq._schema_extraction())}, where each input carries \\texttt{{name}}, \\texttt{{status}}, \\texttt{{value}}, and \\texttt{{source\\_scale}}.\n",
        "\\paragraph{Ambiguity critic (I)} When the contract declares no semantic hints, the two mode rules are replaced by the open-mode rules shown second.\n"
        + listing(crit) + listing(open_rules)
        + f"Output fields: {fields(vq._schema_critic_check())}.\n",
    ]


def oracle_prompts() -> List[str]:
    from verifiquant.pipeline import run_framework_guided_self_improve_pipeline as fg
    from verifiquant.pipeline import run_cot_self_improve_pipeline as cot
    diag, step = {"<<": "DIAGNOSTIC"}, {"<<": "COT"}
    vq_oracle = capture(fg, "_llm_json", lambda: fg._oracle_rewrite_for_framework(
        client=None, model="model", row={"code": "<<REFERENCE SOLUTION CODE>>"},
        current_question="<<QUESTION>>", current_context="<<CONTEXT>>", diagnostic=diag))
    vq_oracle = swap_json(vq_oracle, diag, "DIAGNOSTICREPORT OF THE PREVIOUS TURN (JSON)")
    cot_oracle = capture(cot, "_llm_json", lambda: cot._oracle_rewrite_for_cot(
        client=None, model="model", row={"code": "<<REFERENCE SOLUTION CODE>>"},
        current_question="<<QUESTION>>", current_context="<<CONTEXT>>", cot_step_output=step))
    cot_oracle = swap_json(cot_oracle, step, "COT OUTPUT OF THE PREVIOUS TURN (JSON)")
    return [
        "\\subsection{Clarification oracle ($K{>}1$)}\n"
        "The oracle sees the reference solution code of the question, never the gold number, and may only rewrite the question and context. "
        "VerifiQuant's oracle reads the DiagnosticReport of a structured exit. The CoT oracle of Table~\\ref{tab:v2kabl} reads the CoT output of every turn.\n",
        "\\paragraph{VerifiQuant oracle.}\\leavevmode\n" + listing(vq_oracle) + f"Output fields: {fields(fg._schema_oracle_rewrite())}.\n",
        "\\paragraph{CoT oracle.}\\leavevmode\n" + listing(cot_oracle) + f"Output fields: {fields(cot._schema_oracle_rewrite())}.\n",
    ]


def baseline_prompts() -> List[str]:
    from verifiquant.pipeline import run_cot_self_improve_pipeline as cot
    import run_pal_baseline as pal
    import run_retrieval_pal_baseline as rpal
    import run_retrieval_pal_governance_eval as gov
    import run_sc_baseline as sc
    row = {"question": "<<QUESTION>>", "context": "<<CONTEXT>>"}
    cand = [{"rank": "<<RANK>>", "title": "<<TITLE>>", "summary": "<<SUMMARY>>", "function": "<<REFERENCE FUNCTION CODE>>"}]
    cot_step = capture(cot, "_llm_json", lambda: cot._cot_step(client=None, model="model", question="<<QUESTION>>", context="<<CONTEXT>>"))
    return [
        "\\subsection{Baselines}\n"
        "R-PAL and R-PAL$^{+}$ receive one reference block per retrieved candidate, top 4, in rank order. "
        "The same R-PAL prompt serves the clean questions, the traps, the out-of-library split, and the human-labeled set, and the same holds for R-PAL$^{+}$.\n",
        "\\paragraph{CoT (single shot)}\\leavevmode\n" + listing(cot_step) + f"Output fields: {fields(cot._schema_cot_step())}.\n",
        "\\paragraph{PAL (no retrieval)}\\leavevmode\n" + listing(pal.PROMPT.format(**row)),
        "\\paragraph{R-PAL.}\\leavevmode\n" + listing(rpal._prompt_for(row, cand)),
        "\\paragraph{R-PAL$^{+}$ (taxonomy prompt)}\\leavevmode\n" + listing(gov._prompted_prompt(row, cand)),
        "\\paragraph{Self-consistency.} Five samples at temperature 1.0, majority vote over answers within 1\\%.\n" + listing(sc.PROMPT.format(**row)),
    ]


def staged_prompts() -> List[str]:
    import run_staged_rpal_governance_eval as st
    src = (ROOT / "scripts/run_staged_rpal_governance_eval.py").read_text(encoding="utf-8")
    row = {"question": "<<QUESTION>>", "context": "<<CONTEXT>>"}
    ref = st._render_reference({"rank": "<<RANK>>", "title": "<<TITLE>>", "summary": "<<SUMMARY>>", "function": "<<REFERENCE FUNCTION CODE>>"})
    spec = lambda label: f"<<{label}>>"  # noqa: E731
    retrieval = st._contract_retrieval_text({k: [spec(k.upper())] for k in ("selection_hints", "applicable_when", "not_applicable_when", "scope_boundaries")})
    headers = {
        "scope": "Retrieved reference functions (ranked):\n",
        "meta": "\nContract metadata:\n",
        "schema": "\n\nContract input schema:\n",
        "checks": "\n\nContract boundary checks (declared rules; judge them, they are not run):\n",
        "hints": "\n\nContract semantic hints (declared ambiguities):\n",
        "bound": "Bound inputs (from stage 2):\n",
        "soft": "\n\nI_SOFT assumption (from stage 4):\n",
    }
    for h in headers.values():  # every header is copied from run_case; fail if it drifted
        assert json.dumps(h)[1:-1] in src or h.strip() in src, f"header not in source: {h!r}"
    chosen = "<<CHOSEN REFERENCE BLOCK>>"
    bound = headers["bound"] + "<<BOUND INPUTS FROM STAGE 2 (JSON)>>"
    bodies = {
        "scope": headers["scope"] + ref + "  [repeated for each candidate]\n[Staged+Spec only:]" + headers["meta"] + retrieval,
        "inputs": chosen + "\n[Staged+Spec only:]" + headers["schema"] + "<<CONTRACT INPUT SCHEMA (JSON: name, type, required, unit, description)>>",
        "bounds": chosen + "\n\n" + bound + "\n[Staged+Spec only:]" + headers["checks"] + "<<DECLARED E-CHECKS (JSON: rule_id, name, description, expression, applies_to)>>",
        "ambiguity": chosen + "\n\n" + bound + "\n[Staged+Spec only:]" + headers["hints"] + "<<DECLARED SEMANTIC HINTS (JSON)>>",
        "program": chosen + "\n\n" + bound + "\n[only if stage 4 returned I_SOFT:]" + headers["soft"] + "<<WARNING FROM STAGE 4>>",
    }
    out = [
        "\\subsection{Staged controls}\n"
        "Every stage prompt starts with the taxonomy block of the R-PAL$^{+}$ prompt above, verbatim from its first line through ``Do not force-fit one.'', shown here as \\texttt{<{}<TAXONOMY>{}>}. "
        "Lines marked \\texttt{[Staged+Spec only:]} are absent from Staged.\n"
    ]
    for stage, body in bodies.items():
        p = st._stage_prompt(stage, row, body).replace(st.TAXONOMY, "<<TAXONOMY>>\n")
        out.append(f"\\paragraph{{Stage {st.STAGES[stage]['k']}: {st.STAGES[stage]['name']}.}}\\leavevmode\n" + listing(p))
    return out


# Common constants excluded when testing whether a card repeats a number from its source question.
COMMON = {"0", "1", "2", "3", "4", "5", "10", "12", "24", "30", "52", "60", "100", "252", "360", "365",
          "1000", "0.0", "1.0", "2.0", "0.5", "0.01", "0.05", "1e-9"}


def card_overlap() -> Dict[str, int]:
    """Cards that hold a number found in their source question (text or worked solution) but not in
    its reference function, by field. Deterministic; no model call."""
    import re
    num = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])")
    nums = lambda t: set(num.findall(t or "")) - COMMON  # noqa: E731
    base = ROOT / "results/paper_v2_250"
    qs = {json.loads(l)["question_id"]: json.loads(l) for l in (base / "questions_250.jsonl").open()}
    counts = {"cards": 0, "total": 0, "semantic_hints": 0, "diagnostic_checks": 0, "inputs": 0, "code": 0}
    for line in (base / "fic/core.jsonl").open():
        card = json.loads(line)
        q = qs.get((card.get("source_meta") or {}).get("question_id"))
        if not q:
            continue
        counts["total"] += 1
        own = (nums(q.get("python_solution")) | nums(q.get("question", "") + " " + q.get("context", ""))) - nums(q.get("function"))
        fields_hit = {
            "semantic_hints": nums(json.dumps(card.get("semantic_hints"), ensure_ascii=False)) & own,
            "diagnostic_checks": nums(json.dumps(card.get("diagnostic_checks"), ensure_ascii=False)) & own,
            "inputs": nums(json.dumps(card.get("inputs"), ensure_ascii=False)) & own,
            "code": nums((card.get("execution") or {}).get("code")) & own,
        }
        for k, v in fields_hit.items():
            counts[k] += bool(v)
        counts["cards"] += any(fields_hit.values())
    return counts


def library_prompts() -> List[str]:
    import ast, re, subprocess
    from verifiquant.preprocessing import stage_retrieval, stage_repair
    path = "verifiquant/preprocessing/stage_core.py"
    src = subprocess.run(["git", "show", f"708a7bd:{path}"], capture_output=True, text=True, cwd=ROOT, check=True).stdout
    core_prompt = next(ast.literal_eval(n.value) for n in ast.parse(src).body if isinstance(n, ast.Assign)
                       and any(getattr(t, "id", None) == "STAGE_CORE_PROMPT" for t in n.targets))
    hint_max = int(re.search(r"^SEMANTIC_HINT_MAX = (\d+)", src, flags=re.M).group(1))
    core = core_prompt.format(
        definition_json="<<SOURCE RECORD (JSON): source_meta, article_title, article_doc_id, article_content_excerpt, function, python_solution, context, question>>",
        taxonomy_json="<<DOMAIN TAXONOMY (JSON)>>", semantic_hint_max=hint_max)
    card = "<<CORE CARD FROM STAGE 1 (JSON)>>"
    c = card_overlap()
    return [
        "\\subsection{Library construction}\n"
        "Each formula family is built by three model calls. The first writes the core card from the source record, which holds the "
        "family's reference function together with the question it was sampled from, that question's context, and its worked solution "
        "(\\texttt{python\\_solution}), but never its numeric gold answer. The second writes retrieval metadata and the third writes "
        "repair rules, both from the core card. Deterministic normalization, lint, and the type audit of \\S\\ref{sec:v2method} follow. "
        "176 of the 250 cards were built on June 15, 2026 with the core prompt below, as committed on May 23. The other 74 were carried "
        "over from earlier builds made with earlier revisions of the same prompt, all kept in the repository history. A July revision "
        "added an input and output type policy, which the library meets through the build-time type audit rather than regeneration.\n",
        "\\paragraph{Question details in cards.} Because the first call sees the source question, a card can repeat details of it. "
        "We count a card as doing so when one of its fields holds a number that appears in its source question or worked solution but "
        "not in its reference function, ignoring common constants (0 to 5, 10, 12, 24, 30, 52, 60, 100, 252, 360, 365, 1000, and a few "
        f"unit values). {c['cards']} of the {c['total']} cards do: {c['semantic_hints']} in a declared semantic hint, "
        f"{c['diagnostic_checks']} in a boundary check, {c['inputs']} in an input description, and {c['code']} in the compute code. "
        "Most are example values inside a declared ambiguity, such as a hint that asks whether a stated rate of 7\\% is meant as 0.07 or as 7.\n",
        "\\paragraph{Core card (stage 1)}\\leavevmode\n" + listing(core),
        "\\paragraph{Retrieval metadata (stage 2)}\\leavevmode\n" + listing(stage_retrieval.STAGE_RETRIEVAL_PROMPT.format(core_json=card)),
        "\\paragraph{Repair rules (stage 3)}\\leavevmode\n" + listing(stage_repair.STAGE_REPAIR_PROMPT.format(core_json=card)),
    ]


def authoring_prompts() -> List[str]:
    import ast, subprocess
    import build_messy_query_pilot as mq
    path = "scripts/build_messy_query_pilot.py"
    src = subprocess.run(["git", "show", f"2b27507:{path}"], capture_output=True, text=True, cwd=ROOT, check=True).stdout
    old_intents = next(ast.literal_eval(n.value) for n in ast.parse(src).body if isinstance(n, ast.Assign)
                       and any(getattr(t, "id", None) == "INTENTS" for t in n.targets))
    prompt = mq.AUTHOR_PROMPT.format(persona_desc="<<PERSONA>>", intent_desc="<<INTENT SITUATION>>",
                                     context_rule="<<CONTEXT RULE>>", q="<<REFERENCE QUESTION>>",
                                     c="<<REFERENCE CONTEXT, FIRST 800 CHARACTERS>>")
    personas = "\n".join(f"{k}: {v}" for k, v in mq.PERSONAS.items())
    intents = {k: mq.INTENTS[k] for k in ("clean", "ambiguous", "judgment")}
    intents["missing (original; outputs revised, Appendix E)"] = old_intents["missing"]
    intent_text = "\n\n".join(f"{k}\n  situation: {v[0]}\n  context rule: {v[1]}" for k, v in intents.items())
    return [
        "\\subsection{Authoring the human-labeled set}\n"
        "GPT-5.2 wrote each query from a FinanceReasoning topic outside the 250, with one persona and one latent intent. "
        "The 18 queries of the missing-input stratum were later revised offline with a Codex agent, as described in Appendix~\\ref{app:human}, "
        "and no model call was made for that revision.\n",
        "\\paragraph{Author prompt (GPT-5.2)}\\leavevmode\n" + listing(prompt),
        "\\paragraph{Personas}\\leavevmode\n" + listing(personas),
        "\\paragraph{Intents}\\leavevmode\n" + listing(intent_text),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.parse_args()
    parts = [
        "%% Generated by scripts/export_prompts_appendix.py. Do not edit by hand.\n",
        "\\onecolumn\n\\sloppy\n",
        "\\lstset{basicstyle=\\ttfamily\\scriptsize, breaklines=true, breakatwhitespace=false, columns=fullflexible,"
        " keepspaces=true, showstringspaces=false, frame=single, framerule=0.3pt, aboveskip=4pt, belowskip=4pt,"
        " xleftmargin=3pt, xrightmargin=3pt, upquote=true}\n",
        "\\section{Prompts}\\label{app:prompts}\n"
        "Every prompt is printed as the experiment code builds it. Fields filled at run time appear as \\texttt{<{}<NAME>{}>}, "
        "and non-ASCII arrows and dashes are printed in ASCII. Gemini 2.5 Flash answers every call unless stated otherwise, "
        "and the Gemini 2.5 Pro runs of Appendix~\\ref{app:pro} use the same prompts.\n",
    ]
    for block in (runtime_prompts, oracle_prompts, baseline_prompts, staged_prompts, library_prompts, authoring_prompts):
        parts.extend(block())
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
