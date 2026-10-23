from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


def _load_json(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    payload = json.loads(text)
    return payload


def _load_config_ids(path: Path) -> Tuple[List[str], Dict[str, str]]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    target_ids = (cfg.get("target_ids") or {}) if isinstance(cfg, dict) else {}
    correct = list(target_ids.get("correct_samples") or [])
    error = list(target_ids.get("error_samples") or [])

    ordered: List[str] = []
    labels: Dict[str, str] = {}
    for qid in correct:
        q = str(qid).strip()
        if not q:
            continue
        if q not in labels:
            ordered.append(q)
        labels[q] = "correct"
    for qid in error:
        q = str(qid).strip()
        if not q:
            continue
        if q not in labels:
            ordered.append(q)
        labels[q] = "error"
    return ordered, labels


def build_seed_rows_from_config(
    *,
    config_path: Path,
    qa_dataset_path: Path,
    functions_catalog_path: Path,
    financial_docs_path: Path,
    max_doc_chars: int = 4000,
    dedupe_by_function_id: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    selected_ids, labels = _load_config_ids(config_path)
    qa_rows = _load_json(qa_dataset_path)
    if not isinstance(qa_rows, list):
        raise ValueError(f"qa dataset must be a JSON list: {qa_dataset_path}")
    qa_by_id = {str(r.get("question_id", "")).strip(): r for r in qa_rows if isinstance(r, dict)}

    functions = _load_json(functions_catalog_path)
    if not isinstance(functions, list):
        raise ValueError(f"functions catalog must be a JSON list: {functions_catalog_path}")
    func_by_id = {
        str(f.get("function_id", "")).strip(): f
        for f in functions
        if isinstance(f, dict) and str(f.get("function_id", "")).strip()
    }

    docs = _load_json(financial_docs_path)
    if not isinstance(docs, list):
        raise ValueError(f"financial docs must be a JSON list: {financial_docs_path}")
    doc_by_title = {
        str(d.get("title", "")).strip(): d
        for d in docs
        if isinstance(d, dict) and str(d.get("title", "")).strip()
    }

    missing_question_ids: List[str] = []
    missing_function_ids: List[str] = []
    missing_function_entries: List[str] = []
    missing_doc_titles: List[str] = []
    title_mismatches: List[str] = []

    grouped: Dict[str, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []

    for qid in selected_ids:
        qa = qa_by_id.get(qid)
        if qa is None:
            missing_question_ids.append(qid)
            continue

        function_id = str(qa.get("function_id", "")).strip()
        if not function_id:
            missing_function_ids.append(qid)
            continue

        func = func_by_id.get(function_id)
        if func is None:
            missing_function_entries.append(function_id)
            continue

        article_title_qa = str(qa.get("article_title", "")).strip()
        article_title_func = str(func.get("article_title", "")).strip()
        article_title = article_title_func or article_title_qa
        if article_title_qa and article_title_func and article_title_qa != article_title_func:
            title_mismatches.append(f"{qid}: qa='{article_title_qa}' vs function='{article_title_func}'")

        doc = doc_by_title.get(article_title)
        if doc is None:
            missing_doc_titles.append(article_title)
            continue

        function_code = str(func.get("function", "")).strip()
        content = str(doc.get("content", "")).strip()
        if max_doc_chars > 0 and len(content) > max_doc_chars:
            content = content[:max_doc_chars].rstrip() + "\n...[truncated]"

        seed_question = (
            f"Create a reusable Financial Inference Contract for '{article_title}' "
            f"based on the provided canonical function."
        )
        seed_context = (
            f"Article title: {article_title}\n"
            f"Reference document excerpt:\n{content}"
        )

        built = {
            "question_id": f"seed_{function_id}",
            "source": "config_target_function_seed",
            "function_id": function_id,
            "article_title": article_title,
            "article_doc_id": doc.get("id"),
            "article_content_excerpt": content,
            "function": function_code,
            "python_solution": function_code,
            "question": seed_question,
            "context": seed_context,
            "ground_truth": None,
            "difficulty": None,
            "level": "seed",
            "seed_from_question_ids": [qid],
            "seed_label_counts": {labels.get(qid, "unknown"): 1},
        }

        if dedupe_by_function_id:
            existing = grouped.get(function_id)
            if existing is None:
                grouped[function_id] = built
            else:
                existing["seed_from_question_ids"].append(qid)
                lbl = labels.get(qid, "unknown")
                counts = existing["seed_label_counts"]
                counts[lbl] = int(counts.get(lbl, 0)) + 1
        else:
            copy_row = dict(built)
            copy_row["question_id"] = f"seed_{function_id}_{qid}"
            rows.append(copy_row)

    if dedupe_by_function_id:
        rows = list(grouped.values())

    report = {
        "requested_question_ids": len(selected_ids),
        "built_seed_rows": len(rows),
        "dedupe_by_function_id": dedupe_by_function_id,
        "missing_question_ids": missing_question_ids,
        "missing_function_ids": missing_function_ids,
        "missing_function_entries": sorted(set(missing_function_entries)),
        "missing_doc_titles": sorted(set(t for t in missing_doc_titles if t)),
        "title_mismatches": title_mismatches,
    }
    return rows, report
