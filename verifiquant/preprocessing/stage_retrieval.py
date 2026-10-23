from __future__ import annotations

import json
from typing import Any, Dict, List

from verifiquant.preprocessing.common import call_gemini_json, normalize_label

try:
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover
    genai_types = None


STAGE_RETRIEVAL_PROMPT = """Role:
You are a retrieval engineer generating retrieval metadata for a financial formula card.

Task:
Given one `fic_core`, generate retrieval-focused fields only:
- title
- summary
- selection_hints
- applicable_when
- not_applicable_when
- scope_boundaries
- common_confusions
- keywords

Rules:
1) Optimize for card retrieval and disambiguation.
2) Keep statements concise and concrete.
3) Do not add execution rules or repair workflow details.
4) Keep topic family aligned with fic_core domain/topic.

Input:
<CORE_JSON>
{core_json}
</CORE_JSON>

Output:
Return JSON only.
"""


def stage_retrieval_schema() -> Any:
    if genai_types is None:
        return None
    return genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "title": genai_types.Schema(type=genai_types.Type.STRING),
            "summary": genai_types.Schema(type=genai_types.Type.STRING),
            "selection_hints": genai_types.Schema(
                type=genai_types.Type.ARRAY,
                items=genai_types.Schema(type=genai_types.Type.STRING),
            ),
            "applicable_when": genai_types.Schema(
                type=genai_types.Type.ARRAY,
                items=genai_types.Schema(type=genai_types.Type.STRING),
            ),
            "not_applicable_when": genai_types.Schema(
                type=genai_types.Type.ARRAY,
                items=genai_types.Schema(type=genai_types.Type.STRING),
            ),
            "scope_boundaries": genai_types.Schema(
                type=genai_types.Type.ARRAY,
                items=genai_types.Schema(type=genai_types.Type.STRING),
            ),
            "common_confusions": genai_types.Schema(
                type=genai_types.Type.ARRAY,
                items=genai_types.Schema(
                    type=genai_types.Type.OBJECT,
                    properties={
                        "label": genai_types.Schema(type=genai_types.Type.STRING),
                        "difference": genai_types.Schema(type=genai_types.Type.STRING),
                    },
                    required=["label", "difference"],
                ),
            ),
            "keywords": genai_types.Schema(
                type=genai_types.Type.ARRAY,
                items=genai_types.Schema(type=genai_types.Type.STRING),
            ),
        },
        required=[
            "title",
            "summary",
            "selection_hints",
            "applicable_when",
            "not_applicable_when",
            "scope_boundaries",
            "common_confusions",
            "keywords",
        ],
    )


def build_stage_retrieval_prompt(core: Dict[str, Any]) -> str:
    return STAGE_RETRIEVAL_PROMPT.format(
        core_json=json.dumps(core, ensure_ascii=False, indent=2),
    )


def _norm_str_list(items: Any) -> List[str]:
    if not isinstance(items, list):
        return []
    out = []
    for item in items:
        text = str(item or "").strip()
        if text:
            out.append(text)
    return out


def _build_embedding_text(payload: Dict[str, Any]) -> str:
    pieces: List[str] = []
    for key in ("title", "summary"):
        value = str(payload.get(key, "")).strip()
        if value:
            pieces.append(value)
    pieces.extend(_norm_str_list(payload.get("selection_hints", [])))
    pieces.extend(_norm_str_list(payload.get("applicable_when", [])))
    pieces.extend(_norm_str_list(payload.get("not_applicable_when", [])))
    pieces.extend(_norm_str_list(payload.get("scope_boundaries", [])))
    pieces.extend(_norm_str_list(payload.get("keywords", [])))
    for conf in payload.get("common_confusions", []):
        if isinstance(conf, dict):
            pieces.append(str(conf.get("label", "")).strip())
            pieces.append(str(conf.get("difference", "")).strip())
    return " ".join(p for p in pieces if p)


def formalize_retrieval_payload(raw: Dict[str, Any], core: Dict[str, Any]) -> Dict[str, Any]:
    common_confusions = []
    for item in raw.get("common_confusions", []):
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip()
        difference = str(item.get("difference", "")).strip()
        if label and difference:
            common_confusions.append({"label": label, "difference": difference})

    payload = {
        "fic_id": core["fic_id"],
        "domain": core["domain"],
        "topic": core["topic"],
        "title": str(raw.get("title", "")).strip() or core["name"],
        "summary": str(raw.get("summary", "")).strip() or core.get("short_description", ""),
        "selection_hints": _norm_str_list(raw.get("selection_hints", []))
        or _norm_str_list(core.get("selection_hints", [])),
        "applicable_when": _norm_str_list(raw.get("applicable_when", [])),
        "not_applicable_when": _norm_str_list(raw.get("not_applicable_when", [])),
        "scope_boundaries": _norm_str_list(raw.get("scope_boundaries", []))
        or _norm_str_list(raw.get("not_applicable_when", [])),
        "common_confusions": common_confusions,
        "keywords": [normalize_label(x) for x in _norm_str_list(raw.get("keywords", [])) if normalize_label(x)],
    }
    payload["embedding_text"] = _build_embedding_text(payload)
    return payload


def generate_retrieval(*, client: Any, model: str, core: Dict[str, Any]) -> Dict[str, Any]:
    schema = stage_retrieval_schema()
    if schema is None:
        raise RuntimeError("Stage retrieval schema is unavailable because google.genai is missing.")
    raw = call_gemini_json(
        client,
        model=model,
        prompt=build_stage_retrieval_prompt(core),
        schema=schema,
    )
    return formalize_retrieval_payload(raw, core)
