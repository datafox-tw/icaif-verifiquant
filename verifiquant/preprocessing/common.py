from __future__ import annotations

import json
import math
import os
import re
import statistics as _statistics
import sys
import datetime as _datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover - runtime dependency guard
    genai = None
    genai_types = None


@dataclass
class ConversionInput:
    source_meta: Dict[str, Any]
    article_title: str
    article_doc_id: Optional[int]
    article_content_excerpt: str
    function: str
    python_solution: str
    context: str
    question: str


def load_records(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    raise ValueError(f"Unsupported JSON payload type in {path}")


def dump_records(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = list(rows)
    if path.suffix.lower() == ".jsonl":
        with path.open("w", encoding="utf-8") as fh:
            for row in payload:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_fic_import(name: str, globals=None, locals=None, fromlist=(), level: int = 0):  # type: ignore[no-untyped-def]
    """Import hook for deterministic FIC execution sandboxes.

    The allowlist covers stdlib helpers plus project-pinned numerical
    dependencies used by finance formulas.  It intentionally remains
    root-package based and does not install packages at execution time.
    """
    if level != 0:
        raise ImportError("relative imports are disabled in FIC execution")
    root = name.split(".", 1)[0]
    allowed = {
        "_strptime": None,
        "datetime": _datetime,
        "json": json,
        "math": math,
        "numpy": None,
        "scipy": None,
        "statistics": _statistics,
    }
    if root not in allowed:
        raise ImportError(f"import of '{name}' is not allowed in FIC execution")
    try:
        return __import__(name, globals, locals, fromlist, level)
    except ModuleNotFoundError as exc:
        missing = str(getattr(exc, "name", "") or root)
        if missing == root or missing.startswith(root + "."):
            raise ModuleNotFoundError(
                f"Allowed FIC dependency '{missing}' is not installed; "
                "install the pinned project requirements before running experiments."
            ) from exc
        raise


def to_conversion_input(record: Dict[str, Any]) -> ConversionInput:
    doc_id_raw = record.get("article_doc_id")
    article_doc_id: Optional[int] = None
    try:
        if doc_id_raw is not None and str(doc_id_raw).strip() != "":
            article_doc_id = int(doc_id_raw)
    except Exception:
        article_doc_id = None

    return ConversionInput(
        source_meta={
            "function_id": record.get("function_id"),
            "source": record.get("source"),
            "question_id": record.get("question_id"),
            "difficulty": record.get("difficulty"),
            "level": record.get("level"),
        },
        article_title=str(record.get("article_title", "") or ""),
        article_doc_id=article_doc_id,
        article_content_excerpt=str(record.get("article_content_excerpt", "") or ""),
        function=str(record.get("function", "") or ""),
        python_solution=str(record.get("python_solution", "") or ""),
        context=str(record.get("context", "") or ""),
        question=str(record.get("question", "") or ""),
    )


def safe_id(raw: str, *, default_prefix: str = "fic") -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", str(raw or "").strip()).strip("_").lower()
    if not cleaned:
        return f"{default_prefix}_generated"
    if not cleaned.startswith("fic_"):
        return f"fic_{cleaned}"
    return cleaned


def normalize_label(raw: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(raw or "").strip().lower()).strip("_")


def normalize_input_type(raw: str) -> str:
    t = normalize_label(raw)
    if not t:
        return "number"

    aliases = {
        "float": "number",
        "double": "number",
        "numeric": "number",
        "int": "integer",
        "list": "array[number]",
        "array": "array[number]",
        "list_number": "array[number]",
        "array_number": "array[number]",
        "array_float": "array[number]",
        "array_int": "array[number]",
        "array_array": "array[array[number]]",
        "array_array_number": "array[array[number]]",
        "array_list_number": "array[array[number]]",
        "list_array_number": "array[array[number]]",
        "list_list_number": "array[array[number]]",
        "matrix": "array[array[number]]",
        "correlation_matrix": "array[array[number]]",
        "dict": "dict[string,number]",
        "dict_string_number": "dict[string,number]",
        "dict_str_number": "dict[string,number]",
        "dictionary_string_number": "dict[string,number]",
        "map_string_number": "dict[string,number]",
        "dictionary": "dict[string,number]",
        "mapping": "dict[string,number]",
        "object": "object",
        "json": "object",
        "list_dict": "array[object]",
        "list_object": "array[object]",
        "array_dict": "array[object]",
        "array_object": "array[object]",
        "bool": "boolean",
        "str": "string",
    }
    if t in aliases:
        return aliases[t]
    if t in {
        "number",
        "integer",
        "boolean",
        "string",
        "array[number]",
        "array[array[number]]",
        "array[object]",
        "dict[string,number]",
        "object",
    }:
        return t
    if "dict" in t or "dictionary" in t or "mapping" in t:
        return "dict[string,number]"
    if ("object" in t or "json" in t) and ("array" in t or "list" in t):
        return "array[object]"
    if ("matrix" in t) or (("array" in t or "list" in t) and any(x in t for x in ("nested", "2d", "array_array", "list_list"))):
        return "array[array[number]]"
    if "array" in t or "list" in t:
        return "array[number]"
    return "number"


def require_compute_code(code: str) -> str:
    c = str(code or "").strip()
    if "def compute(inputs)" not in c:
        raise ValueError("execution.code must define compute(inputs)")
    try:
        compile(c, "<fic-compute>", "exec")
    except SyntaxError as exc:
        raise ValueError(f"execution.code must be valid Python: {exc}") from exc
    return c


def require_gemini_client() -> Any:
    if genai is None:
        raise RuntimeError(
            "Missing dependency: google.genai import failed. Please install compatible google-genai package."
        )
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY in environment.")
    return genai.Client(api_key=api_key)


def call_gemini_json(
    client: Any,
    *,
    model: str,
    prompt: str,
    schema: Any,
) -> Dict[str, Any]:
    if genai_types is None:
        raise RuntimeError("google.genai.types is unavailable in this environment.")
    config = genai_types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=schema,
    )
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    try:
        return json.loads(response.text)
    except json.JSONDecodeError as err:
        raise ValueError(f"Gemini returned invalid JSON: {response.text}") from err


def optional_str(value: Any) -> Optional[str]:
    s = str(value or "").strip()
    return s or None
