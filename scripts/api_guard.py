"""Keep API trouble from turning into abstentions in long reruns (camera-ready E13-E15).

The July runners (run_ood_family_eval.py, run_vq50_pro.py) catch every exception from
``diagnose_row`` and record the case as an error, which their scorers count as an
abstention. That is the right rule for a pipeline failure, but a 503, a 429 or a dead
connection is not the system declining. This module separates the two:

- ``client_with_timeout``: the usual client plus an HTTP timeout, so a request on a dead
  connection raises instead of blocking forever.
- ``diagnose_with_api_retry``: retries only API-level failures (google.genai APIError,
  httpx transport errors, timeouts) with backoff. Any other exception propagates exactly
  as before, so the caller's original error handling still applies to it.
- When the retries run out it raises ``ApiExhausted``; the runners then exit with code 3
  without writing that row, and ``--resume`` continues from the same file later.
"""
from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from verifiquant.pipeline.run_error_classification_pipeline import create_genai_client_from_env

EXIT_API_EXHAUSTED = 3
_API_ERRORS = (genai_errors.APIError, httpx.TransportError, TimeoutError, ConnectionError)
_BACKOFF_S = (5, 10, 20, 40, 80, 160, 300)


class ApiExhausted(RuntimeError):
    pass


def client_with_timeout(timeout_s: int = 120) -> Any:
    create_genai_client_from_env()  # same key check and error message as every other runner
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"],
                        http_options=genai_types.HttpOptions(timeout=timeout_s * 1000))


def diagnose_with_api_retry(call: Callable[[], Dict[str, Any]], *, label: str) -> Dict[str, Any]:
    """Run ``call``; retry API-level failures, re-raise everything else unchanged."""
    last = ""
    for attempt, wait in enumerate((0,) + _BACKOFF_S):
        if wait:
            print(f"API_RETRY {label} attempt {attempt} after {wait}s: {last}", flush=True)
            time.sleep(wait)
        try:
            return call()
        except _API_ERRORS as exc:
            last = f"{type(exc).__name__}: {str(exc)[:200]}"
    raise ApiExhausted(f"{label}: {last}")
