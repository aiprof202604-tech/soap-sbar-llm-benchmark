"""
Common utilities for resumable, retry-aware, parallel API call execution.

Provides:
- ResumableCSVWriter: thread-safe append-only CSV writer
- load_completed_keys: read existing CSV and return set of completed keys
- retry_with_backoff: decorator for transient API errors
- VENDOR_PARALLEL: per-vendor recommended worker counts (Tier-aware)

Used by: run_experiment.py, run_ai_judges.py, run_arbiter.py
"""

import csv
import os
import threading
import time
from pathlib import Path

# Per-vendor recommended parallel worker counts.
# Conservative values to fit Anthropic Tier 1 (50 RPM for Opus).
# OpenAI Tier 1 typically allows ~500 RPM for gpt-4o.
# Google Gemini Free Tier allows ~15 RPM but paid tier is more lenient.
VENDOR_PARALLEL = {
    "openai":    12,   # comfortably below typical Tier 1 RPM
    "anthropic":  4,   # Tier 1 = 50 RPM; 4 workers ≈ 240 RPM theoretical, but token-per-min is the binding constraint
    "google":    20,   # most permissive
}

# HTTP / API error categories
PERMANENT_ERROR_TOKENS = (
    "401",  # unauthorized
    "403",  # forbidden
    "404",  # not found
    "invalid_api_key",
    "authentication_error",
    "model_not_found",
    "permission_denied",
)

TRANSIENT_ERROR_TOKENS = (
    "429",  # rate limit
    "500",  # internal error
    "502",  # bad gateway
    "503",  # service unavailable
    "504",  # gateway timeout
    "rate_limit",
    "overloaded",
    "timeout",
    "connection",
)


def is_permanent_error(exc):
    """Return True if the exception should NOT be retried."""
    msg = str(exc).lower()
    return any(token in msg for token in PERMANENT_ERROR_TOKENS)


def is_transient_error(exc):
    """Return True if the exception is a candidate for retry."""
    msg = str(exc).lower()
    return any(token in msg for token in TRANSIENT_ERROR_TOKENS)


def call_with_retry(fn, *args, max_retries=2, base_backoff=1.0, **kwargs):
    """Call fn with exponential backoff for transient errors.

    Returns (result, retry_count, error_str).
    On success: (result, retry_count, "")
    On final failure: (None, retry_count, error_message)

    Retries only on transient errors (rate limits, timeouts, 5xx).
    Permanent errors (auth, invalid model) fail immediately.
    """
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            result = fn(*args, **kwargs)
            return result, attempt, ""
        except Exception as exc:
            last_exc = exc
            if is_permanent_error(exc):
                # don't retry permanent errors
                return None, attempt, f"{type(exc).__name__}: {exc}"
            if attempt < max_retries:
                # exponential backoff: 1s, 2s, 4s, ...
                wait = base_backoff * (2 ** attempt)
                time.sleep(wait)
                continue
            else:
                return None, attempt, f"{type(exc).__name__}: {exc}"
    return None, max_retries, f"{type(last_exc).__name__}: {last_exc}"


class ResumableCSVWriter:
    """Thread-safe append-only CSV writer for resumable execution.

    On instantiation:
    - If the output file does not exist, creates it with the header.
    - If it exists, opens in append mode (header not rewritten).

    Usage:
        writer = ResumableCSVWriter(path, fieldnames)
        writer.write_row({...})  # safe to call from multiple threads
        writer.close()
    """

    def __init__(self, path, fieldnames):
        self.path = Path(path)
        self.fieldnames = fieldnames
        self._lock = threading.Lock()
        is_new = not self.path.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8", newline="", buffering=1)
        self._writer = csv.DictWriter(self._fh, fieldnames=fieldnames)
        if is_new:
            self._writer.writeheader()
            self._fh.flush()

    def write_row(self, row):
        with self._lock:
            self._writer.writerow(row)
            self._fh.flush()

    def close(self):
        with self._lock:
            if self._fh and not self._fh.closed:
                self._fh.close()


def load_completed_keys(path, key_columns):
    """Read existing CSV and return a set of tuples representing completed keys.

    A row is considered "completed" if all key columns are present AND
    the row was successfully processed. Caller decides what counts as success
    by passing a filter function via load_completed_keys_with_filter.

    For run_experiment.py: success = is_valid==True
    For run_ai_judges.py:  success = judge_success==True
    For run_arbiter.py:    success = arbiter_success==True

    To support retry-on-failure logic (option (b) from session), use
    load_completed_keys_with_filter and exclude failed rows from the set.

    Returns: set of tuples (col1_value, col2_value, ...).
    """
    if not Path(path).exists():
        return set()
    completed = set()
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                key = tuple(row[c] for c in key_columns)
                completed.add(key)
            except KeyError:
                continue
    return completed


def load_completed_keys_with_filter(path, key_columns, success_filter):
    """Same as load_completed_keys but excludes rows that match a failure filter.

    Args:
        path: CSV path
        key_columns: list of column names that uniquely identify a "cell"
        success_filter: function(row_dict) -> bool, True if successfully completed

    Returns: set of tuples representing successfully completed keys.
    """
    if not Path(path).exists():
        return set()
    completed = set()
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                if success_filter(row):
                    key = tuple(row[c] for c in key_columns)
                    completed.add(key)
            except (KeyError, ValueError):
                continue
    return completed


def normalise_key(*values):
    """Convert key values to strings (CSV reads everything as str)."""
    return tuple(str(v) for v in values)
