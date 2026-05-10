"""
SOAP-to-SBAR translation experiment runner (resumable, retry-aware).

Calls 3 commercial LLMs (GPT-4o, Claude Opus 4.5, Gemini 2.5 Flash-Lite)
to convert each of 9 SOAP scenarios into SBAR format.

Design:
    9 scenarios x 3 models x 2 temperatures x 30 trials = 1,620 API calls

Resumable execution:
    Successfully completed cells (is_valid=True) are loaded from the
    output CSV at startup and skipped. To restart from scratch, pass
    --restart, which removes the existing output file before running.

Retry policy:
    Transient errors (rate limits, timeouts, 5xx) are retried up to
    twice with exponential backoff (1s, 2s, 4s).
    Permanent errors (auth, invalid model) fail immediately.
    Failed cells (is_valid=False) are NOT considered completed and
    will be re-attempted on the next run.

Per-vendor parallelism:
    Each vendor uses its own thread pool sized to its rate-limit tier:
        OpenAI    : 12 workers
        Anthropic :  4 workers (Tier 1, 50 RPM)
        Google    : 20 workers
    Vendors run independently in parallel.

Usage:
    python scripts/run_experiment.py            # resume from existing CSV
    python scripts/run_experiment.py --dry-run  # show plan only
    python scripts/run_experiment.py --restart  # delete existing CSV and start fresh
    python scripts/run_experiment.py --workers-openai 8  # override per-vendor

Set environment variables before running:
    OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY
"""

import argparse
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

from _common import (
    VENDOR_PARALLEL,
    ResumableCSVWriter,
    call_with_retry,
    load_completed_keys_with_filter,
    normalise_key,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SCENARIOS_DIR = DATA / "scenarios"

SYSTEM_PROMPT = (
    "あなたは病棟看護師です。患者の状態を申し送るSBARを作成します。"
)

USER_PROMPT_TEMPLATE = """以下のSOAP記録を読み、申し送り場面で医師に報告するためのSBAR形式に変換してください。

【厳守事項】
- 各要素50語以内で簡潔に記述
- SOAPに記載されていない情報は追加しない
- 緊急度に応じてRecommendationを記述

【SOAP記録】
{soap_text}

【出力形式】
S: 
B: 
A: 
R: """

# Model configuration. OpenAI and Anthropic use explicit snapshot IDs
# for reproducibility; Gemini uses alias because Google does not expose
# dated snapshots in the same way.
# Verify these IDs against vendor documentation before running;
# see scripts/pilot_check.py.
MODELS = [
    {"key": "gpt",    "vendor": "openai",    "name": "gpt-4o-2024-08-06",        "temps": [0.0, 1.0]},
    {"key": "claude", "vendor": "anthropic", "name": "claude-opus-4-5-20251101", "temps": [0.0, 1.0]},
    {"key": "gemini", "vendor": "google",    "name": "gemini-2.5-flash-lite",    "temps": [0.0, 1.0]},
]

N_TRIALS = 30
MAX_TOKENS = 400

# Reusable HTTP session for Gemini (TCP/TLS connection reuse)
_gemini_session = requests.Session()

OUTPUT_FIELDS = [
    "case_id", "model", "model_name", "resolved_model",
    "temperature", "trial",
    "sbar_text", "is_valid", "error", "retry_count", "elapsed_sec",
]


# ── Vendor-specific callers ──────────────────────────────────────────

def call_openai(model_name, system, user, temperature):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=MAX_TOKENS,
    )
    return resp.choices[0].message.content.strip(), resp.model


def call_anthropic(model_name, system, user, temperature):
    from anthropic import Anthropic
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model=model_name,
        system=system,
        messages=[{"role": "user", "content": user}],
        temperature=temperature,
        max_tokens=MAX_TOKENS,
    )
    return resp.content[0].text.strip(), resp.model


def call_gemini(model_name, system, user, temperature):
    """Gemini via REST (matches NCI paper). Uses session for connection reuse."""
    api_key = os.environ["GEMINI_API_KEY"]
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_name}:generateContent?key={api_key}"
    )
    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"parts": [{"text": user}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": MAX_TOKENS,
        },
    }
    r = _gemini_session.post(url, json=payload, timeout=60)
    if not r.ok:
        # Surface Google's detailed error message instead of the generic
        # 'Bad Request' that requests.raise_for_status() would emit.
        try:
            err_body = r.json()
        except Exception:
            err_body = r.text[:500]
        raise RuntimeError(
            f"Gemini API error {r.status_code}: {err_body}"
        )
    text = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    return text, model_name


CALLERS = {
    "openai": call_openai,
    "anthropic": call_anthropic,
    "google": call_gemini,
}


# ── Scenario loading ─────────────────────────────────────────────────

def load_scenarios():
    scenarios = []
    for path in sorted(SCENARIOS_DIR.glob("*.txt")):
        case_id = path.stem.split("_")[0]
        text = path.read_text(encoding="utf-8")
        scenarios.append({"case_id": case_id, "soap_text": text})
    return scenarios


# ── Single call ──────────────────────────────────────────────────────

def run_one(spec):
    """Execute one API call with retry. Returns a result dict suitable for CSV."""
    model = spec["model"]
    user = USER_PROMPT_TEMPLATE.format(soap_text=spec["soap_text"])
    t0 = time.time()
    result, retry_count, error = call_with_retry(
        CALLERS[model["vendor"]],
        model["name"], SYSTEM_PROMPT, user, spec["temperature"],
        max_retries=2, base_backoff=1.0,
    )
    elapsed = time.time() - t0
    if result is None:
        sbar, resolved = "", ""
        is_valid = False
    else:
        sbar, resolved = result
        is_valid = bool(sbar)
    return {
        "case_id": spec["case_id"],
        "model": model["key"],
        "model_name": model["name"],
        "resolved_model": resolved,
        "temperature": spec["temperature"],
        "trial": spec["trial"],
        "sbar_text": sbar,
        "is_valid": is_valid,
        "error": error if not is_valid else "",
        "retry_count": retry_count,
        "elapsed_sec": round(elapsed, 2),
    }


# ── Per-vendor execution ─────────────────────────────────────────────

def run_vendor(vendor_name, vendor_specs, writer, n_workers, label_prefix=""):
    """Run all calls for one vendor with its own thread pool.

    This isolates rate-limit pressure to per-vendor budgets, so a slow
    Anthropic does not block fast Gemini.
    """
    n_total = len(vendor_specs)
    if n_total == 0:
        return 0, 0
    print(f"  [{vendor_name}] {n_total} calls with {n_workers} workers")
    n_success, n_failure = 0, 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(run_one, s): i for i, s in enumerate(vendor_specs)}
        for k, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            writer.write_row(row)
            if row["is_valid"]:
                n_success += 1
            else:
                n_failure += 1
            if k % 50 == 0 or k == n_total:
                elapsed = time.time() - t0
                rate = k / elapsed if elapsed > 0 else 0
                eta_min = (n_total - k) / rate / 60 if rate > 0 else 0
                print(f"    {label_prefix}[{vendor_name}] [{k:>5}/{n_total}] "
                      f"{rate:.1f}/s, ok={n_success}, fail={n_failure}, "
                      f"ETA {eta_min:.1f}min")
    return n_success, n_failure


# ── Main ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="Show plan without calling APIs")
    ap.add_argument("--restart", action="store_true",
                    help="Delete existing output file and start fresh")
    ap.add_argument("--out", default=str(DATA / "raw_sbars.csv"))
    ap.add_argument("--workers-openai", type=int,
                    default=VENDOR_PARALLEL["openai"])
    ap.add_argument("--workers-anthropic", type=int,
                    default=VENDOR_PARALLEL["anthropic"])
    ap.add_argument("--workers-google", type=int,
                    default=VENDOR_PARALLEL["google"])
    args = ap.parse_args()

    out_path = Path(args.out)

    if args.restart and out_path.exists():
        print(f"Removing existing {out_path} (--restart)")
        out_path.unlink()

    # Build full task list
    scenarios = load_scenarios()
    if len(scenarios) != 9:
        print(f"WARNING: expected 9 scenarios, found {len(scenarios)}",
              file=sys.stderr)

    full_specs = []
    for sc in scenarios:
        for model in MODELS:
            for temp in model["temps"]:
                for trial in range(1, N_TRIALS + 1):
                    full_specs.append({
                        "case_id": sc["case_id"],
                        "soap_text": sc["soap_text"],
                        "model": model,
                        "temperature": temp,
                        "trial": trial,
                    })

    # Determine which cells are already completed
    key_cols = ["case_id", "model", "temperature", "trial"]
    completed = load_completed_keys_with_filter(
        out_path, key_cols,
        success_filter=lambda row: row.get("is_valid", "").lower() == "true",
    )
    print(f"Total planned API calls: {len(full_specs):,}")
    print(f"Already completed (skipping): {len(completed):,}")

    pending = [
        s for s in full_specs
        if normalise_key(s["case_id"], s["model"]["key"],
                         s["temperature"], s["trial"]) not in completed
    ]
    print(f"Pending: {len(pending):,}")

    if args.dry_run:
        print(f"\nDry-run mode; no API calls made.")
        # Show pending breakdown by vendor
        by_vendor = {}
        for s in pending:
            v = s["model"]["vendor"]
            by_vendor[v] = by_vendor.get(v, 0) + 1
        for v, n in sorted(by_vendor.items()):
            workers = getattr(args, f"workers_{v}",
                             VENDOR_PARALLEL.get(v, 4))
            print(f"  {v:10s}: {n:>5} calls, {workers} workers")
        return

    if not pending:
        print("\nAll cells already completed. Nothing to do.")
        return

    # API key check (only for vendors that have pending work)
    pending_vendors = {s["model"]["vendor"] for s in pending}
    required_keys = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GEMINI_API_KEY",
    }
    for v in pending_vendors:
        if not os.environ.get(required_keys[v]):
            print(f"ERROR: ${required_keys[v]} not set.", file=sys.stderr)
            sys.exit(1)

    # Open CSV writer (append mode if file exists)
    writer = ResumableCSVWriter(out_path, OUTPUT_FIELDS)

    # Group pending specs by vendor
    by_vendor = {}
    for s in pending:
        by_vendor.setdefault(s["model"]["vendor"], []).append(s)

    # Run each vendor in its own thread pool, vendors in parallel
    print(f"\nLaunching per-vendor pools...")
    vendor_workers = {
        "openai": args.workers_openai,
        "anthropic": args.workers_anthropic,
        "google": args.workers_google,
    }

    overall_t0 = time.time()
    # Run vendors concurrently using one outer ThreadPool
    vendor_summary = {}
    with ThreadPoolExecutor(max_workers=len(by_vendor)) as outer:
        futures = {
            outer.submit(
                run_vendor, v, specs, writer, vendor_workers[v]
            ): v
            for v, specs in by_vendor.items()
        }
        for fut in as_completed(futures):
            v = futures[fut]
            try:
                vendor_summary[v] = fut.result()
            except Exception as exc:
                print(f"  [{v}] vendor pool crashed: {exc}", file=sys.stderr)
                vendor_summary[v] = (0, len(by_vendor[v]))

    writer.close()
    overall_elapsed = time.time() - overall_t0

    print(f"\n{'=' * 60}")
    print(f"Run complete in {overall_elapsed/60:.1f} minutes")
    print(f"{'=' * 60}")
    total_ok, total_fail = 0, 0
    for v, (ok, fail) in sorted(vendor_summary.items()):
        print(f"  {v:10s}: {ok} ok, {fail} failed")
        total_ok += ok
        total_fail += fail
    print(f"  TOTAL    : {total_ok} ok, {total_fail} failed")
    if total_fail > 0:
        print(f"\nNote: {total_fail} calls failed. Re-run this script to retry "
              f"failed cells (without --restart).")


if __name__ == "__main__":
    main()
