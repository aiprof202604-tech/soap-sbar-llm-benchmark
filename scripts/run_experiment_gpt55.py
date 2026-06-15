"""
C1(A) add-on: generate GPT-5.5 SBARs under the SAME prompt/scenarios as the
original 3-model experiment, so the new model is evaluated by the identical
downstream pipeline (run_rule_based_check.py -> run_ai_judges.py ->
run_arbiter.py). Resumable, retry-aware, parallel, with progress display.

WHY A SEPARATE SCRIPT (not just a 4th entry in run_experiment.py):
    GPT-5.5 is a *reasoning* model. Per OpenAI's API docs it does NOT support
    temperature / top_p / max_tokens; it uses `reasoning_effort` and
    `max_output_tokens`, and works best on the Responses API. The original
    call_openai() (chat.completions + temperature + max_tokens=400) would
    therefore error or return empty (reasoning tokens consume a 400 budget).
    This script uses the Responses API with the correct parameters and leaves
    the validated run_experiment.py untouched.

DESIGN DECISIONS (documented for the manuscript Methods/Limitations):
    * No temperature sweep. GPT-5.5 has no temperature control, so all trials
      run at a fixed reasoning_effort (default: medium = GPT-5.5's own default).
      The `temperature` column is written as a NOMINAL 0.0 placeholder purely
      so the existing pipeline (which casts temperature to float) keeps working.
      It does NOT mean T=0.0 was requested; it cannot be.
    * Model key = "gpt55" (distinct from "gpt" = gpt-4o), so the new rows never
      collide with existing rows and downstream resume logic processes only the
      new model.
    * Default 60 trials/scenario to match the 60 generations/scenario the other
      models have (2 temps x 30 trials), keeping pooled cell counts balanced.
      Reduce with --trials if a smoke test shows GPT-5.5 output is near-identical
      across trials (reasoning models at fixed settings vary little).

PREREQUISITES:
    pip install -U openai            # Responses API needs a recent SDK
    set OPENAI_API_KEY=...           # Windows CMD: set (no quotes)
    Confirm your account has gpt-5.5 access and the exact model string
    (OpenAI dashboard / `python scripts/pilot_check.py` style check).

USAGE (Windows Command Prompt, run from the repo root):
    REM 0) ALWAYS back up the canonical file first (this script appends to it):
    copy data\raw_sbars.csv data\raw_sbars.backup.csv

    REM 1) Smoke test: 3 calls only, verify access + eyeball output:
    python scripts\run_experiment_gpt55.py --limit 3

    REM 2) Dry run: show the plan, no API calls:
    python scripts\run_experiment_gpt55.py --dry-run

    REM 3) Full run (resumable; re-run after any interruption to continue):
    python scripts\run_experiment_gpt55.py

    REM Optional overrides:
    python scripts\run_experiment_gpt55.py --trials 30 --reasoning-effort medium
    python scripts\run_experiment_gpt55.py --max-output-tokens 4000 --workers 8
"""

import argparse
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

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

# --- IDENTICAL prompt to run_experiment.py (do not change) ---
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

MODEL_KEY = "gpt55"           # distinct from "gpt" (gpt-4o)
NOMINAL_TEMPERATURE = 0.0     # placeholder only; GPT-5.5 has no temperature

OUTPUT_FIELDS = [
    "case_id", "model", "model_name", "resolved_model",
    "temperature", "trial",
    "sbar_text", "is_valid", "error", "retry_count", "elapsed_sec",
]


# ── GPT-5.5 caller (Responses API, reasoning model) ──────────────────

def call_gpt55(model_name, system, user, reasoning_effort, max_output_tokens):
    """Generate one SBAR via the Responses API. No temperature (unsupported)."""
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.responses.create(
        model=model_name,
        instructions=system,
        input=user,
        reasoning={"effort": reasoning_effort},
        max_output_tokens=max_output_tokens,
    )
    text = (getattr(resp, "output_text", None) or "").strip()
    if not text:
        status = getattr(resp, "status", None)
        reason = ""
        try:
            reason = resp.incomplete_details.reason
        except Exception:
            pass
        raise RuntimeError(
            f"empty_output (status={status}, reason={reason}); "
            f"try larger --max-output-tokens or different --reasoning-effort"
        )
    resolved = getattr(resp, "model", model_name) or model_name
    return text, resolved


def load_scenarios():
    scenarios = []
    for path in sorted(SCENARIOS_DIR.glob("*.txt")):
        case_id = path.stem.split("_")[0]
        scenarios.append({"case_id": case_id,
                          "soap_text": path.read_text(encoding="utf-8")})
    return scenarios


def run_one(spec):
    user = USER_PROMPT_TEMPLATE.format(soap_text=spec["soap_text"])
    t0 = time.time()
    result, retry_count, error = call_with_retry(
        call_gpt55,
        spec["model_name"], SYSTEM_PROMPT, user,
        spec["reasoning_effort"], spec["max_output_tokens"],
        max_retries=2, base_backoff=2.0,
    )
    elapsed = time.time() - t0
    if result is None:
        sbar, resolved, is_valid = "", "", False
    else:
        sbar, resolved = result
        is_valid = bool(sbar)
    return {
        "case_id": spec["case_id"],
        "model": MODEL_KEY,
        "model_name": spec["model_name"],
        "resolved_model": resolved,
        "temperature": NOMINAL_TEMPERATURE,
        "trial": spec["trial"],
        "sbar_text": sbar,
        "is_valid": is_valid,
        "error": error if not is_valid else "",
        "retry_count": retry_count,
        "elapsed_sec": round(elapsed, 2),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DATA / "raw_sbars.csv"),
                    help="appends here (back this file up first)")
    ap.add_argument("--model", default="gpt-5.5",
                    help="OpenAI model string (confirm exact id for your account)")
    ap.add_argument("--reasoning-effort", default="medium",
                    choices=["none", "low", "medium", "high", "xhigh"])
    ap.add_argument("--max-output-tokens", type=int, default=3000,
                    help="must cover reasoning + visible output; 400 is too low")
    ap.add_argument("--trials", type=int, default=60,
                    help="trials per scenario (others have 60 = 2 temps x 30)")
    ap.add_argument("--workers", type=int, default=VENDOR_PARALLEL["openai"])
    ap.add_argument("--limit", type=int, default=None,
                    help="process only first N calls (smoke test)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out)
    scenarios = load_scenarios()
    if len(scenarios) != 9:
        print(f"WARNING: expected 9 scenarios, found {len(scenarios)}",
              file=sys.stderr)

    full_specs = []
    for sc in scenarios:
        for trial in range(1, args.trials + 1):
            full_specs.append({
                "case_id": sc["case_id"],
                "soap_text": sc["soap_text"],
                "model_name": args.model,
                "reasoning_effort": args.reasoning_effort,
                "max_output_tokens": args.max_output_tokens,
                "trial": trial,
            })

    # Resume: skip gpt55 cells already completed successfully
    key_cols = ["case_id", "model", "temperature", "trial"]
    completed = load_completed_keys_with_filter(
        out_path, key_cols,
        success_filter=lambda row: (row.get("model") == MODEL_KEY
                                    and row.get("is_valid", "").lower() == "true"),
    )
    pending = [
        s for s in full_specs
        if normalise_key(s["case_id"], MODEL_KEY,
                         NOMINAL_TEMPERATURE, s["trial"]) not in completed
    ]
    print(f"Model: {args.model}  (key='{MODEL_KEY}', effort={args.reasoning_effort}, "
          f"max_output_tokens={args.max_output_tokens})")
    print(f"Planned gpt55 calls: {len(full_specs):,}  |  "
          f"already done: {len(completed):,}  |  pending: {len(pending):,}")

    if args.limit:
        pending = pending[:args.limit]
        print(f"--limit active: running only {len(pending)} call(s)")

    if args.dry_run:
        print("Dry-run; no API calls made.")
        return
    if not pending:
        print("All gpt55 cells already completed. Nothing to do.")
        return
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: $OPENAI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    writer = ResumableCSVWriter(out_path, OUTPUT_FIELDS)
    n_total, n_ok, n_fail = len(pending), 0, 0
    t0 = time.time()
    print(f"Generating {n_total} GPT-5.5 SBARs with {args.workers} workers...")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_one, s): i for i, s in enumerate(pending)}
        for k, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            writer.write_row(row)
            if row["is_valid"]:
                n_ok += 1
            else:
                n_fail += 1
                if n_fail <= 5:
                    print(f"    [fail] {row['case_id']} trial {row['trial']}: "
                          f"{row['error'][:120]}")
            if k % 25 == 0 or k == n_total:
                el = time.time() - t0
                rate = k / el if el > 0 else 0
                eta = (n_total - k) / rate / 60 if rate > 0 else 0
                print(f"  [{k:>4}/{n_total}] {rate:.2f}/s  ok={n_ok} fail={n_fail}  "
                      f"ETA {eta:.1f}min")
    writer.close()
    print(f"\nDone in {(time.time()-t0)/60:.1f} min: {n_ok} ok, {n_fail} failed.")
    if n_fail:
        print("Re-run the same command to retry failed cells (resumable).")
    else:
        print("Next: re-run run_rule_based_check.py, run_ai_judges.py, "
              "run_arbiter.py (resumable -> only gpt55 gets processed).")


if __name__ == "__main__":
    main()
