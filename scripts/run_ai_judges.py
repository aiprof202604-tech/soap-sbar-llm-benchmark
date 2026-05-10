"""
Stage 2: AI judges for SOAP-to-SBAR preservation evaluation
(resumable, retry-aware).

Two judges (GPT-4o + Gemini 2.5 Flash-Lite) score each SBAR's
preservation of medium/hard tags independently. Stage 1 (easy tags)
is handled by run_rule_based_check.py and not re-judged here.

Resumable execution:
    Successfully judged (SBAR x judge) cells are loaded from each
    judge's output CSV at startup and skipped. Pass --restart to
    delete existing files and start fresh.

Retry policy (same as run_experiment.py):
    Up to 2 retries with exponential backoff for transient errors.

Per-vendor parallelism:
    GPT-4o: 12 workers (independent of generation phase)
    Gemini: 20 workers

Usage:
    python scripts/run_ai_judges.py            # resume from existing files
    python scripts/run_ai_judges.py --restart  # start fresh
    python scripts/run_ai_judges.py --judge gpt4o   # one judge only
    python scripts/run_ai_judges.py --limit 5  # first 5 SBARs only (testing)
"""

import argparse
import json
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

JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_TOKENS = 2000

JUDGE_SYSTEM = (
    "あなたは看護記録の評価専門家です。SOAP記録から変換されたSBAR記録について、"
    "ソースSOAPに含まれていた情報がSBARでどの程度保存されているかを判定し、"
    "ハルシネーション（ソースに存在しない情報の付加）も検出します。"
    "臨床的妥当性は問わず、事実情報の転記忠実性のみを評価してください。"
)

JUDGE_USER_TEMPLATE = """【ソースSOAP】
{soap_text}

【変換後SBAR】
{sbar_text}

【判定対象タグ（{n_tags}項目）】
{tag_lines}

【保存判定基準】
- 2 (fully_preserved): 概念と具体性が保たれている。同義語による言い換えを含む。
- 1 (partially_preserved): 言及はあるが具体性が大きく失われている。
- 0 (lost): 完全に欠落している。

【ハルシネーション検出（binary）】
変換後SBARに、ソースSOAPに存在しない事実情報が付加されていれば、その文を抽出。
重大度判定は不要、検出のみ。

【出力形式】
以下のJSON形式のみで応答してください。前置き・コードブロック・説明文は不要。

{{
  "preservation": [
    {{"tag_id": "XX_NN", "score": 0|1|2}}
  ],
  "hallucinations": [
    {{"sbar_section": "S|B|A|R", "sentence": "問題の文"}}
  ]
}}
"""

# Reusable HTTP session for Gemini
_gemini_session = requests.Session()


# ── Judge call functions ─────────────────────────────────────────────

def call_gpt4o_judge(system, user):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model="gpt-4o-2024-08-06",  # match generation snapshot
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=JUDGE_TEMPERATURE,
        max_tokens=JUDGE_MAX_TOKENS,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content


def call_gemini_judge(system, user):
    api_key = os.environ["GEMINI_API_KEY"]
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-2.5-flash-lite:generateContent?key=" + api_key
    )
    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"parts": [{"text": user}]}],
        "generationConfig": {
            "temperature": JUDGE_TEMPERATURE,
            "maxOutputTokens": JUDGE_MAX_TOKENS,
            "responseMimeType": "application/json",
        },
    }
    r = _gemini_session.post(url, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


JUDGE_CALLERS = {
    "gpt4o": call_gpt4o_judge,
    "gemini": call_gemini_judge,
}

JUDGE_VENDOR = {
    "gpt4o": "openai",
    "gemini": "google",
}

JUDGE_API_KEY = {
    "gpt4o": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


# ── Helpers ──────────────────────────────────────────────────────────

def load_scenarios():
    return {
        path.stem.split("_")[0]: path.read_text(encoding="utf-8")
        for path in sorted(SCENARIOS_DIR.glob("*.txt"))
    }


def build_tag_lines(tags_for_case):
    return "\n".join(
        f'  - {row["tag_id"]}: 「{row["tag_label_jp"]}」'
        for _, row in tags_for_case.iterrows()
    )


def parse_response(text):
    if not text:
        return [], []
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
    try:
        data = json.loads(text)
        return data.get("preservation", []), data.get("hallucinations", [])
    except (json.JSONDecodeError, TypeError):
        return [], []


# ── Per-SBAR judge call (with retry) ────────────────────────────────

EVAL_FIELDS = [
    "case_id", "model", "temperature", "trial",
    "judge", "tag_id",
    "score", "judge_success", "judge_error",
    "retry_count", "elapsed_sec",
]
HALL_FIELDS = [
    "case_id", "model", "temperature", "trial",
    "judge", "sbar_section", "sentence",
]


def judge_one(spec, eval_writer, hall_writer):
    """Judge one SBAR against its target tags. Returns (n_eval_rows, n_hall_rows)."""
    user = JUDGE_USER_TEMPLATE.format(
        soap_text=spec["soap_text"],
        sbar_text=spec["sbar_text"],
        n_tags=spec["n_tags"],
        tag_lines=spec["tag_lines"],
    )
    t0 = time.time()
    raw, retry_count, error = call_with_retry(
        JUDGE_CALLERS[spec["judge"]],
        JUDGE_SYSTEM, user,
        max_retries=2, base_backoff=1.0,
    )
    elapsed = round(time.time() - t0, 2)

    if raw is None:
        preservation, hallucinations = [], []
        success = False
    else:
        preservation, hallucinations = parse_response(raw)
        success = bool(preservation)
        if not success and not error:
            error = "empty_or_unparseable_response"

    base = {
        "case_id": spec["case_id"],
        "model": spec["model"],
        "temperature": spec["temperature"],
        "trial": spec["trial"],
        "judge": spec["judge"],
    }
    n_eval, n_hall = 0, 0
    if preservation:
        for j in preservation:
            eval_writer.write_row({
                **base,
                "tag_id": j.get("tag_id", ""),
                "score": j.get("score", ""),
                "judge_success": True,
                "judge_error": "",
                "retry_count": retry_count,
                "elapsed_sec": elapsed,
            })
            n_eval += 1
    else:
        # write a single failure-marker row so resume logic knows
        # we attempted but failed
        eval_writer.write_row({
            **base,
            "tag_id": "",
            "score": "",
            "judge_success": False,
            "judge_error": error,
            "retry_count": retry_count,
            "elapsed_sec": elapsed,
        })
    for h in hallucinations:
        hall_writer.write_row({
            **base,
            "sbar_section": h.get("sbar_section", ""),
            "sentence": h.get("sentence", ""),
        })
        n_hall += 1
    return success, n_eval, n_hall


def run_judge(judge_name, sbars, scenarios, target_tags, args):
    """Run all SBAR judgements for one judge."""
    if not os.environ.get(JUDGE_API_KEY[judge_name]):
        print(f"ERROR: ${JUDGE_API_KEY[judge_name]} not set.", file=sys.stderr)
        return

    out_eval = DATA / f"stage2_{judge_name}.csv"
    out_hall = DATA / f"stage2_hallucinations_{judge_name}.csv"

    if args.restart:
        for p in (out_eval, out_hall):
            if p.exists():
                print(f"  Removing {p}")
                p.unlink()

    # Build pre-computed tag lines per case
    tag_lines_per_case = {
        cid: build_tag_lines(target_tags[target_tags["case_id"] == cid])
        for cid in target_tags["case_id"].unique()
    }
    n_tags_per_case = {
        cid: int((target_tags["case_id"] == cid).sum())
        for cid in target_tags["case_id"].unique()
    }

    # Determine which (sbar, judge) cells are already completed.
    # Completion definition: at least one row exists with judge_success=True
    # OR a stub failure row exists -- but for retry semantics we only count
    # success rows as "done".
    sbar_keys = ["case_id", "model", "temperature", "trial"]
    completed = load_completed_keys_with_filter(
        out_eval, sbar_keys,
        success_filter=lambda row: row.get("judge_success", "").lower() == "true",
    )
    print(f"\n[{judge_name}] Existing successes: {len(completed):,}")

    # Build pending specs
    pending = []
    for _, row in sbars.iterrows():
        key = normalise_key(row["case_id"], row["model"],
                            row["temperature"], row["trial"])
        if key in completed:
            continue
        case_id = row["case_id"]
        pending.append({
            "judge": judge_name,
            "case_id": case_id,
            "model": row["model"],
            "temperature": row["temperature"],
            "trial": row["trial"],
            "soap_text": scenarios[case_id],
            "sbar_text": row["sbar_text"],
            "tag_lines": tag_lines_per_case[case_id],
            "n_tags": n_tags_per_case[case_id],
        })
    if args.limit:
        pending = pending[:args.limit]
    print(f"[{judge_name}] Pending: {len(pending):,}")

    if not pending:
        print(f"[{judge_name}] Nothing to do.")
        return

    n_workers = VENDOR_PARALLEL[JUDGE_VENDOR[judge_name]]
    if args.workers is not None:
        n_workers = args.workers

    eval_writer = ResumableCSVWriter(out_eval, EVAL_FIELDS)
    hall_writer = ResumableCSVWriter(out_hall, HALL_FIELDS)

    print(f"[{judge_name}] Judging {len(pending)} SBARs (workers={n_workers})")
    n_success, n_failure = 0, 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {
            ex.submit(judge_one, s, eval_writer, hall_writer): i
            for i, s in enumerate(pending)
        }
        for k, fut in enumerate(as_completed(futures), 1):
            success, _n_eval, _n_hall = fut.result()
            if success:
                n_success += 1
            else:
                n_failure += 1
            if k % 50 == 0 or k == len(pending):
                elapsed = time.time() - t0
                rate = k / elapsed if elapsed > 0 else 0
                eta = (len(pending) - k) / rate / 60 if rate > 0 else 0
                print(f"  [{judge_name}] [{k:>5}/{len(pending)}] "
                      f"{rate:.1f}/s, ok={n_success}, fail={n_failure}, "
                      f"ETA {eta:.1f}min")

    eval_writer.close()
    hall_writer.close()
    print(f"[{judge_name}] Complete: {n_success} ok, {n_failure} failed")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_path", default=str(DATA / "raw_sbars.csv"))
    ap.add_argument("--tags", default=str(DATA / "fact_tags.csv"))
    ap.add_argument("--judge", choices=["gpt4o", "gemini", "both"], default="both")
    ap.add_argument("--restart", action="store_true",
                    help="Delete existing stage2_* files for the chosen judges")
    ap.add_argument("--workers", type=int, default=None,
                    help="Override default per-vendor worker count")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only first N SBARs (for testing)")
    args = ap.parse_args()

    if not Path(args.in_path).exists():
        print(f"ERROR: {args.in_path} not found.", file=sys.stderr)
        sys.exit(1)

    sbars = pd.read_csv(args.in_path)
    sbars = sbars[sbars["is_valid"] == True].copy()
    sbars["temperature"] = sbars["temperature"].astype(float)
    sbars["trial"] = sbars["trial"].astype(int)

    tags = pd.read_csv(args.tags)
    target_tags = tags[tags["preservation_difficulty"].isin(["medium", "hard"])].copy()
    print(f"Stage 2 tag scope: {len(target_tags)} of {len(tags)} total")
    print(f"Valid SBARs to judge: {len(sbars)}")

    scenarios = load_scenarios()
    judges_to_run = ["gpt4o", "gemini"] if args.judge == "both" else [args.judge]

    # Run judges concurrently (different vendors, no rate-limit interference)
    if len(judges_to_run) > 1:
        with ThreadPoolExecutor(max_workers=len(judges_to_run)) as outer:
            futures = [
                outer.submit(run_judge, j, sbars, scenarios, target_tags, args)
                for j in judges_to_run
            ]
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as exc:
                    print(f"  judge crashed: {exc}", file=sys.stderr)
    else:
        run_judge(judges_to_run[0], sbars, scenarios, target_tags, args)


if __name__ == "__main__":
    main()
