"""
Stage 3: Arbiter for Stage 2 inter-judge disagreements
(resumable, retry-aware).

For (SBAR x tag) cells where Stage 2 judges disagree, queries Claude
Opus 4.5 as a third independent judge. The final consensus score is
the majority vote across the three judges (ties resolved by median).

Resumable execution:
    Successfully arbitrated cells are loaded from the existing
    stage3_arbiter.csv at startup and skipped. Pass --restart to
    delete existing files and start fresh.

Retry policy:
    Up to 2 retries with exponential backoff for transient errors.

Anthropic Tier 1 rate limits (50 RPM for Claude Opus):
    Default 4 workers. Adjust with --workers if your account is
    on a higher tier.

Usage:
    python scripts/run_arbiter.py            # resume
    python scripts/run_arbiter.py --restart  # start fresh
    python scripts/run_arbiter.py --workers 8  # higher tier
"""

import argparse
import json
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

ARBITER_MODEL = "claude-opus-4-5-20251101"  # match generation snapshot
ARBITER_TEMPERATURE = 0.0
ARBITER_MAX_TOKENS = 200

ARBITER_SYSTEM = (
    "あなたは看護記録の評価専門家です。あるSOAP記録から変換された"
    "SBAR記録について、特定の事実情報の保存度合いを2人の評価者が"
    "異なる判定をしました。あなたは独立した第三者として、同じ判定を行います。"
    "臨床的妥当性は問わず、事実情報の転記忠実性のみを評価してください。"
)

ARBITER_USER_TEMPLATE = """【ソースSOAP】
{soap_text}

【変換後SBAR】
{sbar_text}

【判定対象タグ】
{tag_label}

【保存判定基準】
- 2 (fully_preserved): 概念と具体性が保たれている。同義語による言い換えを含む。
- 1 (partially_preserved): 言及はあるが具体性が大きく失われている。
- 0 (lost): 完全に欠落している。

【出力形式】
以下のJSON形式のみで応答してください。前置き・コードブロック・説明文は不要。

{{"score": 0|1|2}}
"""

ARBITER_FIELDS = [
    "case_id", "model", "temperature", "trial", "tag_id",
    "arbiter_score", "arbiter_success", "arbiter_error",
    "retry_count", "elapsed_sec",
]

FINAL_FIELDS = [
    "case_id", "model", "temperature", "trial", "tag_id",
    "score_gpt4o", "score_gemini", "arbiter_score",
    "agree", "arbiter_used", "final_score",
]


def call_claude_arbiter(system, user):
    from anthropic import Anthropic
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model=ARBITER_MODEL,
        system=system,
        messages=[{"role": "user", "content": user}],
        temperature=ARBITER_TEMPERATURE,
        max_tokens=ARBITER_MAX_TOKENS,
    )
    return resp.content[0].text


def parse_arbiter(text):
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
    try:
        data = json.loads(text)
        s = data.get("score")
        if s in (0, 1, 2):
            return int(s)
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def load_scenarios():
    return {
        path.stem.split("_")[0]: path.read_text(encoding="utf-8")
        for path in sorted(SCENARIOS_DIR.glob("*.txt"))
    }


def majority_vote(scores):
    valid = [s for s in scores if s is not None]
    if not valid:
        return None
    counts = {s: valid.count(s) for s in set(valid)}
    max_count = max(counts.values())
    winners = [s for s, c in counts.items() if c == max_count]
    if len(winners) == 1:
        return winners[0]
    return int(sorted(valid)[len(valid) // 2])


def arbiter_one(spec, writer):
    user = ARBITER_USER_TEMPLATE.format(
        soap_text=spec["soap_text"],
        sbar_text=spec["sbar_text"],
        tag_label=spec["tag_label_jp"],
    )
    t0 = time.time()
    raw, retry_count, error = call_with_retry(
        call_claude_arbiter, ARBITER_SYSTEM, user,
        max_retries=2, base_backoff=1.0,
    )
    elapsed = round(time.time() - t0, 2)
    if raw is None:
        score = None
        success = False
    else:
        score = parse_arbiter(raw)
        success = score is not None
        if not success and not error:
            error = "parse_failed"
    writer.write_row({
        "case_id": spec["case_id"],
        "model": spec["model"],
        "temperature": spec["temperature"],
        "trial": spec["trial"],
        "tag_id": spec["tag_id"],
        "arbiter_score": score if score is not None else "",
        "arbiter_success": success,
        "arbiter_error": error if not success else "",
        "retry_count": retry_count,
        "elapsed_sec": elapsed,
    })
    return success


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-sbars", default=str(DATA / "raw_sbars.csv"))
    ap.add_argument("--tags", default=str(DATA / "fact_tags.csv"))
    ap.add_argument("--gpt4o", default=str(DATA / "stage2_gpt4o.csv"))
    ap.add_argument("--gemini", default=str(DATA / "stage2_gemini.csv"))
    ap.add_argument("--out-arbiter", default=str(DATA / "stage3_arbiter_opus_sensitivity.csv"))
    ap.add_argument("--out-final", default=str(DATA / "final_scores_opus_sensitivity.csv"))
    ap.add_argument("--restart", action="store_true",
                    help="Delete existing stage3 and final outputs")
    ap.add_argument("--workers", type=int,
                    default=VENDOR_PARALLEL["anthropic"])
    args = ap.parse_args()

    out_arbiter = Path(args.out_arbiter)
    out_final = Path(args.out_final)

    if args.restart:
        for p in (out_arbiter, out_final):
            if p.exists():
                print(f"Removing {p}")
                p.unlink()

    for path in [args.in_sbars, args.tags, args.gpt4o, args.gemini]:
        if not Path(path).exists():
            print(f"ERROR: {path} not found.", file=sys.stderr)
            sys.exit(1)

    sbars = pd.read_csv(args.in_sbars)
    tags = pd.read_csv(args.tags)
    gpt = pd.read_csv(args.gpt4o)
    gem = pd.read_csv(args.gemini)

    # Filter to successful judgements only
    gpt = gpt[gpt["judge_success"] == True].copy()
    gem = gem[gem["judge_success"] == True].copy()
    gpt = gpt[gpt["score"].notna() & (gpt["score"] != "")].copy()
    gem = gem[gem["score"].notna() & (gem["score"] != "")].copy()
    gpt["score"] = gpt["score"].astype(int)
    gem["score"] = gem["score"].astype(int)

    # Normalise types for merge
    for df in (gpt, gem):
        df["temperature"] = df["temperature"].astype(float)
        df["trial"] = df["trial"].astype(int)

    key = ["case_id", "model", "temperature", "trial", "tag_id"]
    merged = gpt[key + ["score"]].merge(
        gem[key + ["score"]],
        on=key, how="outer",
        suffixes=("_gpt4o", "_gemini"),
    )
    merged["agree"] = (
        merged["score_gpt4o"].notna() &
        merged["score_gemini"].notna() &
        (merged["score_gpt4o"] == merged["score_gemini"])
    )
    disagree = merged[
        merged["score_gpt4o"].notna() &
        merged["score_gemini"].notna() &
        (~merged["agree"])
    ].copy()

    # === SENSITIVITY ANALYSIS: stratified random sample of 500 cells ===
    # Strata: model x temperature x preservation_difficulty
    # Proportional allocation, fixed RNG seed for reproducibility.
    import numpy as np
    RNG = np.random.default_rng(42)
    SAMPLE_N = 500
    diff_lookup = tags.set_index("tag_id")["preservation_difficulty"].to_dict()
    disagree["difficulty"] = disagree["tag_id"].map(diff_lookup)
    total = len(disagree)
    parts = []
    for _name, group in disagree.groupby(["model", "temperature", "difficulty"]):
        n_alloc = max(1, round(SAMPLE_N * len(group) / total))
        take = min(n_alloc, len(group))
        parts.append(group.sample(n=take,
                                  random_state=int(RNG.integers(0, 2**31))))
    sampled = pd.concat(parts).reset_index(drop=True)
    if len(sampled) > SAMPLE_N:
        sampled = sampled.sample(n=SAMPLE_N, random_state=42).reset_index(drop=True)
    elif len(sampled) < SAMPLE_N:
        rest = disagree[~disagree.index.isin(sampled.index)]
        extra = rest.sample(n=SAMPLE_N - len(sampled), random_state=42)
        sampled = pd.concat([sampled, extra]).reset_index(drop=True)
    print(f"Stratified sample for sensitivity: {len(sampled)} cells")
    print("Distribution by stratum:")
    print(sampled.groupby(["model", "temperature", "difficulty"]).size())
    disagree = sampled


    print(f"Total Stage 2 cells: {len(merged)}")
    print(f"Concordant cells:    {int(merged['agree'].sum())}")
    print(f"Disagreement cells:  {len(disagree)} (arbiter scope)")

    # Identify already-arbitrated cells (resumable)
    arbiter_keys = ["case_id", "model", "temperature", "trial", "tag_id"]
    completed = load_completed_keys_with_filter(
        out_arbiter, arbiter_keys,
        success_filter=lambda row: row.get("arbiter_success", "").lower() == "true",
    )
    print(f"Already arbitrated:  {len(completed):,}")

    # Build pending specs
    sbar_lookup = sbars.set_index(
        ["case_id", "model", "temperature", "trial"]
    )["sbar_text"].to_dict()
    tag_lookup = tags.set_index("tag_id")["tag_label_jp"].to_dict()
    scenarios = load_scenarios()

    pending = []
    for _, row in disagree.iterrows():
        # disagree DataFrame uses temperature as float; matching depends on type
        sbar_key = (row["case_id"], row["model"],
                    float(row["temperature"]), int(row["trial"]))
        sbar_text = sbar_lookup.get(sbar_key, "")
        if not sbar_text:
            continue
        check_key = normalise_key(row["case_id"], row["model"],
                                  row["temperature"], row["trial"], row["tag_id"])
        if check_key in completed:
            continue
        pending.append({
            "case_id": row["case_id"],
            "model": row["model"],
            "temperature": row["temperature"],
            "trial": row["trial"],
            "tag_id": row["tag_id"],
            "tag_label_jp": tag_lookup.get(row["tag_id"], ""),
            "soap_text": scenarios[row["case_id"]],
            "sbar_text": sbar_text,
        })

    print(f"Pending arbiter calls: {len(pending):,}")

    if pending and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: $ANTHROPIC_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    if pending:
        writer = ResumableCSVWriter(out_arbiter, ARBITER_FIELDS)
        print(f"Querying Claude Opus 4.5 (sensitivity) arbiter on {len(pending)} cells "
              f"(workers={args.workers})")
        n_success, n_failure = 0, 0
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(arbiter_one, s, writer): i
                       for i, s in enumerate(pending)}
            for k, fut in enumerate(as_completed(futures), 1):
                if fut.result():
                    n_success += 1
                else:
                    n_failure += 1
                if k % 25 == 0 or k == len(pending):
                    elapsed = time.time() - t0
                    rate = k / elapsed if elapsed > 0 else 0
                    eta = (len(pending) - k) / rate / 60 if rate > 0 else 0
                    print(f"  [{k:>4}/{len(pending)}] {rate:.1f}/s, "
                          f"ok={n_success}, fail={n_failure}, ETA {eta:.1f}min")
        writer.close()
        print(f"\nArbiter complete: {n_success} ok, {n_failure} failed")

    # Build final consensus table from current state
    print(f"\nBuilding final consensus table...")
    if out_arbiter.exists():
        arb = pd.read_csv(out_arbiter)
        arb = arb[arb["arbiter_success"] == True].copy()
        if len(arb) > 0:
            arb["arbiter_score"] = arb["arbiter_score"].astype(int)
            arb["temperature"] = arb["temperature"].astype(float)
            arb["trial"] = arb["trial"].astype(int)
            final = merged.merge(
                arb[key + ["arbiter_score"]],
                on=key, how="left",
            )
        else:
            final = merged.copy()
            final["arbiter_score"] = None
    else:
        final = merged.copy()
        final["arbiter_score"] = None

    def determine_final(row):
        scores = []
        for col in ("score_gpt4o", "score_gemini", "arbiter_score"):
            v = row.get(col)
            if v is not None and pd.notna(v):
                scores.append(int(v))
        return majority_vote(scores)

    final["final_score"] = final.apply(determine_final, axis=1)
    final["arbiter_used"] = final["arbiter_score"].notna()

    out_cols = key + ["score_gpt4o", "score_gemini", "arbiter_score",
                      "agree", "arbiter_used", "final_score"]
    final[out_cols].to_csv(out_final, index=False)
    print(f"Saved {len(final)} final-score rows -> {out_final}")
    if len(final) > 0:
        used = int(final["arbiter_used"].sum())
        print(f"  Arbiter intervention rate: {used}/{len(final)} "
              f"({used/len(final)*100:.1f}%)")
        n_final_valid = int(final["final_score"].notna().sum())
        print(f"  Final score determined for: {n_final_valid}/{len(final)} cells")


if __name__ == "__main__":
    main()
