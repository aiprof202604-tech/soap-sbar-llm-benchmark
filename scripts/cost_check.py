"""
Cost check script.

Two functions:

1. PRE-EXPERIMENT (no data yet):
   Display current price assumptions and the URLs where they should
   be re-verified. Estimate the cost of the planned experiment based
   on rough token counts. Run before the main experiment.

2. POST-EXPERIMENT (raw_sbars.csv exists):
   Estimate the actual cost of the completed generation phase by
   counting input/output tokens against current price assumptions.
   Compare with vendor billing dashboards to verify.

Usage:
    python scripts/cost_check.py             # Pre-experiment estimate
    python scripts/cost_check.py --post      # Post-experiment estimate
    python scripts/cost_check.py --verify    # Show only verification URLs

Note: Token counts in pre-experiment mode are rough estimates;
post-experiment mode uses approximate Japanese-to-token conversion
(0.5 tokens/character for input prompts, 0.5 tokens/character for
output SBARs). For exact billed cost, refer to each vendor's
billing dashboard.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRICES_PATH = ROOT / "data" / "prices.json"
SCENARIOS_DIR = ROOT / "data" / "scenarios"


def load_prices():
    if not PRICES_PATH.exists():
        print(f"ERROR: {PRICES_PATH} not found.", file=sys.stderr)
        sys.exit(1)
    with open(PRICES_PATH, encoding="utf-8") as f:
        return json.load(f)


def show_verification_urls(prices):
    print("=" * 70)
    print("Price verification URLs")
    print("=" * 70)
    for vendor, url in prices["_meta"]["verification_urls"].items():
        print(f"  {vendor:10s}: {url}")
    print()
    print(f"Last updated: {prices['_meta']['last_updated']}")
    print(f"JPY rate assumed: ¥{prices['_meta']['jpy_rate_estimate']}/USD")
    print()
    print("Action: open each URL in a browser, verify input_per_1m and")
    print("output_per_1m against data/prices.json. Update prices.json if")
    print("the rates differ. Re-run cost_check.py to recompute the")
    print("estimate.")


def estimate_tokens_per_call_pre():
    """Rough estimate of tokens per call before any data is collected."""
    # Average SOAP scenario length: ~600 characters in Japanese
    # System prompt + user prompt template overhead: ~200 characters
    # Total input: ~800 chars ≈ 400 tokens (Japanese ~2 chars/token)
    # Wait, actually for Japanese in OpenAI tokenisers, it's closer to
    # 1 token per character for kanji/kana. Let's use a conservative
    # estimate: 1 token per character for input and output.
    if not SCENARIOS_DIR.exists():
        avg_input_tokens = 800
    else:
        scenario_lengths = [
            len(p.read_text(encoding="utf-8"))
            for p in SCENARIOS_DIR.glob("*.txt")
        ]
        # input = system + user template + scenario; ~200 + scenario
        avg_scenario_chars = sum(scenario_lengths) / len(scenario_lengths) if scenario_lengths else 600
        avg_input_tokens = int((200 + avg_scenario_chars) * 1.0)
    # SBAR output: 4 sections × 50 words ≈ 200 chars ≈ 200 tokens
    # plus padding / formatting: cap at max_tokens (400)
    avg_output_tokens = 250
    return avg_input_tokens, avg_output_tokens


def cost_of_call(price, input_tokens, output_tokens):
    return (
        price["input_per_1m"] * input_tokens / 1_000_000 +
        price["output_per_1m"] * output_tokens / 1_000_000
    )


def pre_experiment_estimate(prices):
    print("=" * 70)
    print("Pre-experiment cost estimate")
    print("=" * 70)
    print()

    in_tok, out_tok = estimate_tokens_per_call_pre()
    print(f"Estimated tokens per generation call:")
    print(f"  input:  {in_tok}")
    print(f"  output: {out_tok}")
    print()

    n_calls_per_vendor = 9 * 2 * 30  # 540 calls per vendor
    print(f"Generation calls per vendor: {n_calls_per_vendor}")
    print()

    total_usd = 0.0
    print(f"{'Vendor':10s}  {'Model':35s}  {'$/call':>10s}  {'Total $':>12s}")
    print("-" * 75)
    for vendor in ["openai", "anthropic", "google"]:
        for model_id, price in prices[vendor].items():
            cost_per_call = cost_of_call(price, in_tok, out_tok)
            vendor_total = cost_per_call * n_calls_per_vendor
            print(f"{vendor:10s}  {model_id:35s}  {cost_per_call:>10.4f}  {vendor_total:>12.2f}")
            total_usd += vendor_total

    # Approximate Stage 2 + Stage 3 costs
    stage2_in_tok = 1500   # SOAP + SBAR + tag list + rubric
    stage2_out_tok = 800   # JSON with per-tag scores
    stage2_calls_per_judge = 540 * 3  # 1,620 SBARs per judge × ? — wait, this is per vendor
    # Each judge processes all 1,620 valid SBARs (regardless of which vendor produced them)
    stage2_calls_per_judge = 1620
    stage3_in_tok = 1500   # SOAP + SBAR + single tag
    stage3_out_tok = 100   # short JSON
    # Assume 20% of cells go to Stage 3 (rough)
    n_stage2_cells = 1620 * 92  # 92 medium+hard tags judged per SBAR — wait, no, judges do all tags at once
    stage3_calls_estimate = int(1620 * 0.2)  # 20% of SBARs trigger arbiter approximately

    # Get judge prices
    gpt_price = list(prices["openai"].values())[0]
    gem_price = list(prices["google"].values())[0]
    cla_price = list(prices["anthropic"].values())[0]

    stage2_gpt_cost = cost_of_call(gpt_price, stage2_in_tok, stage2_out_tok) * stage2_calls_per_judge
    stage2_gem_cost = cost_of_call(gem_price, stage2_in_tok, stage2_out_tok) * stage2_calls_per_judge
    stage3_cla_cost = cost_of_call(cla_price, stage3_in_tok, stage3_out_tok) * stage3_calls_estimate

    print()
    print("Estimated judge phase cost:")
    print(f"  Stage 2 GPT-4o (1,620 calls):     ${stage2_gpt_cost:>8.2f}")
    print(f"  Stage 2 Gemini (1,620 calls):     ${stage2_gem_cost:>8.2f}")
    print(f"  Stage 3 Claude (~{stage3_calls_estimate} calls):   ${stage3_cla_cost:>8.2f}")
    judge_total = stage2_gpt_cost + stage2_gem_cost + stage3_cla_cost
    print(f"  judge phase subtotal:              ${judge_total:>8.2f}")

    print()
    grand_total = total_usd + judge_total
    jpy_rate = prices["_meta"]["jpy_rate_estimate"]
    print(f"GRAND TOTAL (rough estimate):  ${grand_total:.2f}  ≈  ¥{grand_total*jpy_rate:.0f}")
    print()
    print("⚠ This is a pre-experiment rough estimate based on assumed")
    print("  token counts. Actual cost will depend on real prompt and")
    print("  response lengths, retries, and the actual Stage 3 trigger")
    print("  rate. Re-run with --post after the generation phase to")
    print("  recompute using actual data.")


def post_experiment_estimate(prices):
    """Estimate actual cost based on raw_sbars.csv (and stage2 if available)."""
    raw_path = ROOT / "data" / "raw_sbars.csv"
    if not raw_path.exists():
        print(f"ERROR: {raw_path} not found. Has the experiment been run?",
              file=sys.stderr)
        sys.exit(1)

    import pandas as pd
    df = pd.read_csv(raw_path)
    df_valid = df[df["is_valid"]].copy()
    print(f"Loaded {len(df)} rows; {len(df_valid)} valid generations.\n")

    # Use SBAR text length as a proxy for output tokens
    df_valid["sbar_chars"] = df_valid["sbar_text"].fillna("").str.len()
    print("SBAR length distribution per model:")
    print(df_valid.groupby("model")["sbar_chars"].describe().round(1).to_string())
    print()

    # Estimate input tokens from scenario file size + prompt overhead
    scenario_lens = {
        p.stem.split("_")[0]: len(p.read_text(encoding="utf-8"))
        for p in SCENARIOS_DIR.glob("*.txt")
    }
    # Add ~300 chars for SYSTEM_PROMPT + USER_PROMPT_TEMPLATE overhead
    df_valid["input_chars_est"] = df_valid["case_id"].map(scenario_lens) + 300

    total_usd = 0.0
    print(f"{'Model':10s}  {'N valid':>8s}  {'Input tok':>10s}  {'Output tok':>10s}  {'Cost USD':>10s}")
    print("-" * 60)
    for model_key, model_id in [
        ("gpt", list(prices["openai"].keys())[0]),
        ("claude", list(prices["anthropic"].keys())[0]),
        ("gemini", list(prices["google"].keys())[0]),
    ]:
        sub = df_valid[df_valid["model"] == model_key]
        if len(sub) == 0:
            continue
        # Approx: 1 char ≈ 1 token for Japanese text under modern tokenisers
        in_tok = sub["input_chars_est"].sum()
        out_tok = sub["sbar_chars"].sum()
        vendor = {"gpt": "openai", "claude": "anthropic", "gemini": "google"}[model_key]
        price = prices[vendor][model_id]
        cost = (price["input_per_1m"] * in_tok / 1_000_000 +
                price["output_per_1m"] * out_tok / 1_000_000)
        print(f"{model_key:10s}  {len(sub):>8d}  {int(in_tok):>10d}  {int(out_tok):>10d}  {cost:>10.4f}")
        total_usd += cost

    jpy_rate = prices["_meta"]["jpy_rate_estimate"]
    print(f"\nGeneration phase total: ${total_usd:.2f}  ≈  ¥{total_usd*jpy_rate:.0f}")
    print()
    print("⚠ This is an estimate based on character count as a proxy for")
    print("  tokens. For exact billed cost, check each vendor's billing")
    print("  dashboard:")
    for vendor, url in prices["_meta"]["verification_urls"].items():
        print(f"  - {vendor}: {url.replace('pricing', 'usage')}  (or vendor's dashboard)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--post", action="store_true",
                    help="Post-experiment cost estimate (requires raw_sbars.csv)")
    ap.add_argument("--verify", action="store_true",
                    help="Show only verification URLs and exit")
    args = ap.parse_args()

    prices = load_prices()

    if args.verify:
        show_verification_urls(prices)
        return

    show_verification_urls(prices)
    print()

    if args.post:
        post_experiment_estimate(prices)
    else:
        pre_experiment_estimate(prices)


if __name__ == "__main__":
    main()
