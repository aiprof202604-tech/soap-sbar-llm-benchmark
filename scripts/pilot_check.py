"""
Pilot check script.

Verifies that all three commercial LLMs respond correctly at both
sampling temperatures used in the main experiment (0.0 and 1.0).
Use this BEFORE running the main experiment to catch issues like:
- model alias / snapshot ID changes
- API key authentication failures
- temperature parameter rejection (Anthropic's >1.0 cap)
- network / endpoint changes

Output:
    Console output showing the resolved model name and a short
    sample of the SBAR text returned by each (vendor x temperature)
    cell. 6 cells total (3 vendors x 2 temperatures).

Usage:
    python scripts/pilot_check.py             # all 3 vendors x 2 temps
    python scripts/pilot_check.py --vendor openai     # one vendor only
    python scripts/pilot_check.py --temp 0.0          # one temperature
    python scripts/pilot_check.py --case A1           # specific scenario

Estimated cost: ~$0.10 per call (at most 6 calls = ~$0.60 / ~¥100)
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Ensure scripts/ is importable when running from the repo root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_experiment import (  # noqa: E402
    MODELS, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE,
    call_openai, call_anthropic, call_gemini,
)

CALLERS = {
    "openai": call_openai,
    "anthropic": call_anthropic,
    "google": call_gemini,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vendor", choices=["openai", "anthropic", "google", "all"],
                    default="all", help="Which vendor(s) to test (default: all)")
    ap.add_argument("--temp", type=float, choices=[0.0, 1.0],
                    default=None, help="Specific temperature only (default: both)")
    ap.add_argument("--case", default="A1",
                    help="Scenario case_id (default: A1)")
    ap.add_argument("--max-preview", type=int, default=300,
                    help="Max characters of SBAR to preview")
    args = ap.parse_args()

    # Load the test scenario
    scenario_files = list((ROOT / "data" / "scenarios").glob(f"{args.case}_*.txt"))
    if not scenario_files:
        print(f"ERROR: scenario for case {args.case} not found.", file=sys.stderr)
        sys.exit(1)
    soap_text = scenario_files[0].read_text(encoding="utf-8")
    user_prompt = USER_PROMPT_TEMPLATE.format(soap_text=soap_text)
    print(f"Test scenario: {scenario_files[0].name}")
    print(f"Source SOAP length: {len(soap_text)} characters\n")

    # API key check
    required_keys = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GEMINI_API_KEY",
    }
    vendors_to_test = list(required_keys.keys()) if args.vendor == "all" else [args.vendor]
    for v in vendors_to_test:
        if not os.environ.get(required_keys[v]):
            print(f"ERROR: ${required_keys[v]} not set.", file=sys.stderr)
            sys.exit(1)
    temps_to_test = [args.temp] if args.temp is not None else [0.0, 1.0]

    # Build test grid
    print("=" * 70)
    print(f"Pilot check: {len(vendors_to_test)} vendors × {len(temps_to_test)} temperatures = {len(vendors_to_test)*len(temps_to_test)} calls")
    print("=" * 70)

    n_success = 0
    n_failure = 0
    results = []

    for model_spec in MODELS:
        if model_spec["vendor"] not in vendors_to_test:
            continue
        for temp in temps_to_test:
            label = f"{model_spec['key']:6s} @ T={temp}"
            print(f"\n--- {label} ({model_spec['name']}) ---")
            t0 = time.time()
            try:
                text, resolved = CALLERS[model_spec["vendor"]](
                    model_spec["name"], SYSTEM_PROMPT, user_prompt, temp
                )
                elapsed = time.time() - t0
                if not text or len(text.strip()) < 10:
                    raise ValueError(f"empty or too short response: {text!r}")
                # quick sanity: SBAR pattern check (allows markdown formatting)
                # Claude often emits **S:**, **S（Situation）:**, etc.
                # We check for any of: 'S:', '**S:**', '**S（', 'S（', 'S (',
                # within the text.
                import re as _re
                sbar_marker_patterns = [
                    r"(?:^|\n)\s*\**\s*S\s*[:：(（]",
                    r"(?:^|\n)\s*\**\s*B\s*[:：(（]",
                    r"(?:^|\n)\s*\**\s*A\s*[:：(（]",
                    r"(?:^|\n)\s*\**\s*R\s*[:：(（]",
                ]
                contains_sbar_markers = sum(
                    1 for pat in sbar_marker_patterns
                    if _re.search(pat, text)
                )
                preview = text[:args.max_preview]
                if len(text) > args.max_preview:
                    preview += "...[truncated]"
                print(f"  ✓ resolved model: {resolved}")
                print(f"  ✓ response time: {elapsed:.1f}s")
                print(f"  ✓ response length: {len(text)} chars")
                print(f"  ✓ SBAR markers found: {contains_sbar_markers}/4")
                if contains_sbar_markers < 4:
                    print(f"  ⚠ WARNING: not all SBAR markers (S:/B:/A:/R:) detected")
                print(f"  --- preview ---")
                for line in preview.split("\n"):
                    print(f"  | {line}")
                n_success += 1
                results.append({"label": label, "ok": True})
            except Exception as exc:
                elapsed = time.time() - t0
                print(f"  ✗ FAILED after {elapsed:.1f}s: {type(exc).__name__}: {exc}")
                n_failure += 1
                results.append({"label": label, "ok": False, "error": str(exc)})

    # Summary
    print("\n" + "=" * 70)
    print(f"Summary: {n_success} successful, {n_failure} failed")
    print("=" * 70)
    for r in results:
        mark = "✓" if r["ok"] else "✗"
        line = f"  {mark} {r['label']}"
        if not r["ok"]:
            line += f"  ({r['error'][:60]})"
        print(line)

    if n_failure == 0:
        print("\nAll pilot checks passed. Safe to run the main experiment.")
        sys.exit(0)
    else:
        print("\nOne or more pilot checks failed.")
        print("Do NOT run the main experiment until all failures are resolved.")
        print("\nCommon causes:")
        print("  - Model alias/snapshot ID changed (check vendor's docs)")
        print("  - API key revoked or out of credit")
        print("  - Temperature parameter rejected (Anthropic max=1.0)")
        print("  - Network/firewall issue")
        sys.exit(1)


if __name__ == "__main__":
    main()
