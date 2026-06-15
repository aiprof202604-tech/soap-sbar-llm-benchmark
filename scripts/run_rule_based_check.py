"""
Stage 1: Rule-based fact preservation check.

For each "easy" tag (numeric values, named drugs, named tests, etc.),
verify whether the corresponding fact appears verbatim or as a
recognised variant in the generated SBAR text. Output is fully
deterministic and requires no human or LLM judgement.

Algorithm:
    For each (SBAR x easy-tag) pair:
        1. Extract candidate patterns from the tag label.
        2. Check whether any pattern matches the SBAR text.
        3. Score 2 (preserved) if match; 0 (lost) if not.
       (Stage 1 does not produce score=1; partial preservation is
        the domain of Stage 2 AI judging.)

Usage:
    python scripts/run_rule_based_check.py
    python scripts/run_rule_based_check.py --limit 10

Outputs:
    data/stage1_rule_based.csv  (one row per (sbar x easy-tag) pair)
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


# ── Pattern extractors ───────────────────────────────────────────────

NUMERIC_PATTERN = re.compile(r"\d+(?:[\.,]\d+)?")
# Patterns of the form "X/Y" where Y is a denominator/scale max
# (e.g., NRS 4/10, FACIT-Sp 9/12). These are treated as X-only for matching
# because clinicians often drop the /Y part in the SBAR ("NRS 4").
# Excluded: blood-pressure-style "X/Y" (both values are clinically meaningful).
SCALE_FRACTION_PATTERN = re.compile(r"(\d+)\s*/\s*\d+")


def extract_numbers(text):
    """Return all numeric tokens from a string.

    Special case: for tokens of the form 'X/Y' that look like a score
    over a maximum (NRS, HADS, FACIT-Sp, etc.), only X is returned.
    Blood pressure values like '132/78mmHg' are preserved as-is because
    of the trailing 'mmHg' unit, which we detect heuristically.
    """
    if "mmhg" in text.lower() or "ｍｍｈｇ" in text.lower():
        # blood pressure-style fractions: keep both values
        return NUMERIC_PATTERN.findall(text)
    # Strip the denominator from scale-fraction tokens before extraction
    text_stripped = SCALE_FRACTION_PATTERN.sub(r"\1", text)
    return NUMERIC_PATTERN.findall(text_stripped)


def normalise_text(text):
    """Lowercase, strip whitespace, remove some punctuation for matching."""
    if not isinstance(text, str):
        return ""
    return text.lower().replace(" ", "").replace(",", "")


def numbers_match(tag_label, sbar_text):
    """Check whether all numeric tokens in tag_label appear in sbar_text.

    For tags like '体温37.2℃' (one number) or 'BNP 1420→285pg/mL' (two
    numbers), require all numbers to be present in SBAR for a positive
    match. This is conservative: if SBAR mentions only the latest BNP
    value, that still counts (since both are extracted), but if it
    mentions neither, the tag is judged 'lost'.
    """
    tag_numbers = extract_numbers(tag_label)
    if not tag_numbers:
        return None  # not numeric-based; fall back to keyword match
    sbar_norm = normalise_text(sbar_text)
    # All extracted numbers must be present
    return all(n in sbar_norm for n in tag_numbers)


# Selected drug / test / device tokens recognised as named entities.
# Patterns are case-insensitive after normalisation.
NAMED_ENTITIES = [
    # drugs
    "ロキソプロフェン", "loxoprofen",
    "ソルデム", "soldem",
    "セレコキシブ", "celecoxib",
    "オキシコドン", "oxycodone",
    "オキノーム", "oxynorm",
    "フェンタニル", "fentanyl",
    "g-csf", "ppi", "nsaids",
    # tests / scales
    "hba1c", "egfr", "bun", "cr", "bnp", "lvef", "hct",
    "pao2", "paco2", "hco3", "sp02", "spo2",
    "nrs", "mmrc", "hads", "facit-sp", "zarit", "jcs", "ppc", "pps",
    "k-l", "kellgren",
    # devices / interventions
    "hot", "tka", "smbg", "folfirinox",
]


def keyword_match(tag_label, sbar_text):
    """Check whether any named entity from the tag appears in SBAR."""
    tag_norm = normalise_text(tag_label)
    sbar_norm = normalise_text(sbar_text)
    matched_entities = [e for e in NAMED_ENTITIES if e in tag_norm and e in sbar_norm]
    if matched_entities:
        return True
    # Fallback: if the tag label contains no recognised entity, return None
    # so the caller can fall back to the numeric check or AI stage.
    has_known = any(e in tag_norm for e in NAMED_ENTITIES)
    if has_known:
        # tag had an entity but sbar didn't
        return False
    return None  # undecidable by rule


def rule_based_judge(tag_label, sbar_text):
    """Return ('match'|'no_match'|'undecidable', evidence_str)."""
    # Try numeric match first
    num_result = numbers_match(tag_label, sbar_text)
    if num_result is True:
        nums = extract_numbers(tag_label)
        return "match", f"all numbers found: {nums}"
    if num_result is False:
        nums = extract_numbers(tag_label)
        return "no_match", f"numbers absent: {nums}"
    # Fall back to keyword
    kw_result = keyword_match(tag_label, sbar_text)
    if kw_result is True:
        return "match", "named entity matched"
    if kw_result is False:
        return "no_match", "named entity absent"
    return "undecidable", "no rule applies (defer to Stage 2 AI judge)"


# ── Pipeline ─────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default=str(DATA / "raw_sbars.csv"))
    ap.add_argument("--tags", default=str(DATA / "fact_tags.csv"))
    ap.add_argument("--out", default=str(DATA / "stage1_rule_based.csv"))
    ap.add_argument("--restart", action="store_true",
                    help="Delete existing output before running (no-op for "
                         "this stage since the check is deterministic and "
                         "fast; provided for API consistency)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N SBARs (testing)")
    args = ap.parse_args()

    if not Path(args.in_path).exists():
        print(f"ERROR: {args.in_path} not found.", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out)
    if args.restart and out_path.exists():
        print(f"Removing existing {out_path}")
        out_path.unlink()

    sbars = pd.read_csv(args.in_path)
    sbars = sbars[sbars["is_valid"]].copy()
    if args.limit:
        sbars = sbars.head(args.limit)

    tags = pd.read_csv(args.tags)
    easy_tags = tags[tags["preservation_difficulty"] == "easy"].copy()
    print(f"Easy tags (Stage 1 scope): {len(easy_tags)} of {len(tags)} total")

    rows = []
    for _, sbar_row in sbars.iterrows():
        case_id = sbar_row["case_id"]
        case_easy = easy_tags[easy_tags["case_id"] == case_id]
        for _, tag_row in case_easy.iterrows():
            verdict, evidence = rule_based_judge(
                tag_row["tag_label_jp"], sbar_row["sbar_text"]
            )
            score = {"match": 2, "no_match": 0, "undecidable": None}[verdict]
            rows.append({
                "case_id": case_id,
                "model": sbar_row["model"],
                "temperature": sbar_row["temperature"],
                "trial": sbar_row["trial"],
                "tag_id": tag_row["tag_id"],
                "score": score,
                "verdict": verdict,
                "evidence": evidence,
                "stage": 1,
            })

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df):,} rows to {args.out}")
    if len(df) > 0:
        print(f"  match:        {(df['verdict']=='match').sum()}")
        print(f"  no_match:     {(df['verdict']=='no_match').sum()}")
        print(f"  undecidable:  {(df['verdict']=='undecidable').sum()}")


if __name__ == "__main__":
    main()
