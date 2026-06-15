"""
C4 gap fix: evaluate the 3 easy-difficulty HIGH-PRIORITY tags that the
deterministic Stage 1 could not rule-match (C3_14, P2_10, P3_16) and that,
through a pipeline gap, were never forwarded to the Stage 2 judges.

This script closes that gap WITHOUT touching any other cell, reusing the
*identical* judge prompts, scoring rubric, arbiter, and majority-vote
consensus rule as the main pipeline (run_ai_judges.py / run_arbiter.py),
so the 3 added tags are evaluated on exactly the same terms as the other 85.

Pipeline (all stages resumable, parallel, with progress):
  Stage A  judge the gap tags with GPT-4o + Gemini   -> stage2_gap_{judge}.csv
  Stage B  arbitrate disagreements with Claude Haiku  -> stage3_gap_arbiter.csv
  Stage C  merge 720 gap cells into the existing      -> final_scores_with_gap.csv
           final_scores.csv  (original file is preserved; 22,080 -> 22,800)
  Stage D  recompute the 88-tag high-priority SCO and report whether the
           headline conclusion (GPT-4o outlier; GPT-5.5 ~ Claude ~ Gemini)
           is unchanged.

Environment (same keys the full run used):
  OPENAI_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY

Usage:
  python scripts/run_gap_fix.py                # run everything, resume as needed
  python scripts/run_gap_fix.py --dry-run      # build specs & report, no API calls
  python scripts/run_gap_fix.py --restart      # delete gap outputs and start fresh
  python scripts/run_gap_fix.py --stage judge  # judge | arbiter | merge | verify | all
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

# ── identical judge configuration to run_ai_judges.py ────────────────
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

import requests
_gemini_session = requests.Session()


def call_gpt4o_judge(system, user):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=JUDGE_TEMPERATURE,
        max_tokens=JUDGE_MAX_TOKENS,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content


def call_gemini_judge(system, user):
    api_key = os.environ["GEMINI_API_KEY"]
    url = ("https://generativelanguage.googleapis.com/v1beta/models/"
           "gemini-2.5-flash-lite:generateContent?key=" + api_key)
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


JUDGE_CALLERS = {"gpt4o": call_gpt4o_judge, "gemini": call_gemini_judge}
JUDGE_VENDOR = {"gpt4o": "openai", "gemini": "google"}
JUDGE_API_KEY = {"gpt4o": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}


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


# ── identical arbiter configuration to run_arbiter.py ────────────────
ARBITER_MODEL = "claude-haiku-4-5-20251001"
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


def call_claude_arbiter(system, user):
    from anthropic import Anthropic
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model=ARBITER_MODEL, system=system,
        messages=[{"role": "user", "content": user}],
        temperature=ARBITER_TEMPERATURE, max_tokens=ARBITER_MAX_TOKENS,
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


# ── helpers ──────────────────────────────────────────────────────────
def load_scenarios():
    return {path.stem.split("_")[0]: path.read_text(encoding="utf-8")
            for path in sorted(SCENARIOS_DIR.glob("*.txt"))}


EVAL_FIELDS = ["case_id", "model", "temperature", "trial", "judge", "tag_id",
               "score", "judge_success", "judge_error", "retry_count", "elapsed_sec"]
HALL_FIELDS = ["case_id", "model", "temperature", "trial", "judge",
               "sbar_section", "sentence"]
ARBITER_FIELDS = ["case_id", "model", "temperature", "trial", "tag_id",
                  "arbiter_score", "arbiter_success", "arbiter_error",
                  "retry_count", "elapsed_sec"]


def get_gap_targets(stage1_path, tags):
    """Derive the gap tags/cases from Stage 1 undecidable rows."""
    s1 = pd.read_csv(stage1_path)
    u = s1[s1["verdict"] == "undecidable"]
    gap_tags = sorted(u["tag_id"].unique())
    # case for each tag (each gap tag belongs to exactly one case)
    tag_case = {t: u[u["tag_id"] == t]["case_id"].iloc[0] for t in gap_tags}
    # high-priority check
    pri = dict(zip(tags["tag_id"], tags["clinical_priority"]))
    hi = [t for t in gap_tags if pri.get(t) == "high"]
    return gap_tags, tag_case, hi


# ── Stage A: judge gap tags ──────────────────────────────────────────
def judge_one(spec, ew, hw):
    user = JUDGE_USER_TEMPLATE.format(
        soap_text=spec["soap_text"], sbar_text=spec["sbar_text"],
        n_tags=spec["n_tags"], tag_lines=spec["tag_lines"])
    t0 = time.time()
    raw, rc, err = call_with_retry(JUDGE_CALLERS[spec["judge"]],
                                   JUDGE_SYSTEM, user, max_retries=2, base_backoff=1.0)
    elapsed = round(time.time() - t0, 2)
    if raw is None:
        preservation, hall = [], []
    else:
        preservation, hall = parse_response(raw)
    success = bool(preservation)
    if raw is not None and not success and not err:
        err = "empty_or_unparseable_response"
    base = {k: spec[k] for k in ("case_id", "model", "temperature", "trial")}
    base["judge"] = spec["judge"]
    if preservation:
        for j in preservation:
            ew.write_row({**base, "tag_id": j.get("tag_id", ""),
                          "score": j.get("score", ""), "judge_success": True,
                          "judge_error": "", "retry_count": rc, "elapsed_sec": elapsed})
    else:
        ew.write_row({**base, "tag_id": "", "score": "", "judge_success": False,
                      "judge_error": err, "retry_count": rc, "elapsed_sec": elapsed})
    for h in hall:
        hw.write_row({**base, "sbar_section": h.get("sbar_section", ""),
                      "sentence": h.get("sentence", "")})
    return success


def run_judge_gap(judge, sbars_gap, scenarios, tag_case, tag_label, args):
    if not os.environ.get(JUDGE_API_KEY[judge]):
        print(f"ERROR: ${JUDGE_API_KEY[judge]} not set.", file=sys.stderr)
        return
    out_eval = DATA / f"stage2_gap_{judge}.csv"
    out_hall = DATA / f"stage2_gap_hallucinations_{judge}.csv"
    if args.restart:
        for p in (out_eval, out_hall):
            if p.exists():
                print(f"  Removing {p}"); p.unlink()

    sbar_keys = ["case_id", "model", "temperature", "trial"]
    completed = load_completed_keys_with_filter(
        out_eval, sbar_keys,
        success_filter=lambda r: r.get("judge_success", "").lower() == "true")
    print(f"\n[{judge}] gap existing successes: {len(completed):,}")

    # reverse map: case -> list of its gap tags (a case may hold >1 gap tag)
    case_tags = {}
    for t, c in tag_case.items():
        case_tags.setdefault(c, []).append(t)
    pending = []
    for _, row in sbars_gap.iterrows():
        key = normalise_key(row["case_id"], row["model"], row["temperature"], row["trial"])
        if key in completed:
            continue
        cid = row["case_id"]
        gts = case_tags[cid]
        tag_lines = "\n".join(f'  - {gt}: 「{tag_label[gt]}」' for gt in gts)
        pending.append({
            "judge": judge, "case_id": cid, "model": row["model"],
            "temperature": row["temperature"], "trial": row["trial"],
            "soap_text": scenarios[cid], "sbar_text": row["sbar_text"],
            "tag_lines": tag_lines, "n_tags": len(gts),
        })
    print(f"[{judge}] gap pending: {len(pending):,}")
    if args.dry_run:
        print(f"[{judge}] DRY-RUN: would judge {len(pending)} gap SBARs"); return
    if not pending:
        print(f"[{judge}] nothing to do."); return

    nw = args.workers or VENDOR_PARALLEL[JUDGE_VENDOR[judge]]
    ew = ResumableCSVWriter(out_eval, EVAL_FIELDS)
    hw = ResumableCSVWriter(out_hall, HALL_FIELDS)
    print(f"[{judge}] judging {len(pending)} gap SBARs (workers={nw})")
    ok = fail = 0; t0 = time.time()
    with ThreadPoolExecutor(max_workers=nw) as ex:
        futs = {ex.submit(judge_one, s, ew, hw): i for i, s in enumerate(pending)}
        for k, fut in enumerate(as_completed(futs), 1):
            if fut.result(): ok += 1
            else: fail += 1
            if k % 25 == 0 or k == len(pending):
                el = time.time() - t0; rate = k/el if el > 0 else 0
                eta = (len(pending)-k)/rate/60 if rate > 0 else 0
                print(f"  [{judge}] [{k:>4}/{len(pending)}] {rate:.1f}/s "
                      f"ok={ok} fail={fail} ETA {eta:.1f}min")
    ew.close(); hw.close()
    print(f"[{judge}] gap complete: {ok} ok, {fail} failed")


def stage_judge(sbars, scenarios, tags, tag_case, tag_label, args):
    cases = set(tag_case.values())
    sbars_gap = sbars[sbars["case_id"].isin(cases)].copy()
    print(f"Gap cases: {sorted(cases)} | gap SBARs per judge: {len(sbars_gap)}")
    judges = ["gpt4o", "gemini"]
    if args.dry_run:
        for j in judges:
            run_judge_gap(j, sbars_gap, scenarios, tag_case, tag_label, args)
        return
    with ThreadPoolExecutor(max_workers=2) as outer:
        futs = [outer.submit(run_judge_gap, j, sbars_gap, scenarios,
                             tag_case, tag_label, args) for j in judges]
        for fut in as_completed(futs):
            try: fut.result()
            except Exception as e: print(f"  judge crashed: {e}", file=sys.stderr)


# ── Stage B: arbitrate gap disagreements ─────────────────────────────
def arbiter_one(spec, writer):
    user = ARBITER_USER_TEMPLATE.format(
        soap_text=spec["soap_text"], sbar_text=spec["sbar_text"],
        tag_label=spec["tag_label_jp"])
    t0 = time.time()
    raw, rc, err = call_with_retry(call_claude_arbiter, ARBITER_SYSTEM, user,
                                   max_retries=2, base_backoff=1.0)
    elapsed = round(time.time()-t0, 2)
    score = parse_arbiter(raw) if raw is not None else None
    success = score is not None
    if not success and not err:
        err = "parse_failed"
    writer.write_row({
        "case_id": spec["case_id"], "model": spec["model"],
        "temperature": spec["temperature"], "trial": spec["trial"],
        "tag_id": spec["tag_id"],
        "arbiter_score": score if score is not None else "",
        "arbiter_success": success, "arbiter_error": err if not success else "",
        "retry_count": rc, "elapsed_sec": elapsed})
    return success


def _load_judge_scores(path):
    df = pd.read_csv(path)
    df = df[df["judge_success"] == True].copy()
    df = df[df["score"].notna() & (df["score"] != "")].copy()
    df["score"] = df["score"].astype(int)
    df["temperature"] = df["temperature"].astype(float)
    df["trial"] = df["trial"].astype(int)
    return df


def build_gap_merged(tags):
    """Merge gap gpt4o + gemini judge scores; return merged df + key list."""
    gp = _load_judge_scores(DATA / "stage2_gap_gpt4o.csv")
    gm = _load_judge_scores(DATA / "stage2_gap_gemini.csv")
    key = ["case_id", "model", "temperature", "trial", "tag_id"]
    merged = gp[key+["score"]].merge(gm[key+["score"]], on=key, how="outer",
                                     suffixes=("_gpt4o", "_gemini"))
    merged["agree"] = (merged["score_gpt4o"].notna() & merged["score_gemini"].notna()
                       & (merged["score_gpt4o"] == merged["score_gemini"]))
    return merged, key


def stage_arbiter(sbars, tags, args):
    out_arb = DATA / "stage3_gap_arbiter.csv"
    if args.restart and out_arb.exists():
        print(f"Removing {out_arb}"); out_arb.unlink()
    for p in ("stage2_gap_gpt4o.csv", "stage2_gap_gemini.csv"):
        if not (DATA / p).exists():
            print(f"ERROR: {DATA/p} not found (run Stage A first).", file=sys.stderr)
            sys.exit(1)
    merged, key = build_gap_merged(tags)
    disagree = merged[merged["score_gpt4o"].notna() & merged["score_gemini"].notna()
                      & (~merged["agree"])].copy()
    print(f"\nGap Stage 2 cells: {len(merged)} | concordant: {int(merged['agree'].sum())} "
          f"| disagreements: {len(disagree)}")

    arb_keys = key
    completed = load_completed_keys_with_filter(
        out_arb, arb_keys,
        success_filter=lambda r: r.get("arbiter_success", "").lower() == "true")
    print(f"Already arbitrated: {len(completed):,}")

    sbar_lookup = sbars.set_index(["case_id", "model", "temperature", "trial"])["sbar_text"].to_dict()
    tag_lookup = tags.set_index("tag_id")["tag_label_jp"].to_dict()
    scenarios = load_scenarios()
    pending = []
    for _, row in disagree.iterrows():
        sk = (row["case_id"], row["model"], float(row["temperature"]), int(row["trial"]))
        st = sbar_lookup.get(sk, "")
        if not st:
            continue
        ck = normalise_key(row["case_id"], row["model"], row["temperature"],
                           row["trial"], row["tag_id"])
        if ck in completed:
            continue
        pending.append({"case_id": row["case_id"], "model": row["model"],
                        "temperature": row["temperature"], "trial": row["trial"],
                        "tag_id": row["tag_id"], "tag_label_jp": tag_lookup.get(row["tag_id"], ""),
                        "soap_text": scenarios[row["case_id"]], "sbar_text": st})
    print(f"Pending gap arbiter calls: {len(pending):,}")
    if args.dry_run:
        print("DRY-RUN: skipping arbiter API calls"); return
    if pending and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: $ANTHROPIC_API_KEY not set.", file=sys.stderr); sys.exit(1)
    if not pending:
        print("No pending arbiter calls."); return
    writer = ResumableCSVWriter(out_arb, ARBITER_FIELDS)
    nw = args.workers or VENDOR_PARALLEL["anthropic"]
    print(f"Querying Claude Haiku arbiter on {len(pending)} gap cells (workers={nw})")
    ok = fail = 0; t0 = time.time()
    with ThreadPoolExecutor(max_workers=nw) as ex:
        futs = {ex.submit(arbiter_one, s, writer): i for i, s in enumerate(pending)}
        for k, fut in enumerate(as_completed(futs), 1):
            if fut.result(): ok += 1
            else: fail += 1
            if k % 10 == 0 or k == len(pending):
                el = time.time()-t0; rate = k/el if el > 0 else 0
                eta = (len(pending)-k)/rate/60 if rate > 0 else 0
                print(f"  [{k:>4}/{len(pending)}] {rate:.1f}/s ok={ok} fail={fail} ETA {eta:.1f}min")
    writer.close()
    print(f"Gap arbiter complete: {ok} ok, {fail} failed")


# ── Stage C: merge gap cells into final_scores ───────────────────────
def stage_merge(tags, args):
    base_final = DATA / "final_scores.csv"
    out_final = DATA / "final_scores_with_gap.csv"
    if not base_final.exists():
        print(f"ERROR: {base_final} not found.", file=sys.stderr); sys.exit(1)
    merged, key = build_gap_merged(tags)

    out_arb = DATA / "stage3_gap_arbiter.csv"
    if out_arb.exists():
        arb = pd.read_csv(out_arb)
        arb = arb[arb["arbiter_success"] == True].copy()
        if len(arb) > 0:
            arb["arbiter_score"] = arb["arbiter_score"].astype(int)
            arb["temperature"] = arb["temperature"].astype(float)
            arb["trial"] = arb["trial"].astype(int)
            gap = merged.merge(arb[key+["arbiter_score"]], on=key, how="left")
        else:
            gap = merged.copy(); gap["arbiter_score"] = None
    else:
        gap = merged.copy(); gap["arbiter_score"] = None

    def det_final(row):
        sc = []
        for c in ("score_gpt4o", "score_gemini", "arbiter_score"):
            v = row.get(c)
            if v is not None and pd.notna(v):
                sc.append(int(v))
        return majority_vote(sc)
    gap["final_score"] = gap.apply(det_final, axis=1)
    gap["arbiter_used"] = gap["arbiter_score"].notna()
    out_cols = key + ["score_gpt4o", "score_gemini", "arbiter_score",
                      "agree", "arbiter_used", "final_score"]
    gap = gap[out_cols]

    base = pd.read_csv(base_final)
    # idempotent: drop any pre-existing gap rows, then add fresh
    gap_keytuples = set(map(tuple, gap[key].astype(str).values.tolist()))
    base_keys = base[key].astype(str)
    mask_dup = base_keys.apply(lambda r: tuple(r) in gap_keytuples, axis=1)
    base_clean = base[~mask_dup]
    out = pd.concat([base_clean, gap], ignore_index=True)
    n_gap_decided = int(gap["final_score"].notna().sum())
    print(f"\nBase final_scores rows: {len(base)}")
    print(f"Gap rows added:         {len(gap)} (decided: {n_gap_decided})")
    print(f"Combined rows:          {len(out)}  (expected {len(base)+len(gap)})")
    if args.dry_run:
        print("DRY-RUN: not writing output"); return
    out.to_csv(out_final, index=False)
    print(f"Saved -> {out_final}  (original {base_final.name} left untouched)")


# ── Stage D: verify 88-tag conclusion ────────────────────────────────
def stage_verify(tags, args):
    from scipy import stats as st
    import itertools, numpy as np
    out_final = DATA / "final_scores_with_gap.csv"
    if not out_final.exists():
        print(f"ERROR: {out_final} not found (run Stage C).", file=sys.stderr); sys.exit(1)
    pri = dict(zip(tags["tag_id"], tags["clinical_priority"]))
    s1 = pd.read_csv(DATA / "stage1_rule_based.csv")
    fs = pd.read_csv(out_final)
    MOD = ["gpt", "gpt55", "claude", "gemini"]
    LAB = {"gpt": "GPT-4o", "gpt55": "GPT-5.5", "claude": "Claude", "gemini": "Gemini"}

    s1h = s1[(s1.tag_id.map(pri) == "high") & (s1.verdict != "undecidable")][
        ["case_id", "model", "tag_id", "score"]].rename(columns={"score": "sc"})
    fsh = fs[fs.tag_id.map(pri) == "high"][
        ["case_id", "model", "tag_id", "final_score"]].rename(columns={"final_score": "sc"})
    hi = pd.concat([s1h, fsh], ignore_index=True)
    hi["sco"] = (hi.sc == 0).astype(int)
    ntags = hi.tag_id.nunique()
    print("\n" + "="*60)
    print(f"88-TAG VERIFICATION  (high-priority tags evaluated: {ntags})")
    print("="*60)
    print("Pooled high-priority SCO:")
    for m in MOD:
        sub = hi[hi.model == m]
        print(f"  {LAB[m]:8s}: {sub.sco.mean():.3f}  (tags={sub.tag_id.nunique()})")
    wide = (hi.groupby(["case_id", "model"])["sco"].mean().reset_index()
            .pivot(index="case_id", columns="model", values="sco")[MOD])
    chi2, p = st.friedmanchisquare(*[wide[m].values for m in MOD])
    n, k = wide.shape[0], 4
    print(f"\n4-model Friedman: chi2={chi2:.3f}, p={p:.5f}, W={chi2/(n*(k-1)):.3f}")
    print("Pairwise (Bonferroni x6):")
    concl_ok = True
    for i, j in itertools.combinations(MOD, 2):
        a, b = wide[i].values, wide[j].values
        w, pp = st.wilcoxon(a, b); pc = min(pp*6, 1.0)
        sig = "SIG" if pc < 0.05 else "ns"
        print(f"  {LAB[i]:7s}vs {LAB[j]:7s}: p_bonf={pc:.3f} [{sig}]")
        # conclusion check: GPT-4o > others SIG; others mutually ns
        if "gpt" in (i, j) and "gpt55" not in (i, j) and i != "gpt55" and j != "gpt55":
            pass
    # explicit conclusion test
    def pc_of(x, y):
        a, b = wide[x].values, wide[y].values
        return min(st.wilcoxon(a, b)[1]*6, 1.0)
    gpt_sig = all(pc_of("gpt", o) < 0.05 for o in ["gpt55", "claude", "gemini"])
    others_ns = all(pc_of(x, y) >= 0.05 for x, y in
                    [("gpt55", "claude"), ("gpt55", "gemini"), ("claude", "gemini")])
    print("\nCONCLUSION CHECK:")
    print(f"  GPT-4o significantly worse than all 3 others: {gpt_sig}")
    print(f"  GPT-5.5 ~ Claude ~ Gemini (mutually n.s.):      {others_ns}")
    if gpt_sig and others_ns:
        print("  => UNCHANGED: the 88-tag analysis reproduces the headline finding.")
    else:
        print("  => CHANGED: re-examine; report the 88-tag numbers exactly as computed.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-sbars", default=str(DATA / "raw_sbars.csv"))
    ap.add_argument("--tags", default=str(DATA / "fact_tags.csv"))
    ap.add_argument("--stage", choices=["judge", "arbiter", "merge", "verify", "all"],
                    default="all")
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    for p in (args.in_sbars, args.tags, DATA / "stage1_rule_based.csv"):
        if not Path(p).exists():
            print(f"ERROR: {p} not found.", file=sys.stderr); sys.exit(1)

    tags = pd.read_csv(args.tags)
    gap_tags, tag_case, hi_tags = get_gap_targets(DATA / "stage1_rule_based.csv", tags)
    tag_label = tags.set_index("tag_id")["tag_label_jp"].to_dict()
    print(f"Gap tags (Stage1 undecidable): {gap_tags}")
    print(f"  high-priority among them:    {hi_tags}")
    print(f"  tag -> case:                 {tag_case}")

    sbars = pd.read_csv(args.in_sbars)
    sbars = sbars[sbars["is_valid"] == True].copy()
    sbars["temperature"] = sbars["temperature"].astype(float)
    sbars["trial"] = sbars["trial"].astype(int)

    scenarios = load_scenarios()

    if args.stage in ("judge", "all"):
        print("\n########## STAGE A: judge gap tags ##########")
        stage_judge(sbars, scenarios, tags, tag_case, tag_label, args)
    if args.stage in ("arbiter", "all"):
        print("\n########## STAGE B: arbitrate gap disagreements ##########")
        stage_arbiter(sbars, tags, args)
    if args.stage in ("merge", "all"):
        print("\n########## STAGE C: merge into final_scores ##########")
        stage_merge(tags, args)
    if args.stage in ("verify", "all"):
        print("\n########## STAGE D: verify 88-tag conclusion ##########")
        if not args.dry_run:
            stage_verify(tags, args)
        else:
            print("DRY-RUN: skipping verification")


if __name__ == "__main__":
    main()
