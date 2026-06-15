# -*- coding: utf-8 -*-
r"""
M1: 独立評価者(rater2)を加えた信頼性解析
================================================================
評価者2(独立臨床家)の盲検採点・優先度割当が揃ったら本スクリプトを実行する。

算出する指標:
  [A] スコア信頼性 (0/1/2 保存度)  ※20-SBAR・400セルのGS部分集合
      A1 評価者間κ           : 著者 vs 評価者2 (unweighted/linear/quadratic, 完全一致%)
      A2 パイプライン vs 著者κ : (現行論文の κ=0.760 を再現)
      A3 パイプライン vs 評価者2κ
      A4 omit/retain 二値κ     : 各ペア (安全性の核となる二値判断)
      A5 パイプライン vs 合意セル(両評価者一致)の一致%
  [B] 優先度信頼性 (high/medium/low)  ※全180タグ
      B1 優先度κ             : 著者 vs 評価者2 (unweighted + ordinal weighted, 完全一致%)

依存: pandas, numpy のみ (scipy不要)。ローカル即時計算のため
      レジューム/並列は不要だが、各段階の進捗を明示し、入力検証を行う。

使い方 (Windowsコマンドプロンプト, 日本語パスOK):
  python analyze_interrater_kappa.py ^
     --rater2-scores   rater2_scores.csv ^
     --rater2-priority rater2_priority.csv

  ※ rater2_scores.csv   = annotation_answers_TEMPLATE.csv の your_score を埋めたもの
     rater2_priority.csv = priority_annotation_TEMPLATE.csv の your_priority を埋めたもの
  既定パスは下の DEFAULTS を参照(同じフォルダに06_dataのファイルを置けば引数なしで動く)。
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

# ---- 既定パス (必要に応じて --... で上書き) -------------------------------
DEFAULTS = dict(
    author_scores   = "data/human_answers.csv",            # author scoring (400 cells)
    rater2_scores   = "data/rater2_scores.csv",            # rater 2 (independent nurse) scoring
    priority_master = "data/fact_tags.csv",                # author priority (clinical_priority)
    rater2_priority = "data/rater2_priority.csv",          # rater 2 priority labels
    stage1          = "data/stage1_rule_based.csv",        # Stage 1 deterministic verdicts
    final_with_gap  = "data/final_scores_with_gap.csv",    # gap-fixed final consensus scores
)
PRI2INT = {"low": 0, "medium": 1, "high": 2}
JPRI = {"高": "high", "中": "medium", "低": "low"}

def _read_sheet(path, sheet):
    p = str(path).lower()
    if p.endswith((".xlsx", ".xlsm")):
        try:
            return pd.read_excel(path, sheet_name=sheet)
        except Exception:
            return pd.read_excel(path)
    return pd.read_csv(path)

def _find_col(df, names):
    for nm in names:
        if nm in df.columns:
            return nm
    return None

def load_score_answers(path):
    df = _read_sheet(path, "採点")
    sc = _find_col(df, ["your_score", "採点", "score"])
    if sc is None or "cell_id" not in df.columns:
        sys.exit(f"[!] 採点ファイルに cell_id 列と採点列が必要です: {path}")
    out = df[["cell_id"]].copy()
    out["your_score"] = df[sc]
    return out

def load_priority_answers(path):
    df = _read_sheet(path, "優先度")
    pc = _find_col(df, ["your_priority", "重要度", "priority"])
    if pc is None or "tag_id" not in df.columns:
        sys.exit(f"[!] 優先度ファイルに tag_id 列と重要度列が必要です: {path}")
    out = df[["tag_id"]].copy()
    out["your_priority"] = (df[pc].astype(str).str.strip()
                            .map(lambda v: JPRI.get(v, v.lower())))
    return out

# ---- 重み付きCohen κ (3カテゴリ; weights: None/linear/quadratic) -----------
def cohen_kappa(a, b, k=3, weights=None):
    a = np.asarray(a, int); b = np.asarray(b, int); n = len(a)
    if n == 0:
        return float("nan")
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[x, y] += 1
    O /= n
    r = O.sum(1); c = O.sum(0); E = np.outer(r, c)
    if weights is None:
        W = 1.0 - np.eye(k)
    else:
        W = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                W[i, j] = abs(i - j) if weights == "linear" else (i - j) ** 2
        if W.max() > 0:
            W /= W.max()
    po = 1 - (W * O).sum(); pe = 1 - (W * E).sum()
    return float("nan") if pe == 0 else 1 - (1 - po) / (1 - pe)

def boot_ci(a, b, fn, B=10000, seed=42):
    rng = np.random.default_rng(seed); a = np.asarray(a); b = np.asarray(b); n = len(a)
    if n == 0:
        return (float("nan"), float("nan"))
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        vals.append(fn(a[idx], b[idx]))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return float(lo), float(hi)

def pct_agree(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return float((a == b).mean() * 100) if len(a) else float("nan")

# ---- パイプライン最終スコアを (case,model,temp,trial,tag) で引けるmapに ----
def build_pipeline_map(stage1_path, final_path):
    s1 = pd.read_csv(stage1_path)
    fg = pd.read_csv(final_path)
    def key(df):
        return list(zip(df.case_id, df.model, df.temperature.astype(str),
                        df.trial.astype(str), df.tag_id))
    pm = {}
    # Stage1: 規則で判定できたセル(undecidable以外)
    s1d = s1[s1.get("verdict", "").astype(str) != "undecidable"]
    for kk, v in zip(key(s1d), s1d["score"]):
        if pd.notna(v):
            pm[kk] = int(v)
    # Stage2/3(gap込): final_score (med/hard + gapの3タグ)
    fgd = fg[fg["final_score"].notna()]
    for kk, v in zip(key(fgd), fgd["final_score"]):
        pm[kk] = int(v)   # Stage2/3が優先(重複時)
    return pm

def attach_pipeline(df, pm):
    df = df.copy()
    df["_key"] = list(zip(df.case_id, df.model, df.temperature.astype(str),
                          df.trial.astype(str), df.tag_id))
    df["pipeline"] = df["_key"].map(pm)
    return df

def report_pair(name, a, b, results):
    out = dict(
        n=int(len(a)),
        pct_agreement=round(pct_agree(a, b), 1),
        kappa_unweighted=round(cohen_kappa(a, b), 3),
        kappa_linear=round(cohen_kappa(a, b, weights="linear"), 3),
        kappa_quadratic=round(cohen_kappa(a, b, weights="quadratic"), 3),
    )
    lo, hi = boot_ci(a, b, lambda x, y: cohen_kappa(x, y))
    out["kappa_unw_95CI"] = [round(lo, 3), round(hi, 3)]
    # omit/retain二値 (0=omit, 1&2=retain)
    ad = (np.asarray(a) == 0).astype(int); bd = (np.asarray(b) == 0).astype(int)
    out["dichotomy_pct"] = round(pct_agree(ad, bd), 1)
    out["dichotomy_kappa"] = round(cohen_kappa(ad, bd, k=2), 3)
    results[name] = out
    print(f"  {name:28s}: n={out['n']:3d}  exact={out['pct_agreement']:5.1f}%  "
          f"κ={out['kappa_unweighted']:.3f} "
          f"(lin {out['kappa_linear']:.3f}, quad {out['kappa_quadratic']:.3f})  "
          f"二値κ={out['dichotomy_kappa']:.3f}")
    return out

def main():
    ap = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k.replace('_','-')}", default=v)
    ap.add_argument("--out", default="data/kappa_results.json")
    a = ap.parse_args()

    print("="*64)
    print(" M1 信頼性解析: 評価者間κ / パイプラインvs各評価者κ / 優先度κ")
    print("="*64)

    # ---- 入力読込 + 検証 -------------------------------------------------
    print("\n[1/4] 入力読込・検証 ...")
    auth = pd.read_csv(a.author_scores)
    r2_path = Path(a.rater2_scores)
    if not r2_path.exists():
        print(f"  [!] 評価者2の採点ファイルが未配置: {r2_path}")
        print("      M1_採点シート.xlsx の『採点』列を埋めて保存してください。")
        sys.exit(2)
    r2 = load_score_answers(r2_path)
    if r2["your_score"].isna().any() or (r2["your_score"].astype(str).str.strip() == "").any():
        n_blank = int(r2["your_score"].isna().sum() +
                      (r2["your_score"].astype(str).str.strip() == "").sum())
        print(f"  [!] 採点に未記入が {n_blank} 件あります。全セル記入後に再実行してください。")
        sys.exit(2)
    print(f"  著者採点: {len(auth)}セル / 評価者2採点: {len(r2)}セル")

    pm = build_pipeline_map(a.stage1, a.final_with_gap)
    print(f"  パイプライン最終スコアmap: {len(pm)}セル分を構築")

    # cell_id で著者・評価者2を突合
    merged = auth[["cell_id","case_id","model","temperature","trial","tag_id","priority","your_score"]] \
        .rename(columns={"your_score":"author"}).merge(
        r2[["cell_id","your_score"]].rename(columns={"your_score":"rater2"}),
        on="cell_id", how="inner")
    merged["author"] = merged["author"].astype(int)
    merged["rater2"] = merged["rater2"].astype(int)
    merged = attach_pipeline(merged, pm)
    n_nopipe = int(merged["pipeline"].isna().sum())
    if n_nopipe:
        print(f"  注: パイプラインスコア未取得 {n_nopipe}セル(突合キー不一致)→当該ペア計算から除外")

    results = {}

    # ---- [A] スコア信頼性 ------------------------------------------------
    print("\n[2/4] スコア信頼性 (0/1/2):")
    report_pair("A1 評価者間(著者 vs 評価者2)", merged["author"], merged["rater2"], results)
    mp = merged.dropna(subset=["pipeline"]).copy(); mp["pipeline"] = mp["pipeline"].astype(int)
    report_pair("A2 パイプライン vs 著者", mp["pipeline"], mp["author"], results)
    report_pair("A3 パイプライン vs 評価者2", mp["pipeline"], mp["rater2"], results)
    # A5 合意セル(両評価者一致)に対するパイプライン一致
    cons = mp[mp["author"] == mp["rater2"]]
    results["A5 合意セルでのパイプライン一致"] = dict(
        n=int(len(cons)),
        pct_pipeline_matches_consensus=round(pct_agree(cons["pipeline"], cons["author"]), 1))
    print(f"  {'A5 合意セルでのP一致':28s}: n={len(cons):3d}  "
          f"パイプライン一致={results['A5 合意セルでのパイプライン一致']['pct_pipeline_matches_consensus']:.1f}%")

    # ---- [B] 優先度信頼性 ------------------------------------------------
    print("\n[3/4] 優先度信頼性 (high/medium/low):")
    pri_path = Path(a.rater2_priority)
    if not pri_path.exists():
        print(f"  [!] 評価者2の優先度ファイル未配置: {pri_path} → 優先度κはスキップ")
    else:
        pa = pd.read_csv(a.priority_master)[["tag_id","clinical_priority"]] \
            .rename(columns={"clinical_priority":"author_pri"})
        pr = load_priority_answers(pri_path).rename(columns={"your_priority":"rater2_pri"})
        pj = pa.merge(pr, on="tag_id", how="inner")
        pj["rater2_pri"] = pj["rater2_pri"].astype(str).str.strip().str.lower()
        bad = pj[~pj["rater2_pri"].isin(PRI2INT)]
        if len(bad):
            print(f"  [!] 優先度に不正値 {len(bad)}件 (high/medium/lowのみ可) → 再確認してください。")
            print(bad.head().to_string())
        pj = pj[pj["rater2_pri"].isin(PRI2INT)]
        ai = pj["author_pri"].str.lower().map(PRI2INT).values
        ri = pj["rater2_pri"].map(PRI2INT).values
        out = dict(
            n=int(len(pj)),
            pct_agreement=round(pct_agree(ai, ri), 1),
            kappa_unweighted=round(cohen_kappa(ai, ri), 3),
            kappa_linear=round(cohen_kappa(ai, ri, weights="linear"), 3),
            kappa_quadratic=round(cohen_kappa(ai, ri, weights="quadratic"), 3),
        )
        lo, hi = boot_ci(ai, ri, lambda x, y: cohen_kappa(x, y))
        out["kappa_unw_95CI"] = [round(lo, 3), round(hi, 3)]
        results["B1 優先度κ(著者 vs 評価者2)"] = out
        print(f"  {'B1 優先度κ':28s}: n={out['n']:3d}  exact={out['pct_agreement']:5.1f}%  "
              f"κ={out['kappa_unweighted']:.3f} (lin {out['kappa_linear']:.3f}, "
              f"quad {out['kappa_quadratic']:.3f})  95%CI {out['kappa_unw_95CI']}")

    # ---- 保存 ------------------------------------------------------------
    print("\n[4/4] 結果保存 ...")
    Path(a.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  -> {a.out}")
    print("\n完了。原稿§Cに追記する数値: 上記 A1(評価者間κ) と B1(優先度κ) が中核。")

if __name__ == "__main__":
    main()
