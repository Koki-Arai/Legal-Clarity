#!/usr/bin/env python3
"""
01_descriptive_stats.py
=======================
"Legal Clarity and Collateral-Setting Behavior: Evidence from Real-Estate Registries"
Koki Arai — JSPS KAKENHI 23K01404

Purpose
-------
Generate Table 1 (descriptive statistics) and all supporting summary
numbers cited in Section 5.1 of the paper.

Inputs
------
data/registry_did.csv          (from 00_data_cleaning.py)
data/マクロ指標.csv             Macro-economic controls (BoJ rate, CPI, Nikkei)

Outputs
-------
results/descriptive_stats.csv  Panel A/B descriptive statistics (Table 1)
results/macro_stats.csv        Macro controls summary (Table A1)
results/monthly_totals.csv     Monthly aggregate series (for Figure 1)
results/prepost_change.csv     Pre/post mean change by category

Run
---
    python src/01_descriptive_stats.py
"""

#!/usr/bin/env python3
"""
descriptive_stats.py
──────────────────────────────────────────────────────────────────────────────
第5節「実証分析結果」記述統計 生成スクリプト
・registry_did.csv（DID用ワイド形式）を読み込み、
  論文第5節（記述統計節）に必要な数値をすべて計算して出力する。
・別途 マクロ指標.csv も読み込み、統制変数の記述統計を作成する。
・出力:
    descriptive_stats.csv   … Table 1 (パネル記述統計、処置前・後×処置・対照)
    macro_stats.csv         … Table A1 (マクロ変数記述統計)
    monthly_totals.csv      … 時系列推移グラフ用月次集計
    prepost_change.csv      … 処置前後変化率（差の差の記述的確認）
    コンソール              … 論文本文挿入用の数値を整形して表示

入力ファイル（同フォルダに置くこと）:
    registry_did.csv   ← registry_data_cleaning.py の出力
    マクロ指標.csv      ← マクロ経済指標
──────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── パス設定 ──────────────────────────────────────────────────────────────────
# Google Colab では適宜書き換えてください
DID_PATH   = "registry_did.csv"
MACRO_PATH = "マクロ指標.csv"
OUT_STATS  = "descriptive_stats.csv"
OUT_MACRO  = "macro_stats.csv"
OUT_MONTHLY = "monthly_totals.csv"
OUT_PREPOST = "prepost_change.csv"

SEP = "=" * 70

# ── 1. データ読み込み ─────────────────────────────────────────────────────────
print(f"\n{SEP}\nデータ読み込み\n{SEP}")
df = pd.read_csv(DID_PATH, encoding="utf-8-sig", parse_dates=["date"])
print(f"  DIDデータ: {len(df):,} 行 × {df.shape[1]} 列")
print(f"  期間: {df['date'].min().strftime('%Y-%m')} ～ {df['date'].max().strftime('%Y-%m')}")
print(f"  局数: {df['bureau_code'].nunique()}")

# ── 2. 変数定義・ラベル ───────────────────────────────────────────────────────
TREATMENT_VARS = ["mortgage", "root_mortgage"]          # コア処置群（2007-）
CONTROL_VARS   = ["sale", "inheritance_combined"]       # コア対照群（2007-）
EXT_TREAT      = ["leasehold"]                          # 拡張処置（2009-）
EXT_CTRL       = ["surface_right"]                      # 拡張対照（2009-）
ALL_VARS       = TREATMENT_VARS + CONTROL_VARS + EXT_TREAT + EXT_CTRL

VAR_LABELS = {
    "mortgage":             "Ordinary mortgage (抵当権設定)",
    "root_mortgage":        "Blanket mortgage (根抵当権設定)",
    "leasehold":            "Leasehold (賃借権設定)",
    "sale":                 "Sale transfer (売買)",
    "inheritance_combined": "Inheritance transfer (相続)",
    "surface_right":        "Surface right (地上権設定)",
}

# 処置前後ダミー（移行月除外）
PRE  = df["post"] == 0
POST = df["post"] == 1
TRANS = df["transition"] == 1
# verdict-only: 2023-12 ～ 2024-03
VERDICT_ONLY = df["post_verdict_only"] == 1
# reform: 2024-04 以降
REFORM = df["post_inheritance_reform"] == 1

# ─────────────────────────────────────────────────────────────────────────────
# 3. Table 1: パネル記述統計（処置前サンプル、全変数）
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}\nTable 1: パネル記述統計（処置前サンプル）\n{SEP}")

rows = []
for var in ALL_VARS:
    if var not in df.columns:
        print(f"  [SKIP] {var} — 列が存在しません")
        continue
    s_pre  = df.loc[PRE,  var].dropna()
    s_post = df.loc[POST, var].dropna()
    s_all  = df.loc[PRE | POST, var].dropna()
    ln_pre = np.log(s_pre.replace(0, np.nan)).dropna()

    is_treat = var in TREATMENT_VARS + EXT_TREAT

    rows.append({
        "variable":    VAR_LABELS.get(var, var),
        "group":       "Treatment" if is_treat else "Control",
        # 処置前
        "pre_N":       int(s_pre.count()),
        "pre_mean":    round(s_pre.mean(), 1),
        "pre_sd":      round(s_pre.std(),  1),
        "pre_p25":     round(s_pre.quantile(0.25), 1),
        "pre_median":  round(s_pre.median(), 1),
        "pre_p75":     round(s_pre.quantile(0.75), 1),
        "pre_min":     round(s_pre.min(), 1),
        "pre_max":     round(s_pre.max(), 1),
        # 処置後
        "post_N":      int(s_post.count()),
        "post_mean":   round(s_post.mean(), 1),
        "post_sd":     round(s_post.std(),  1),
        "post_median": round(s_post.median(), 1),
        # 対数（処置前）
        "ln_pre_mean": round(ln_pre.mean(), 4),
        "ln_pre_sd":   round(ln_pre.std(),  4),
        "ln_pre_median": round(ln_pre.median(), 4),
    })

    print(f"\n  [{VAR_LABELS.get(var, var)}]  group={'Treatment' if is_treat else 'Control'}")
    print(f"    処置前: N={s_pre.count():,}  mean={s_pre.mean():.1f}  "
          f"sd={s_pre.std():.1f}  p25={s_pre.quantile(0.25):.1f}  "
          f"median={s_pre.median():.1f}  p75={s_pre.quantile(0.75):.1f}")
    print(f"    処置後: N={s_post.count():,}  mean={s_post.mean():.1f}  "
          f"sd={s_post.std():.1f}  median={s_post.median():.1f}")
    print(f"    ln(処置前): mean={ln_pre.mean():.4f}  sd={ln_pre.std():.4f}  "
          f"median={ln_pre.median():.4f}")

stats_df = pd.DataFrame(rows)
stats_df.to_csv(OUT_STATS, index=False, encoding="utf-8-sig")
print(f"\n  → {OUT_STATS} を保存しました")

# ─────────────────────────────────────────────────────────────────────────────
# 4. パネル構造サマリー（論文本文に直接引用する数値）
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}\nパネル構造サマリー（論文本文引用用）\n{SEP}")

n_bureaus = df["bureau_code"].nunique()
n_months  = df["date"].nunique()
period_start = df["date"].min().strftime("%Y-%m")
period_end   = df["date"].max().strftime("%Y-%m")
n_pre  = df.loc[PRE,  "date"].nunique()
n_post = df.loc[POST, "date"].nunique()

print(f"  法務局数           : {n_bureaus}")
print(f"  月数（全期間）      : {n_months} ヶ月")
print(f"  期間               : {period_start} ～ {period_end}")
print(f"  処置前月数          : {n_pre} ヶ月")
print(f"  処置後月数（移行月除く）: {n_post} ヶ月")
print(f"  移行月（除外）      : 2023-11")

# コア4系列での総観測数
# ロング形式に変換して確認
df_long = pd.melt(
    df[df["transition"] == 0],
    id_vars=["date", "bureau_code", "post", "post_verdict_only",
             "post_inheritance_reform"],
    value_vars=[v for v in TREATMENT_VARS + CONTROL_VARS if v in df.columns],
    var_name="reg_type", value_name="count"
)
df_long["treatment"] = df_long["reg_type"].isin(TREATMENT_VARS).astype(int)

print(f"\n  コア4系列ロング形式:")
print(f"    全観測数  : {len(df_long):,}")
print(f"    処置群観測: {(df_long['treatment']==1).sum():,}")
print(f"    対照群観測: {(df_long['treatment']==0).sum():,}")
print(f"    処置前    : {(df_long['post']==0).sum():,}")
print(f"    処置後    : {(df_long['post']==1).sum():,}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. 規模別・ブロック別サンプル構造
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}\n規模別・ブロック別サンプル構造\n{SEP}")

# ブロック分類（局コード先頭2桁）
BLOCK_MAP = {
    51: "Hokkaido", 52: "Tohoku",  53: "Kanto",   54: "Chubu",
    55: "Kinki",    56: "Chugoku", 57: "Shikoku",  58: "Kyushu"
}
df["block"] = (df["bureau_code"].astype(int) // 1000).map(BLOCK_MAP)

# 規模分類（平均mortgage件数で三分位）
avg_mort = df.groupby("bureau_code")["mortgage"].mean()
q33, q67 = avg_mort.quantile(0.33), avg_mort.quantile(0.67)
size_map = {c: ("large" if v >= q67 else "medium" if v >= q33 else "small")
            for c, v in avg_mort.items()}
df["size_cat"] = df["bureau_code"].map(size_map)

print("\n  ブロック別局数:")
block_counts = df.groupby("block")["bureau_code"].nunique().sort_index()
for blk, n in block_counts.items():
    print(f"    {blk:12s}: {n} 局")

print(f"\n  規模別局数:")
size_counts = df.groupby("size_cat")["bureau_code"].nunique()
for sz in ["large", "medium", "small"]:
    print(f"    {sz:8s}: {size_counts.get(sz, 0)} 局")
    # 月次平均件数
    sub = df[df["size_cat"] == sz]
    if "mortgage" in sub.columns:
        m = sub.loc[PRE, "mortgage"].mean() if PRE.sum() > 0 else np.nan
        print(f"             mortgage 処置前月次平均: {m:.1f}")

# 各局の基本情報（局別平均処置前mortgage）
bureau_summary = (df[PRE]
    .groupby(["bureau_code", "bureau_name", "block", "size_cat"])
    .agg(
        mort_mean=("mortgage", "mean"),
        mort_median=("mortgage", "median"),
        sale_mean=("sale", "mean"),
        inherit_mean=("inheritance_combined", "mean"),
    )
    .reset_index()
    .round(1)
)
print(f"\n  局別サマリー行数: {len(bureau_summary)}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. 処置前後の変化率（記述的 DID の確認）
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}\n記述的 DID（処置前後平均変化率）\n{SEP}")

prepost_rows = []
for var in TREATMENT_VARS + CONTROL_VARS:
    if var not in df.columns:
        continue
    pre_mean  = df.loc[PRE,  var].mean()
    post_mean = df.loc[POST, var].mean()
    chg_pct   = 100 * (post_mean - pre_mean) / pre_mean if pre_mean > 0 else np.nan

    # ln版
    ln_pre  = df.loc[PRE,  var].apply(lambda x: np.log(x) if x > 0 else np.nan).mean()
    ln_post = df.loc[POST, var].apply(lambda x: np.log(x) if x > 0 else np.nan).mean()
    ln_chg  = ln_post - ln_pre

    is_treat = var in TREATMENT_VARS
    prepost_rows.append({
        "variable": VAR_LABELS.get(var, var),
        "group":    "Treatment" if is_treat else "Control",
        "pre_mean": round(pre_mean, 1),
        "post_mean": round(post_mean, 1),
        "pct_change": round(chg_pct, 2),
        "ln_pre":  round(ln_pre, 4),
        "ln_post": round(ln_post, 4),
        "ln_change": round(ln_chg, 4),
    })
    print(f"  {VAR_LABELS.get(var, var)[:40]:40s} "
          f"('{('Treatment' if is_treat else 'Control')}'): "
          f"前={pre_mean:8.1f}  後={post_mean:8.1f}  "
          f"変化={chg_pct:+.1f}%  ln差={ln_chg:+.4f}")

prepost_df = pd.DataFrame(prepost_rows)

# 記述的 DID（処置群変化 − 対照群変化）
treat_rows = prepost_df[prepost_df["group"] == "Treatment"]
ctrl_rows  = prepost_df[prepost_df["group"] == "Control"]
did_desc = treat_rows["ln_change"].mean() - ctrl_rows["ln_change"].mean()
print(f"\n  記述的 DID (ln差の平均): treat平均={treat_rows['ln_change'].mean():.4f}  "
      f"control平均={ctrl_rows['ln_change'].mean():.4f}  "
      f"DID={did_desc:.4f}")

prepost_df.to_csv(OUT_PREPOST, index=False, encoding="utf-8-sig")
print(f"  → {OUT_PREPOST} を保存しました")

# ─────────────────────────────────────────────────────────────────────────────
# 7. 月次時系列集計（Figure用・全国合計）
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}\n月次時系列集計（全国合計）\n{SEP}")

monthly_rows = []
for var in TREATMENT_VARS + CONTROL_VARS + EXT_TREAT + EXT_CTRL:
    if var not in df.columns:
        continue
    monthly = (df.groupby("date")[var]
               .agg(["sum", "mean", "median"])
               .rename(columns={"sum": "total", "mean": "mean_bureau",
                                 "median": "median_bureau"})
               .reset_index())
    monthly["reg_type"] = var
    monthly["ln_total"] = np.log(monthly["total"].replace(0, np.nan))
    monthly_rows.append(monthly)

monthly_df = pd.concat(monthly_rows, ignore_index=True)
monthly_df["group"] = monthly_df["reg_type"].apply(
    lambda x: "treatment" if x in TREATMENT_VARS + EXT_TREAT else "control"
)
monthly_df["post"] = (monthly_df["date"] >= "2023-12-01").astype(int)
monthly_df.to_csv(OUT_MONTHLY, index=False, encoding="utf-8-sig")
print(f"  → {OUT_MONTHLY} を保存しました  ({len(monthly_df):,} 行)")

# 直近の実数確認（論文本文用）
print("\n  処置前12ヶ月 vs 処置後（判決直後4ヶ月: 2023-12~2024-03）の月次平均比較:")
for var in TREATMENT_VARS + CONTROL_VARS:
    if var not in df.columns:
        continue
    pre12 = df.loc[
        (df["date"] >= "2022-12-01") & (df["date"] < "2023-11-01"), var
    ].mean()
    post4 = df.loc[
        (df["date"] >= "2023-12-01") & (df["date"] < "2024-04-01"), var
    ].mean()
    chg = 100 * (post4 - pre12) / pre12 if pre12 > 0 else np.nan
    print(f"    {VAR_LABELS.get(var, var)[:40]:40s}: "
          f"前12ヶ月平均={pre12:8.1f}  後4ヶ月平均={post4:8.1f}  変化={chg:+.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 8. マクロ変数記述統計
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}\nマクロ変数記述統計\n{SEP}")

try:
    def parse_ym(s):
        import re
        m = re.match(r"(\d{4})[-/年](\d{1,2})", str(s))
        if m:
            return pd.Timestamp(int(m.group(1)), int(m.group(2)), 1)
        return pd.NaT

    raw_m = pd.read_csv(MACRO_PATH, encoding="utf-8-sig", header=None)
    nikkei_dates = [parse_ym(raw_m.iloc[i, 0]) for i in range(1, min(239, len(raw_m)))]
    nikkei_vals  = [float(str(raw_m.iloc[i, 1]).replace(",", ""))
                    if str(raw_m.iloc[i, 1]).replace(",", "").replace(".", "").isdigit()
                    else np.nan
                    for i in range(1, min(239, len(raw_m)))]

    macro_df = pd.DataFrame({"date": nikkei_dates, "nikkei": nikkei_vals})

    # CPI・ドル円もあれば読み込む（列2, 3を試みる）
    try:
        cpi_vals = [float(str(raw_m.iloc[i, 2]).replace(",", ""))
                    if len(raw_m.columns) > 2 else np.nan
                    for i in range(1, min(239, len(raw_m)))]
        macro_df["cpi"] = cpi_vals
    except:
        macro_df["cpi"] = np.nan

    try:
        usd_vals = [float(str(raw_m.iloc[i, 3]).replace(",", ""))
                    if len(raw_m.columns) > 3 else np.nan
                    for i in range(1, min(239, len(raw_m)))]
        macro_df["usd_jpy"] = usd_vals
    except:
        macro_df["usd_jpy"] = np.nan

    macro_df = macro_df.dropna(subset=["date"])
    macro_pre  = macro_df[macro_df["date"] < "2023-11-01"]
    macro_post = macro_df[macro_df["date"] >= "2023-12-01"]

    macro_stats = []
    for var, label in [("nikkei", "Nikkei 225"),
                       ("cpi",    "CPI (base 2020=100)"),
                       ("usd_jpy", "USD/JPY exchange rate")]:
        if var not in macro_df.columns:
            continue
        s_pre  = macro_pre[var].dropna()
        s_post = macro_post[var].dropna()
        if len(s_pre) == 0:
            continue
        macro_stats.append({
            "variable":   label,
            "pre_N":      len(s_pre),
            "pre_mean":   round(s_pre.mean(), 2),
            "pre_sd":     round(s_pre.std(),  2),
            "pre_median": round(s_pre.median(), 2),
            "pre_min":    round(s_pre.min(), 2),
            "pre_max":    round(s_pre.max(), 2),
            "post_N":     len(s_post),
            "post_mean":  round(s_post.mean(), 2),
            "post_sd":    round(s_post.std(), 2),
            "post_median": round(s_post.median(), 2),
        })
        print(f"\n  [{label}]")
        print(f"    処置前: N={len(s_pre)}  mean={s_pre.mean():.2f}  "
              f"sd={s_pre.std():.2f}  median={s_pre.median():.2f}  "
              f"min={s_pre.min():.2f}  max={s_pre.max():.2f}")
        print(f"    処置後: N={len(s_post)}  mean={s_post.mean():.2f}  "
              f"sd={s_post.std():.2f}  median={s_post.median():.2f}")

    pd.DataFrame(macro_stats).to_csv(OUT_MACRO, index=False, encoding="utf-8-sig")
    print(f"\n  → {OUT_MACRO} を保存しました")

except Exception as e:
    print(f"  [WARNING] マクロ変数の読み込みに失敗: {e}")
    print("  マクロ記述統計をスキップします")

# ─────────────────────────────────────────────────────────────────────────────
# 9. 欠損値・バランス検定
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}\n欠損値・バランス検定\n{SEP}")

print("\n  処置前サンプルの欠損値（局×月セルの欠損）:")
for var in TREATMENT_VARS + CONTROL_VARS:
    if var not in df.columns:
        continue
    s = df.loc[PRE, var]
    n_miss = s.isna().sum()
    pct    = 100 * n_miss / len(s)
    print(f"    {VAR_LABELS.get(var, var)[:40]:40s}: {n_miss:5d} ({pct:.1f}%)")

# バランス：処置前に各局が何ヶ月存在するか
balance = df.loc[PRE].groupby("bureau_code")["date"].nunique()
print(f"\n  処置前期間 局別月数 (max={n_pre}):")
print(f"    min={balance.min()}  mean={balance.mean():.1f}  "
      f"max={balance.max()}  完全バランス局数={( balance == n_pre).sum()}")

# ─────────────────────────────────────────────────────────────────────────────
# 10. ブロック別・規模別の処置前記述統計（Table 2用）
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}\nブロック別・規模別 処置前記述統計（Table 2用）\n{SEP}")

print("\n  [ブロック別 mortgage 処置前月次平均±SD]")
for blk in ["Hokkaido","Tohoku","Kanto","Chubu","Kinki","Chugoku","Shikoku","Kyushu"]:
    sub = df.loc[(df["block"] == blk) & PRE, "mortgage"]
    if sub.empty:
        continue
    print(f"    {blk:12s}: n_obs={len(sub):5,}  mean={sub.mean():8.1f}  "
          f"sd={sub.std():8.1f}  median={sub.median():8.1f}  "
          f"min={sub.min():6.0f}  max={sub.max():8.0f}")

print("\n  [規模別 mortgage 処置前月次平均±SD]")
for sz in ["large", "medium", "small"]:
    sub = df.loc[(df["size_cat"] == sz) & PRE, "mortgage"]
    if sub.empty:
        continue
    n_bur = df.loc[df["size_cat"] == sz, "bureau_code"].nunique()
    print(f"    {sz:8s} ({n_bur}局): n_obs={len(sub):5,}  mean={sub.mean():8.1f}  "
          f"sd={sub.std():8.1f}  median={sub.median():8.1f}  "
          f"min={sub.min():6.0f}  max={sub.max():8.0f}")

# ─────────────────────────────────────────────────────────────────────────────
# 11. 処置群・対照群の時系列トレンド確認（論文テキスト用数値）
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}\n時系列トレンド確認（論文テキスト用）\n{SEP}")

# 年次平均で確認
df["year"] = df["date"].dt.year
for var in ["mortgage", "root_mortgage", "sale", "inheritance_combined"]:
    if var not in df.columns:
        continue
    ann = (df[df["transition"] == 0]
           .groupby("year")[var]
           .mean()
           .round(1))
    print(f"\n  {VAR_LABELS.get(var, var)} 年次平均（局別月平均）:")
    for yr in [2007, 2010, 2015, 2019, 2020, 2021, 2022, 2023, 2024, 2025]:
        if yr in ann.index:
            print(f"    {yr}: {ann[yr]:8.1f}")

# 長期ドリフト確認（処置群/対照群 ln比の変化）
print("\n\n  処置群(mortgage)/対照群(sale) ln比の推移:")
df["ln_mort"] = np.log(df["mortgage"].replace(0, np.nan))
df["ln_sale"] = np.log(df["sale"].replace(0, np.nan))
df["ln_ratio"] = df["ln_mort"] - df["ln_sale"]
ann_ratio = (df[df["transition"] == 0]
             .groupby("year")["ln_ratio"]
             .mean()
             .round(4))
for yr in [2007, 2010, 2015, 2019, 2021, 2022, 2023, 2024, 2025]:
    if yr in ann_ratio.index:
        print(f"    {yr}: {ann_ratio[yr]:+.4f}")

drift_07_23 = ann_ratio.get(2023, np.nan) - ann_ratio.get(2007, np.nan)
print(f"\n  2007→2023 ドリフト: {drift_07_23:+.4f} log points")

# ─────────────────────────────────────────────────────────────────────────────
# 12. 論文Table 1の完成版（全数値整理）
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}\n★ 論文 Table 1 完成版数値 ★\n{SEP}")

print("""
【Panel A: Treatment group (コア2系列, 2007-01 ～ 2023-10)】
""")
for var in TREATMENT_VARS:
    if var not in df.columns:
        continue
    s = df.loc[PRE, var].dropna()
    ln_s = np.log(s.replace(0, np.nan)).dropna()
    print(f"  {VAR_LABELS[var]}")
    print(f"    N(obs)   = {len(s):,}")
    print(f"    N(bureau)= {df.loc[PRE, 'bureau_code'].nunique()}")
    print(f"    Mean     = {s.mean():.1f}")
    print(f"    SD       = {s.std():.1f}")
    print(f"    P10      = {s.quantile(0.10):.1f}")
    print(f"    P25      = {s.quantile(0.25):.1f}")
    print(f"    Median   = {s.median():.1f}")
    print(f"    P75      = {s.quantile(0.75):.1f}")
    print(f"    P90      = {s.quantile(0.90):.1f}")
    print(f"    ln(Mean) = {ln_s.mean():.4f}  ln(SD)={ln_s.std():.4f}")

print("""
【Panel B: Control group (コア2系列, 2007-01 ～ 2023-10)】
""")
for var in CONTROL_VARS:
    if var not in df.columns:
        continue
    s = df.loc[PRE, var].dropna()
    ln_s = np.log(s.replace(0, np.nan)).dropna()
    print(f"  {VAR_LABELS[var]}")
    print(f"    N(obs)   = {len(s):,}")
    print(f"    Mean     = {s.mean():.1f}")
    print(f"    SD       = {s.std():.1f}")
    print(f"    P10      = {s.quantile(0.10):.1f}")
    print(f"    P25      = {s.quantile(0.25):.1f}")
    print(f"    Median   = {s.median():.1f}")
    print(f"    P75      = {s.quantile(0.75):.1f}")
    print(f"    P90      = {s.quantile(0.90):.1f}")
    print(f"    ln(Mean) = {ln_s.mean():.4f}  ln(SD)={ln_s.std():.4f}")

print("""
【Panel C: Post-treatment period (2023-12 ～ 2025-12)】
""")
for var in TREATMENT_VARS + CONTROL_VARS:
    if var not in df.columns:
        continue
    s = df.loc[POST, var].dropna()
    print(f"  {VAR_LABELS[var]}: N={len(s):,}  mean={s.mean():.1f}  "
          f"sd={s.std():.1f}  median={s.median():.1f}")

print(f"""
【パネル全体サマリー（論文本文用）】
  コア4系列（処置前+処置後）:
    総obs     = N/A (ロング形式で計算: {len(df_long):,})
    処置前obs  = {(df_long['post']==0).sum():,}
    処置後obs  = {(df_long['post']==1).sum():,}
""")

print(f"\n{SEP}\n完了\n出力ファイル:\n"
      f"  {OUT_STATS}\n  {OUT_MACRO}\n  {OUT_MONTHLY}\n  {OUT_PREPOST}\n{SEP}")
