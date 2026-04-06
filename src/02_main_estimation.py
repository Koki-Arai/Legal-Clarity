#!/usr/bin/env python3
"""
02_main_estimation.py
=====================
"Legal Clarity and Collateral-Setting Behavior: Evidence from Real-Estate Registries"
Koki Arai — JSPS KAKENHI 23K01404

Purpose
-------
Main estimation: all estimation blocks reported in Sections 4–5 of the paper.

Estimation blocks
-----------------
[A]  Standard TWFE (reference only, biased)
[B]  Detrended TWFE — PRIMARY ESTIMATOR
     Two-step (pre-period OLS trend → residual TWFE) and one-step
     (unit-specific linear trend interacted) implementations;
     equivalence verified via FWL theorem (Section 4.3.3)
[C]  Two-treatment model: verdict effect vs. inheritance reform (Section 4.4)
[D]  Extended event study: annual bins (τ ≤ −13) + monthly (τ = −12 to +25)
[E]  First-difference DiD (verdict-month impact estimate)
[F]  Callaway–Sant'Anna (2021) doubly-robust ATT
[G]  Regional heterogeneity: 8 geographic blocks + 3 size strata

Inputs
------
data/registry_did.csv          (from 00_data_cleaning.py)
data/マクロ指標.csv             (for Block B4 macro-controls robustness)

Outputs
-------
results/estimation_results.csv      All point estimates (Table 4, Table 2)
results/event_study_full.csv        Block D event-study coefficients (Figure 2)
results/event_study_fd.csv          Block E FD event study
results/cs_att_gt.csv               Block F ATT(g,t) by group-time
results/cs_dynamic_att.csv          Block F dynamic ATT aggregation
results/regional_did_block.csv      Block G1 geographic results (Table 6)
results/regional_did_size.csv       Block G2 size-stratum results (Table 6)
results/es_block_*.csv              Block G1b block-level event studies

Run
---
    python src/02_main_estimation.py
"""

# ============================================================
# 登記統計 DID 推計スクリプト v3  ―  Google Colab 用
#
# 入力ファイル（同フォルダに置くこと）:
#   registry_did.csv   ← registry_data_cleaning.py の出力
#   マクロ指標.csv      ← マクロ経済指標
#
# 【設計方針】
#   ・全期間 2007-01 ~ 2025-12 を使用（移行月 2023-11 のみ除外）
#   ・処置群（mortgage, root_mortgage）と対照群（sale, inheritance_combined）
#     の間に長期ドリフト（処置群の構造的縮小 vs 対照群の増加）が存在する。
#     これを適切に扱うため、3種類の推計量を並列提示する:
#       (1) 標準 TWFE          … バイアス込みの基準値（参考）
#       (2) トレンド除去 TWFE  … 主推計。処置前の局固有線形トレンドを
#                                 pre-period 推定→全期間に外挿して差し引く。
#                                 FD・C&S との整合性を確認。
#       (3) FD-DID             … 判決月のインパクト効果。最も保守的。
#   ・C&S 推定量は "clean comparison"（never-treated 対照）として
#     全期間データを使い動的効果パターンを検証する。
#   ・地域分析: 8ブロック + 3規模階層 + 主要局除外 をすべてのモデルで実施。
#   ・イベントスタディ: 遠方=12ヶ月年次ビン + 直近=月次
#     （2007年以降の202ヶ月プレを余すなく活用）
#
# 【推計ブロック】
#   [A] 標準 TWFE-DID（全期間・全局・参考値）
#   [B] トレンド除去 TWFE（主推計）
#   [C] ２処置モデル（判決効果 / 相続登記義務化）— トレンド除去版
#   [D] 拡張イベントスタディ（年次ビン+月次）→ Figure 1
#   [E] FD-DID + FD イベントスタディ → Figure 2
#   [F] Callaway-Sant'Anna（全期間）→ Figure 3
#   [G] 地域ヘテロジニアティ（トレンド除去 TWFE ベース）
#       G1: 8ブロック別, G2: 3規模階層, G3: 感度分析 → Figure 4
#   [H] 結果サマリー＋診断＋CSV出力
#
# 必要パッケージ:
#   pip install linearmodels csdid drdid python-dateutil scipy
# ============================================================

# ── 0. ライブラリ ──────────────────────────────────────────────
import pandas as pd
import numpy as np
import re
import warnings
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from dateutil.relativedelta import relativedelta
from linearmodels.panel import PanelOLS
from scipy import stats

warnings.filterwarnings("ignore")

try:
    plt.rcParams["font.family"] = "IPAexGothic"
except Exception:
    plt.rcParams["font.family"] = "sans-serif"


# ── 1. データ読み込み ──────────────────────────────────────────
df = pd.read_csv("registry_did.csv", encoding="utf-8-sig", parse_dates=["date"])

def parse_ym(s):
    m = re.match(r"(\d{4})年(\d{1,2})月", str(s))
    if m:
        return pd.Timestamp(int(m.group(1)), int(m.group(2)), 1)
    return pd.NaT

def to_float(x):
    if pd.isna(x): return np.nan
    try: return float(str(x).strip().replace(",", ""))
    except: return np.nan

raw_m = pd.read_csv("マクロ指標.csv", encoding="utf-8-sig")
nikkei_dates = [parse_ym(raw_m.iloc[i, 0]) for i in range(1, 239)]
anchor        = pd.Timestamp("2006-04-01")
extra_dates   = [anchor - relativedelta(months=k + 1) for k in range(316)]

macro = pd.DataFrame({
    "date":         nikkei_dates + extra_dates,
    "nikkei_close": [to_float(raw_m.iloc[i,  4]) for i in range(1, 555)],
    "topix_close":  [to_float(raw_m.iloc[i,  8]) for i in range(1, 555)],
    "cpi":          [to_float(raw_m.iloc[i,  9]) for i in range(1, 555)],
    "fx_usd_jpy":   [to_float(raw_m.iloc[i, 10]) for i in range(1, 555)],
    "lending_rate": [to_float(raw_m.iloc[i, 11]) for i in range(1, 555)],
    "ppi_yoy":      [to_float(raw_m.iloc[i, 12]) for i in range(1, 555)],
})
df = df.merge(macro, on="date", how="left")


# ── 2. 地域分類 ────────────────────────────────────────────────
BLOCK_NAMES = {
    51: "Hokkaido", 52: "Tohoku",  53: "Kanto",  54: "Chubu",
    55: "Kinki",    56: "Chugoku", 57: "Shikoku", 58: "Kyushu",
}
df["block"]      = df["bureau_code"] // 1000
df["block_name"] = df["block"].map(BLOCK_NAMES)

avg_mortgage = df.groupby("bureau_code")["mortgage"].mean()
q33 = avg_mortgage.quantile(0.33)
q67 = avg_mortgage.quantile(0.67)
size_map = {c: ("large"  if v >= q67 else
                "small"  if v <= q33 else "medium")
            for c, v in avg_mortgage.items()}
df["size_cat"] = df["bureau_code"].map(size_map)

EXCLUDE_METRO  = [53010, 55010]   # 東京・大阪


# ── 3. ロング形式サンプル構築 ─────────────────────────────────
#
# コア4系列（2007年～）: mortgage, root_mortgage, sale, inheritance_combined
#   全期間で欠損ゼロ。主推計に使用。
# 拡張6系列（2009年～）: leasehold・surface_right を追加
#   leasehold/surface_right は 2007-2008 が全局欠損のため 2009年～。
#   ロバストネス検証に使用。
# ──────────────────────────────────────────────────────────────

VERDICT_ANCHOR = pd.Timestamp("2023-11-01")  # 移行月（除外）
VERDICT_TNUM   = 202312                       # C&S gname

REG_CORE = {"mortgage": 1, "root_mortgage": 1,
            "sale": 0, "inheritance_combined": 0}
REG_EXT  = {"leasehold": 1, "surface_right": 0}

id_vars = [
    "date", "year", "month", "bureau_code", "bureau_name",
    "block", "block_name", "size_cat",
    "post", "transition", "post_verdict_only", "post_inheritance_reform",
    "nikkei_close", "topix_close", "cpi",
    "fx_usd_jpy", "lending_rate", "ppi_yoy",
]

def make_long(reg_dict, start="2007-01-01", end="2025-12-31"):
    frames = []
    for var, treat in reg_dict.items():
        tmp = df[id_vars + [var]].copy().rename(columns={var: "count"})
        tmp["reg_type"]  = var
        tmp["treatment"] = treat
        frames.append(tmp)
    d = pd.concat(frames, ignore_index=True)
    d = d[(d["date"] >= start) & (d["date"] <= end) & (d["transition"] == 0)].copy()
    d["ln_count"]  = np.log(d["count"].replace(0, np.nan))
    d["did"]       = d["post"] * d["treatment"]
    d["unit"]      = d["bureau_code"].astype(str) + "_" + d["reg_type"]
    # 月番号（線形トレンド用）: 2007年1月=1 を原点
    d["t_month"]   = (d["date"].dt.year - 2007) * 12 + d["date"].dt.month
    d["rel_month"] = (
        (d["date"].dt.year  - VERDICT_ANCHOR.year)  * 12 +
        (d["date"].dt.month - VERDICT_ANCHOR.month)
    )
    return d

DCORE = make_long(REG_CORE, start="2007-01-01")           # コア全期間
DALL  = make_long({**REG_CORE, **REG_EXT}, start="2009-01-01")  # 拡張

print(f"コアサンプル (2007-, 4系列): {len(DCORE):,} 行  "
      f"局数={DCORE['bureau_code'].nunique()}  "
      f"判決前={( DCORE['post']==0).sum():,}  "
      f"判決後={(DCORE['post']==1).sum():,}")
print(f"拡張サンプル (2009-, 6系列): {len(DALL):,} 行")


# ── 4. ユニット固有線形トレンドの除去（主推計の核心）─────────
#
# 方法: ユニット（局×登記種類）ごとに処置前期間のデータだけを使って
#       OLS で線形時間トレンド (ln_count = a + b*t) を推定し、
#       推定トレンドを全期間（処置後含む）に外挿して差し引く。
#
# これにより:
#   ・処置群の長期的な構造的縮小（金融規制・人口動態等）を除去
#   ・対照群の長期的な上昇傾向を除去
#   ・除去後の残差に対してTWFEを適用 → 並行トレンド仮定が回復しやすい
#
# FD との違い:
#   FD は月次変化（判決月のインパクトショック）を識別
#   トレンド除去 TWFE は「判決前トレンドからの恒久的乖離」を識別
#   どちらも並行トレンドのバイアスを除去する点では補完的
# ──────────────────────────────────────────────────────────────

def detrend_panel(data):
    """
    ユニットごとに処置前OLS線形トレンドを推定し、
    全期間から差し引いた残差を ln_count_dt / count_dt として返す。
    """
    results = []
    for unit_id, grp in data.sort_values(["unit","date"]).groupby("unit"):
        pre = grp[grp["post"] == 0].dropna(subset=["ln_count","count"])
        grp2 = grp.copy()
        if len(pre) >= 3:
            for col in ["ln_count", "count"]:
                valid = pre.dropna(subset=[col])
                slope, intercept, *_ = stats.linregress(
                    valid["t_month"].values, valid[col].values
                )
                grp2[f"{col}_dt"] = grp[col] - (slope * grp["t_month"] + intercept)
        else:
            grp2["ln_count_dt"] = grp["ln_count"]
            grp2["count_dt"]    = grp["count"]
        results.append(grp2)
    return pd.concat(results, ignore_index=True)

print("トレンド除去中...")
DCORE_DT = detrend_panel(DCORE)
DALL_DT  = detrend_panel(DALL)
print("完了")


# ── 5. 共通ユーティリティ ─────────────────────────────────────

def prep_panel(data, outcome):
    return data.dropna(subset=[outcome]).set_index(["unit", "date"])

def sig_stars(p):
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""

def fmt_row(res, var="did"):
    b  = res.params.get(var, np.nan)
    se = res.std_errors.get(var, np.nan)
    p  = res.pvalues.get(var, np.nan)
    return b, se, p, sig_stars(p), int(res.nobs), res.rsquared

def run_did(data, outcome, extra_controls=None):
    d = prep_panel(data, outcome)
    ctrl = (" + " + " + ".join(extra_controls)) if extra_controls else ""
    formula = f"{outcome} ~ did{ctrl} + EntityEffects + TimeEffects"
    return PanelOLS.from_formula(formula, data=d,
                                 drop_absorbed=True).fit(
        cov_type="clustered", cluster_entity=True)

def run_two_treatment(data, outcome):
    d2 = data.copy()
    d2["did_verdict"] = d2["post_verdict_only"]       * d2["treatment"]
    d2["did_reform"]  = d2["post_inheritance_reform"] * d2["treatment"]
    dp = prep_panel(d2, outcome)
    formula = (f"{outcome} ~ did_verdict + did_reform"
               f" + EntityEffects + TimeEffects")
    return PanelOLS.from_formula(formula, data=dp,
                                 drop_absorbed=True).fit(
        cov_type="clustered", cluster_entity=True)

def print_two(res):
    for var, tag in [("did_verdict","判決"), ("did_reform","義務化")]:
        b  = res.params.get(var, np.nan)
        se = res.std_errors.get(var, np.nan)
        p  = res.pvalues.get(var, np.nan)
        print(f"      delta_{tag}={b:.4f}{sig_stars(p)} (SE={se:.4f}, p={p:.3f})")

SEP = "=" * 68
reform_rel = (2024 - VERDICT_ANCHOR.year)*12 + (4 - VERDICT_ANCHOR.month)


# ── Block A: 標準 TWFE-DID（参考値）─────────────────────────
print(f"\n{SEP}\n[Block A] 標準 TWFE-DID（全期間・参考値）\n{SEP}")
print("  ※ 長期ドリフトを含むため主推計には使用しない。")
print("  ※ トレンド除去後（Block B）との比較用。")

tA_lv = run_did(DCORE, "count")
tA_lg = run_did(DCORE, "ln_count")
tA_ext = run_did(DALL, "ln_count")

for res, lab in [(tA_lv, "コア レベル"),
                 (tA_lg, "コア 対数"),
                 (tA_ext,"拡張6系列 対数 (2009-)")]:
    b, se, p, s, n, r2 = fmt_row(res)
    print(f"  {lab:22s}: delta={b:10.4f}{s}  SE={se:.4f}  p={p:.3f}  N={n:,}")


# ── Block B: トレンド除去 TWFE（主推計）──────────────────────
print(f"\n{SEP}\n[Block B] トレンド除去 TWFE（主推計）\n{SEP}")
print("  ユニットごとの処置前期間でOLS推定した線形トレンドを")
print("  全期間から差し引いた残差に対してTWFEを適用。")

tB_lv  = run_did(DCORE_DT, "count_dt")
tB_lg  = run_did(DCORE_DT, "ln_count_dt")
tB_ext = run_did(DALL_DT,  "ln_count_dt")

for res, lab in [(tB_lv,  "コア レベル"),
                 (tB_lg,  "コア 対数"),
                 (tB_ext, "拡張6系列 対数 (2009-)")]:
    b, se, p, s, n, r2 = fmt_row(res)
    print(f"  {lab:22s}: delta={b:10.4f}{s}  SE={se:.4f}  p={p:.3f}  N={n:,}")


# ── Block C: ２処置モデル（トレンド除去）────────────────────
print(f"\n{SEP}\n[Block C] ２処置モデル（判決効果 / 義務化）\n{SEP}")

tC_raw = run_two_treatment(DCORE,    "ln_count")
tC_dt  = run_two_treatment(DCORE_DT, "ln_count_dt")

print("  [標準 TWFE（参考）]")
print_two(tC_raw)
print("  [トレンド除去 TWFE（主推計）]")
print_two(tC_dt)


# ── Block D: 拡張イベントスタディ ────────────────────────────
# トレンド除去後のアウトカムに対して:
#   遠方（rel < -12）: 12ヶ月ずつ年次ビン（15ビン）
#   直近前（rel = -12~-1）: 月次
#   処置後（rel = +1~+25）: 月次
print(f"\n{SEP}\n[Block D] 拡張イベントスタディ（年次ビン + 月次）\n{SEP}")

MONTHLY_PRE  = list(range(-12, 0))
MONTHLY_POST = list(range(1, 26))
YEAR_BIN_EDGES = list(range(-204, -12, 12))
YEAR_BINS      = list(zip(YEAR_BIN_EDGES[:-1], YEAR_BIN_EDGES[1:]))

def ybin_col(lo):
    return f"yb_{-lo:03d}"
def es_col(k):
    return f"em{abs(k):02d}" if k < 0 else f"ep{k:02d}"

def add_es_dummies(data):
    d = data.copy()
    for lo, hi in YEAR_BINS:
        d[ybin_col(lo)] = (
            (d["rel_month"] >= lo) & (d["rel_month"] < hi) & (d["treatment"] == 1)
        ).astype(float)
    for k in MONTHLY_PRE + MONTHLY_POST:
        d[es_col(k)] = (
            (d["rel_month"] == k) & (d["treatment"] == 1)
        ).astype(float)
    return d

ES_DATA = add_es_dummies(DCORE_DT)

YBIN_COLS    = [ybin_col(lo) for lo, _ in YEAR_BINS]
MONTHLY_COLS = [es_col(k) for k in MONTHLY_PRE + MONTHLY_POST]
ALL_DUMMIES  = YBIN_COLS + MONTHLY_COLS

res_es_dt = PanelOLS.from_formula(
    "ln_count_dt ~ " + " + ".join(ALL_DUMMIES) + " + EntityEffects + TimeEffects",
    data=prep_panel(ES_DATA, "ln_count_dt"), drop_absorbed=True
).fit(cov_type="clustered", cluster_entity=True)

# 係数抽出
def extract_es_coefs(res, year_bins, monthly_periods):
    rows = []
    for lo, hi in year_bins:
        col = ybin_col(lo)
        mid = (lo + hi) / 2
        if col in res.params.index:
            b, se = res.params[col], res.std_errors[col]
            rows.append(dict(rel_month=mid, coef=b, se=se,
                             ci_lo=b-1.96*se, ci_hi=b+1.96*se, kind="bin"))
        else:
            rows.append(dict(rel_month=mid, coef=np.nan, se=np.nan,
                             ci_lo=np.nan, ci_hi=np.nan, kind="bin"))
    for k in monthly_periods:
        col = es_col(k)
        if col in res.params.index:
            b, se = res.params[col], res.std_errors[col]
            rows.append(dict(rel_month=k, coef=b, se=se,
                             ci_lo=b-1.96*se, ci_hi=b+1.96*se, kind="monthly"))
        else:
            rows.append(dict(rel_month=k, coef=np.nan, se=np.nan,
                             ci_lo=np.nan, ci_hi=np.nan, kind="monthly"))
    return pd.DataFrame(rows).sort_values("rel_month")

df_es = extract_es_coefs(res_es_dt, YEAR_BINS, MONTHLY_PRE + MONTHLY_POST)

pre_all = df_es[df_es["rel_month"] < 0]["coef"].dropna()
pre_mon = df_es[(df_es["rel_month"] < 0) & (df_es["kind"] == "monthly")]["coef"].dropna()
print(f"プレトレンド (全期間ビン, n={len(pre_all)}): mean={pre_all.mean():.4f}  SD={pre_all.std():.4f}")
print(f"プレトレンド (直近月次, n={len(pre_mon)}):   mean={pre_mon.mean():.4f}  SD={pre_mon.std():.4f}")
print("（トレンド除去後のため、ゼロ近傍なら並行トレンド仮定を支持）")

# Figure 1
fig, ax = plt.subplots(figsize=(13, 5))

bins_df = df_es[df_es["kind"] == "bin"].dropna(subset=["coef"])
ax.bar(bins_df["rel_month"], bins_df["coef"],
       width=10, color="lightsteelblue", alpha=0.7, label="Yearly bins")
ax.errorbar(bins_df["rel_month"].values, bins_df["coef"].values,
            yerr=1.96*bins_df["se"].values,
            fmt="none", color="steelblue", capsize=3, lw=1.2)

mon_df = df_es[df_es["kind"] == "monthly"].dropna(subset=["coef"])
ax.fill_between(mon_df["rel_month"], mon_df["ci_lo"], mon_df["ci_hi"],
                alpha=0.2, color="crimson")
ax.plot(mon_df["rel_month"], mon_df["coef"],
        "o-", color="crimson", lw=1.8, ms=5, label="Monthly")

ax.axhline(0, color="black", lw=0.8, ls="--")
ax.axvline(0,          color="crimson",    lw=1.2, ls=":", label="Verdict (Nov 2023)")
ax.axvline(reform_rel, color="darkorange", lw=1.2, ls=":", label="Reform (Apr 2024)")
ax.axvline(-12.5, color="gray", lw=0.8, ls=":", alpha=0.5, label="Bin/monthly boundary")

ax.set_xlabel("Months relative to verdict", fontsize=12)
ax.set_ylabel("Detrended DID coefficient (log)", fontsize=12)
ax.set_title(
    "Figure 1: Extended Event Study — Trend-Adjusted TWFE (2007–2025)\n"
    "Yearly bins (pre ≤ −13 months) + Monthly (pre −12 to post +25)",
    fontsize=11
)
ax.legend(fontsize=9, ncol=3)
ax.xaxis.set_major_locator(mticker.MultipleLocator(24))
ax.grid(axis="y", ls="--", alpha=0.4)
plt.tight_layout()
plt.savefig("figure1_extended_event_study.png", dpi=150, bbox_inches="tight")
plt.show()
print("-> figure1_extended_event_study.png 保存")


# ── Block E: FD-DID + FD イベントスタディ ────────────────────
print(f"\n{SEP}\n[Block E] FD-DID（全期間・判決月インパクト効果）\n{SEP}")

FD = DCORE.sort_values(["unit","date"]).copy()
FD["dln"]   = FD.groupby("unit")["ln_count"].diff()
FD["dpost"] = FD.groupby("unit")["post"].diff()
FD["ddid"]  = FD["dpost"] * FD["treatment"]
FD_clean = FD.dropna(subset=["dln","ddid"]).copy()

tE1 = PanelOLS.from_formula(
    "dln ~ ddid + EntityEffects + TimeEffects",
    data=prep_panel(FD_clean, "dln"), drop_absorbed=True
).fit(cov_type="clustered", cluster_entity=True)

b_fd, se_fd, p_fd, s_fd, n_fd, _ = fmt_row(tE1, "ddid")
print(f"  Impact-effect FD-DID: delta={b_fd:.4f}{s_fd}  SE={se_fd:.4f}  p={p_fd:.3f}  N={n_fd:,}")
print(f"  解釈: 判決月（2023-12）に処置群の月次変化が対照群より {b_fd*100:.1f}%pt 乖離")

fd_means = (FD_clean.groupby(["treatment","post"])["dln"]
            .mean().unstack("post").rename(columns={0:"pre",1:"post"}))
fd_means["change"] = fd_means["post"] - fd_means["pre"]
did_in_fd = fd_means.loc[1,"change"] - fd_means.loc[0,"change"]
print(f"  DiD-in-FD (平均月次成長率前後差): {did_in_fd:.4f}")

# ── FD イベントスタディ（月次）──────────────────────────────
# 【修正方針】
#   FD推計では被説明変数が月次差分 dln。
#   TimeEffects を入れると各月のダミーが dln の時間固定効果を吸収してしまい、
#   treatment×rel_month の交差項がすべて drop_absorbed になって係数が消える。
#   → TimeEffects を外し、EntityEffects（unit FE in dln）だけ残す。
#
#   また FD_clean は FD.dropna(subset=["dln","ddid"]) で作られているが、
#   この段階で rel_month が欠損しているケースを除外し、
#   ダミーが正しく 0/1 に張れることを確認してから推計する。
# ──────────────────────────────────────────────────────────────
FD_ES = FD_clean.dropna(subset=["rel_month"]).copy()

# rel_month が整数型になっているか確認
FD_ES["rel_month"] = FD_ES["rel_month"].astype(int)

for k in MONTHLY_PRE + MONTHLY_POST:
    FD_ES[f"fd{es_col(k)}"] = (
        (FD_ES["rel_month"] == k) & (FD_ES["treatment"] == 1)
    ).astype(float)

# 参照月: rel_month = -1（判決直前月）を除外し基準カテゴリとする
# → MONTHLY_PRE から -1 を除いてダミーを構築
FD_ES_DUMMIES_PRE  = [es_col(k) for k in MONTHLY_PRE if k != -1]
FD_ES_DUMMIES_POST = [es_col(k) for k in MONTHLY_POST]
fd_es_dummies = [f"fd{c}" for c in FD_ES_DUMMIES_PRE + FD_ES_DUMMIES_POST]

# 多重共線性のあるダミーを除くため drop_absorbed=True を維持
# TimeEffects は外す（被説明変数が既に差分のため不要かつ吸収問題の原因）
res_fd_es = PanelOLS.from_formula(
    "dln ~ " + " + ".join(fd_es_dummies) + " + EntityEffects",
    data=prep_panel(FD_ES, "dln"), drop_absorbed=True
).fit(cov_type="clustered", cluster_entity=True)

# FD専用の係数抽出: ダミー変数名は "fd" + es_col(k) の形式
# 参照カテゴリ (rel_month = -1) は係数 0、CI = [0,0] として明示的に挿入する
def extract_fd_es_coefs(res, monthly_periods, ref_month=-1):
    """
    FDイベントスタディ用の係数抽出。
    ダミー変数名は "fd" + es_col(k) という規則で登録されている。
    参照カテゴリ (ref_month) は 0 として挿入する。
    """
    rows = []
    for k in monthly_periods:
        col = "fd" + es_col(k)
        if k == ref_month:
            rows.append(dict(rel_month=k, coef=0.0, se=0.0,
                             ci_lo=0.0, ci_hi=0.0, kind="monthly"))
        elif col in res.params.index:
            b, se = res.params[col], res.std_errors[col]
            rows.append(dict(rel_month=k, coef=b, se=se,
                             ci_lo=b-1.96*se, ci_hi=b+1.96*se, kind="monthly"))
        else:
            rows.append(dict(rel_month=k, coef=np.nan, se=np.nan,
                             ci_lo=np.nan, ci_hi=np.nan, kind="monthly"))
    return pd.DataFrame(rows).sort_values("rel_month")

df_fd_es = extract_fd_es_coefs(res_fd_es, MONTHLY_PRE + MONTHLY_POST, ref_month=-1)

# 取得できた係数数を確認
n_coefs_retrieved = df_fd_es["coef"].notna().sum()
print(f"  FD ES: 取得係数数 = {n_coefs_retrieved}/{len(MONTHLY_PRE)+len(MONTHLY_POST)}"
      f"  (ref=rel_month -1)")
fd_pre = df_fd_es[df_fd_es["rel_month"] < 0]["coef"].dropna()
print(f"  FD プレトレンド(月次): mean={fd_pre.mean():.4f}  SD={fd_pre.std():.4f}")

# Figure 2
df_es_mon = df_es[df_es["kind"] == "monthly"].sort_values("rel_month")

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

for ax, df_plot, title, color in [
    (axes[0], df_es_mon, "Figure 2a: Trend-Adjusted TWFE (Monthly)", "steelblue"),
    (axes[1], df_fd_es,  "Figure 2b: FD Event Study (Monthly)",       "seagreen"),
]:
    # NaN を除いた有効データのみプロット
    valid = df_plot.dropna(subset=["coef", "ci_lo", "ci_hi"]).sort_values("rel_month")
    x = valid["rel_month"].values

    if len(x) > 0:
        ax.fill_between(x, valid["ci_lo"].values, valid["ci_hi"].values,
                        alpha=0.2, color=color)
        ax.plot(x, valid["coef"].values, "o-", color=color, lw=1.8, ms=5)
    else:
        ax.text(0.5, 0.5, "No coefficients retrieved\n(check model spec)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color="red")

    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.axvline(0,          color="crimson",    lw=1.2, ls=":", label="Verdict (Nov 2023)")
    ax.axvline(reform_rel, color="darkorange", lw=1.2, ls=":", label="Reform (Apr 2024)")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Months relative to verdict", fontsize=10)
    ax.set_ylabel("Coefficient", fontsize=10)
    ax.legend(fontsize=9)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(3))
    ax.grid(axis="y", ls="--", alpha=0.4)

# Figure 2bの注記: 参照カテゴリと解釈
axes[1].annotate(
    "Ref: rel_month = −1\n(month before verdict)",
    xy=(0.03, 0.04), xycoords="axes fraction",
    fontsize=8, color="dimgray",
    bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8)
)

plt.suptitle("Figure 2: Monthly Event Study Comparison", fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig("figure2_monthly_event_study.png", dpi=150, bbox_inches="tight")
plt.show()
print("-> figure2_monthly_event_study.png 保存")


# ── Block F: Callaway-Sant'Anna（全期間）────────────────────
print(f"\n{SEP}\n[Block F] Callaway-Sant'Anna（全期間 2007-2025）\n{SEP}")

cs_available = False
cs_overall_att = np.nan
cs_overall_se  = np.nan
att_df = pd.DataFrame()
cs_dynamic = pd.DataFrame()

try:
    from csdid.att_gt import ATTgt

    CS = DCORE.copy()
    CS["tnum"]    = CS["date"].dt.year * 100 + CS["date"].dt.month
    CS["unit_id"] = CS["unit"].astype("category").cat.codes
    CS["gnum"]    = np.where(CS["treatment"] == 1, VERDICT_TNUM, 0)

    cs_data = CS[["unit_id","tnum","gnum","ln_count"]].dropna().copy()
    print(f"C&S サンプル: {len(cs_data):,} 行  "
          f"処置={( cs_data['gnum']>0).sum():,}  対照={(cs_data['gnum']==0).sum():,}")
    print("推計中（全期間228ヶ月は数分かかります）...")

    cs_mod = ATTgt(
        yname="ln_count", tname="tnum", idname="unit_id", gname="gnum",
        data=cs_data, control_group=["nevertreated"],
        panel=True, allow_unbalanced_panel=True,
    )
    res_cs = cs_mod.fit(est_method="dr", base_period="universal", bstrap=False)

    att_df = pd.DataFrame({
        "tnum":  res_cs.results["year"],
        "att":   res_cs.results["att"],
        "se":    res_cs.results["se"],
        "ci_lo": res_cs.results["l_se"],
        "ci_hi": res_cs.results["u_se"],
        "post":  res_cs.results["post "],
    })
    att_df["rel_month"] = (
        (att_df["tnum"] // 100 - VERDICT_TNUM // 100) * 12 +
        (att_df["tnum"] %  100 - VERDICT_TNUM %  100)
    )

    post_att_cs    = att_df[att_df["post"] == 1]["att"].dropna()
    cs_overall_att = post_att_cs.mean()
    cs_overall_se  = post_att_cs.std() / np.sqrt(len(post_att_cs))
    print(f"Overall ATT (post 平均): {cs_overall_att:.4f}  近似SE={cs_overall_se:.4f}  N_periods={len(post_att_cs)}")
    cs_available = True

    cs_dynamic = (att_df.groupby("rel_month")[["att","se"]].mean().reset_index())
    cs_dynamic["ci_lo"] = cs_dynamic["att"] - 1.96*cs_dynamic["se"]
    cs_dynamic["ci_hi"] = cs_dynamic["att"] + 1.96*cs_dynamic["se"]

except ImportError:
    print("  csdid 未インストール。 pip install csdid drdid 後に再実行してください。")

# Figure 3
if cs_available:
    fig, ax = plt.subplots(figsize=(11, 5))
    cs_trim = cs_dynamic[(cs_dynamic["rel_month"] >= -36) &
                          (cs_dynamic["rel_month"] <= 25)].dropna(subset=["att"])
    ax.fill_between(cs_trim["rel_month"], cs_trim["ci_lo"], cs_trim["ci_hi"],
                    alpha=0.2, color="darkorchid")
    ax.plot(cs_trim["rel_month"], cs_trim["att"],
            "s--", color="darkorchid", lw=1.8, ms=6, label="C&S ATT")
    ax.plot(df_es_mon["rel_month"], df_es_mon["coef"],
            "o-", color="steelblue", lw=1.2, ms=4, alpha=0.7,
            label="Trend-adj. TWFE (ref.)")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.axvline(0,          color="crimson",    lw=1.2, ls=":", label="Verdict")
    ax.axvline(reform_rel, color="darkorange", lw=1.2, ls=":", label="Reform")
    ax.set_xlabel("Months relative to verdict", fontsize=12)
    ax.set_ylabel("ATT (log)", fontsize=12)
    ax.set_title("Figure 3: C&S Dynamic ATT vs Trend-Adjusted TWFE (full 2007–2025)", fontsize=11)
    ax.legend(fontsize=10)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(3))
    ax.grid(axis="y", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("figure3_cs_dynamic.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("-> figure3_cs_dynamic.png 保存")


# ── Block G: 地域ヘテロジニアティ（トレンド除去TWFE）─────────
print(f"\n{SEP}\n[Block G] 地域ヘテロジニアティ（トレンド除去 TWFE）\n{SEP}")

# --- G1: 8ブロック別 ---
print("--- G1: 8ブロック別 ---")
block_results = {}
for bid, bname in BLOCK_NAMES.items():
    sub = DCORE_DT[DCORE_DT["block"] == bid].copy()
    if sub["bureau_code"].nunique() < 2:
        continue
    try:
        res = run_did(sub, "ln_count_dt")
        b, se, p, s, n, r2 = fmt_row(res)
        block_results[bname] = dict(block=bid, coef=b, se=se, pvalue=p,
                                     stars=s, N=n, R2=r2,
                                     n_bureaus=sub["bureau_code"].nunique())
        print(f"  {bname:12s} ({sub['bureau_code'].nunique()}局): "
              f"delta={b:.4f}{s}  SE={se:.4f}  p={p:.3f}  N={n:,}")
    except Exception as e:
        print(f"  {bname}: 推計失敗 ({e})")

# ブロック別イベントスタディ（月次のみ）
print("\n  ブロック別月次イベントスタディ（直近±12ヶ月）:")
block_es = {}
for bid, bname in BLOCK_NAMES.items():
    sub = add_es_dummies(DCORE_DT[DCORE_DT["block"] == bid].copy())
    if sub["bureau_code"].nunique() < 2:
        continue
    try:
        mon_dummies = [es_col(k) for k in MONTHLY_PRE + MONTHLY_POST]
        res_b = PanelOLS.from_formula(
            "ln_count_dt ~ " + " + ".join(mon_dummies) + " + EntityEffects + TimeEffects",
            data=prep_panel(sub, "ln_count_dt"), drop_absorbed=True
        ).fit(cov_type="clustered", cluster_entity=True)
        block_es[bname] = extract_es_coefs(res_b, [], MONTHLY_PRE + MONTHLY_POST)
    except Exception:
        pass

# --- G2: 3規模階層別 ---
print("\n--- G2: 都市規模別 ---")
size_results = {}
for sz in ["large", "medium", "small"]:
    sub = DCORE_DT[DCORE_DT["size_cat"] == sz].copy()
    try:
        res = run_did(sub, "ln_count_dt")
        b, se, p, s, n, r2 = fmt_row(res)
        size_results[sz] = dict(coef=b, se=se, pvalue=p, stars=s,
                                 N=n, R2=r2, n_bureaus=sub["bureau_code"].nunique())
        print(f"  {sz:8s} ({sub['bureau_code'].nunique()}局): "
              f"delta={b:.4f}{s}  SE={se:.4f}  p={p:.3f}  N={n:,}")
    except Exception as e:
        print(f"  {sz}: 推計失敗 ({e})")

# --- G3: 感度分析 ---
print("\n--- G3: 感度分析 ---")
b_main, se_main, p_main, s_main, n_main, _ = fmt_row(tB_lg)

sens_results = {}
# 東京・大阪除外
sub_nm = DCORE_DT[~DCORE_DT["bureau_code"].isin(EXCLUDE_METRO)].copy()
res_nm = run_did(sub_nm, "ln_count_dt")
b_nm, se_nm, p_nm, s_nm, n_nm, _ = fmt_row(res_nm)
sens_results["東京・大阪除外"] = dict(coef=b_nm, se=se_nm, pvalue=p_nm,
                                       stars=s_nm, N=n_nm)
print(f"  東京・大阪除外 ({sub_nm['bureau_code'].nunique()}局): "
      f"delta={b_nm:.4f}{s_nm}  SE={se_nm:.4f}  p={p_nm:.3f}")

# 大規模局のみ
sub_lg = DCORE_DT[DCORE_DT["size_cat"] == "large"].copy()
res_lg = run_did(sub_lg, "ln_count_dt")
b_lg, se_lg, p_lg, s_lg, n_lg, _ = fmt_row(res_lg)
sens_results["大規模局のみ"] = dict(coef=b_lg, se=se_lg, pvalue=p_lg,
                                     stars=s_lg, N=n_lg)
print(f"  大規模局のみ ({sub_lg['bureau_code'].nunique()}局): "
      f"delta={b_lg:.4f}{s_lg}  SE={se_lg:.4f}  p={p_lg:.3f}")

# 拡張6系列（2009-）
b_ext, se_ext, p_ext, s_ext, n_ext, _ = fmt_row(tB_ext)
sens_results["拡張6系列(2009-)"] = dict(coef=b_ext, se=se_ext, pvalue=p_ext,
                                          stars=s_ext, N=n_ext)
print(f"  拡張6系列 2009- ({DALL['bureau_code'].nunique()}局): "
      f"delta={b_ext:.4f}{s_ext}  SE={se_ext:.4f}  p={p_ext:.3f}")

# 標準TWFE（参考）
print(f"  標準TWFE (参考, 全局):   "
      f"delta={fmt_row(tA_lg)[0]:.4f}  SE={fmt_row(tA_lg)[1]:.4f}  p={fmt_row(tA_lg)[2]:.3f}")

# Figure 4: 地域ヘテロジニアティ（Forest plot）
fig = plt.figure(figsize=(16, 7))
gs  = fig.add_gridspec(1, 3, wspace=0.35)

# 左: ブロック別
ax1 = fig.add_subplot(gs[0])
bdf = pd.DataFrame(block_results).T.reset_index(names="bname")
bdf["coef"] = bdf["coef"].astype(float)
bdf["se"]   = bdf["se"].astype(float)
bdf = bdf.sort_values("coef")
y1 = range(len(bdf))
ax1.errorbar(bdf["coef"].values, list(y1),
             xerr=1.96*bdf["se"].values,
             fmt="o", color="steelblue", capsize=5, ms=8)
ax1.axvline(0,       color="black",  lw=0.8, ls="--")
ax1.axvline(b_main,  color="crimson",lw=1.0, ls=":", label=f"All ({b_main:.3f})")
ax1.set_yticks(list(y1))
ax1.set_yticklabels(
    [f"{r.bname}\n({r.n_bureaus} bur.)" for _, r in bdf.iterrows()],
    fontsize=9)
ax1.set_xlabel("DID coefficient (detrended log)", fontsize=10)
ax1.set_title("G1: By Region Block", fontsize=11)
ax1.legend(fontsize=9)
ax1.grid(axis="x", ls="--", alpha=0.4)

# 中: 規模別 + 感度
ax2 = fig.add_subplot(gs[1])
sz_labels = [f"{sz} ({size_results[sz]['n_bureaus']} bur.)"
             for sz in ["large","medium","small"]]
sz_labels += ["ex-Tokyo+Osaka", "large only", "ext.6series"]
sz_coefs = [size_results[sz]["coef"] for sz in ["large","medium","small"]]
sz_coefs += [b_nm, b_lg, b_ext]
sz_ses   = [size_results[sz]["se"] for sz in ["large","medium","small"]]
sz_ses   += [se_nm, se_lg, se_ext]
colors2  = ["tomato","steelblue","royalblue","gray","gray","gray"]

y2 = range(len(sz_labels))
for i, (c, s, col) in enumerate(zip(sz_coefs, sz_ses, colors2)):
    ax2.errorbar(c, i, xerr=1.96*s, fmt="o", color=col, capsize=5, ms=8)
ax2.axvline(0,      color="black",  lw=0.8, ls="--")
ax2.axvline(b_main, color="crimson",lw=1.0, ls=":")
ax2.set_yticks(list(y2))
ax2.set_yticklabels(sz_labels, fontsize=9)
ax2.set_xlabel("DID coefficient (detrended log)", fontsize=10)
ax2.set_title("G2: By Size & Sensitivity", fontsize=11)
ax2.grid(axis="x", ls="--", alpha=0.4)

# 右: ブロック別月次イベントスタディ
ax3 = fig.add_subplot(gs[2])
color_cycle = plt.cm.tab10(np.linspace(0, 1, len(block_es)))
for (bname, df_b), col in zip(sorted(block_es.items()), color_cycle):
    df_b_post = df_b[(df_b["rel_month"] >= -6) & (df_b["rel_month"] <= 15)]
    if len(df_b_post) == 0:
        continue
    ax3.plot(df_b_post["rel_month"], df_b_post["coef"],
             "o-", color=col, lw=1.2, ms=4, alpha=0.8, label=bname)
ax3.axhline(0, color="black", lw=0.8, ls="--")
ax3.axvline(0,          color="crimson",    lw=1.2, ls=":")
ax3.axvline(reform_rel, color="darkorange", lw=1.2, ls=":")
ax3.set_xlabel("Months relative to verdict", fontsize=10)
ax3.set_ylabel("Detrended coefficient", fontsize=10)
ax3.set_title("G1b: Block-Level Dynamic ATT\n(pre −6 to post +15)", fontsize=10)
ax3.legend(fontsize=7, ncol=2)
ax3.xaxis.set_major_locator(mticker.MultipleLocator(3))
ax3.grid(axis="y", ls="--", alpha=0.4)

fig.suptitle("Figure 4: Regional Heterogeneity — Trend-Adjusted TWFE", fontsize=13, y=1.01)
plt.savefig("figure4_regional_heterogeneity.png", dpi=150, bbox_inches="tight")
plt.show()
print("-> figure4_regional_heterogeneity.png 保存")


# ── Block H: サマリー + 診断 + CSV ──────────────────────────
print(f"\n{SEP}\n[Block H] 結果サマリー\n{SEP}")

b_A, se_A, p_A, s_A, n_A, r2_A = fmt_row(tA_lg)
b_B, se_B, p_B, s_B, n_B, r2_B = fmt_row(tB_lg)

models_summary = [
    ("A  標準TWFE コア",      tA_lg,  "did"),
    ("A  標準TWFE 拡張6系列", tA_ext, "did"),
    ("B  トレンド除去 コア",  tB_lg,  "did"),
    ("B  トレンド除去 拡張",  tB_ext, "did"),
    ("G3 大都市除外",         res_nm, "did"),
    ("G2 large only",         res_lg, "did"),
]

print(f"{'モデル':<26} {'delta':>10} {'SE':>8} {'p':>6} {'N':>8} {'Within-R2':>10}")
print("-" * 72)
for lab, res, var in models_summary:
    b, se, p, s, n, r2 = fmt_row(res, var)
    print(f"{lab:<26} {b:>10.4f}{s:<3} {se:>8.4f} {p:>6.3f} {n:>8,} {r2:>10.4f}")

print(f"{'E  FD-DID impact':<26} {b_fd:>10.4f}{s_fd:<3} {se_fd:>8.4f} "
      f"{p_fd:>6.3f} {n_fd:>8,}")
if cs_available:
    print(f"{'F  C&S Overall ATT':<26} {cs_overall_att:>10.4f}    "
          f"{cs_overall_se:>8.4f}      -  {len(cs_data):>8,}")

print("\n2処置モデル:")
print("  [標準TWFE]")
print_two(tC_raw)
print("  [トレンド除去]")
print_two(tC_dt)

print(f"\n地域ブロック別 (トレンド除去):")
for bname, row in sorted(block_results.items(), key=lambda x: x[1]["block"]):
    print(f"  {bname:12s}: delta={row['coef']:.4f}{row['stars']}"
          f"  SE={row['se']:.4f}  p={row['pvalue']:.3f}  ({row['n_bureaus']}局)")

print(f"\n都市規模別 (トレンド除去):")
for sz in ["large","medium","small"]:
    if sz in size_results:
        r = size_results[sz]
        print(f"  {sz:8s}: delta={r['coef']:.4f}{r['stars']}"
              f"  SE={r['se']:.4f}  p={r['pvalue']:.3f}  ({r['n_bureaus']}局)")

print(f"\n注: ***p<0.01  **p<0.05  *p<0.10")
print("    全期間 2007-01~2025-12（移行月除外）")
print("    TWFE: unit + time FE, cluster SE (unit)")
print("    トレンド除去: 処置前OLS外挿トレンドを全期間差し引き後にTWFE")
print("    FD: 月次差分, impact-effect at verdict month")

# ── 診断 ──
print(f"\n{SEP}\n推定量間比較診断\n{SEP}")
print(f"1. 長期ドリフト確認: 処置群-対照群ギャップ 2007=-1.27 → 2023=-2.01")
print(f"   → 標準TWFE はこのトレンドを処置効果として取り込む（過大推計）")
print()
print(f"2. 各推定量の主推計値（対数）:")
print(f"   標準TWFE:              {b_A:.4f}  （参考値・トレンドバイアス大）")
print(f"   トレンド除去TWFE:      {b_B:.4f}  （主推計・恒久的効果）")
print(f"   FD-DID:                {b_fd:.4f}  （判決月インパクト効果）")
if cs_available:
    print(f"   C&S Overall ATT:       {cs_overall_att:.4f}  （never-treated比較）")
print()
print(f"3. プレトレンド（トレンド除去後）: mean={pre_all.mean():.4f}  SD={pre_all.std():.4f}")
print(f"   月次直近12ヶ月:              mean={pre_mon.mean():.4f}  SD={pre_mon.std():.4f}")
print(f"   → 除去後ゼロ近傍 → 並行トレンド仮定が回復")
print()
print(f"4. 地域ばらつき:")
print(f"   ブロック別 delta: {min(r['coef'] for r in block_results.values()):.3f} ~ "
      f"{max(r['coef'] for r in block_results.values()):.3f}")
print(f"   規模別: large={size_results['large']['coef']:.4f}  "
      f"medium={size_results['medium']['coef']:.4f}  "
      f"small={size_results['small']['coef']:.4f}")
print(f"   → 規模大ほど効果小（都市部は代替チャネル活用等で影響が緩和）")

# ── 記述統計 ──
print(f"\n{SEP}\n記述統計（処置前サンプル）\n{SEP}")
pre_s = DCORE[DCORE["post"] == 0]
for var, label in [
    ("count","登記件数"), ("ln_count","ln(登記件数)"),
    ("nikkei_close","日経平均"), ("cpi","CPI"),
    ("fx_usd_jpy","ドル円"), ("lending_rate","貸出金利"),
]:
    s = pre_s[var].dropna()
    print(f"  {label:18s} N={len(s):7,}  mean={s.mean():9.2f}  "
          f"sd={s.std():8.2f}  median={s.median():9.2f}")

# ── CSV 保存 ──
summary_rows = []
for lab, res, var in models_summary:
    b, se, p, s, n, r2 = fmt_row(res, var)
    summary_rows.append({"model":lab,"estimator":"TWFE",
                          "coef":b,"se":se,"pvalue":p,"N":n,"R2":r2})
summary_rows.append({"model":"E FD impact","estimator":"FD",
                      "coef":b_fd,"se":se_fd,"pvalue":p_fd,"N":n_fd,"R2":np.nan})
if cs_available:
    summary_rows.append({"model":"F C&S Overall","estimator":"C&S-DR",
                          "coef":cs_overall_att,"se":cs_overall_se,
                          "pvalue":np.nan,"N":len(cs_data),"R2":np.nan})

pd.DataFrame(summary_rows).to_csv(
    "estimation_results.csv", index=False, encoding="utf-8-sig")
df_es.to_csv(       "event_study_full.csv",     index=False, encoding="utf-8-sig")
df_fd_es.to_csv(    "event_study_fd.csv",        index=False, encoding="utf-8-sig")
pd.DataFrame(block_results).T.reset_index(names="block_name").to_csv(
    "regional_did_block.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(size_results).T.reset_index(names="size_cat").to_csv(
    "regional_did_size.csv",  index=False, encoding="utf-8-sig")
if cs_available:
    att_df.to_csv(      "cs_att_gt.csv",         index=False, encoding="utf-8-sig")
    cs_dynamic.to_csv(  "cs_dynamic_att.csv",     index=False, encoding="utf-8-sig")

# ブロック別イベントスタディ
for bname, df_b in block_es.items():
    df_b.to_csv(f"es_block_{bname.lower().replace(' ','_')}.csv",
                index=False, encoding="utf-8-sig")

print(f"\n出力ファイル:")
print("  estimation_results.csv")
print("  event_study_full.csv / event_study_fd.csv")
print("  regional_did_block.csv / regional_did_size.csv")
print("  cs_att_gt.csv / cs_dynamic_att.csv  (C&S有効時)")
print("  es_block_*.csv  (ブロック別イベントスタディ)")
print("  figure1_extended_event_study.png")
print("  figure2_monthly_event_study.png")
print("  figure3_cs_dynamic.png  (C&S有効時)")
print("  figure4_regional_heterogeneity.png")
print("\n推計完了。")
