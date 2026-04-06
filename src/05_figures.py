#!/usr/bin/env python3
"""
05_figures.py
=============
"Legal Clarity and Collateral-Setting Behavior: Evidence from Real-Estate Registries"
Koki Arai — JSPS KAKENHI 23K01404

Purpose
-------
Generate all publication-quality figures from pre-computed result CSVs.
Run AFTER 02_main_estimation.py and 04_loan_market.py.

Figure mapping
--------------
Figure 1  Pre-treatment trend diagnostic (log mortgage/sale ratio by year)
Figure 2  Extended event study (annual bins + monthly post-treatment path)
Figure 3  Two-treatment model fitted values + C&S dynamic ATT comparison
Figure 4  Regional heterogeneity forest plot (block + size strata)
Figure 5  Syndicated loan substitution ITS (Section 6)

Inputs
------
results/event_study_full.csv        From 02_main_estimation.py
results/estimation_results.csv      From 02_main_estimation.py
results/cs_dynamic_att.csv          From 02_main_estimation.py
results/regional_did_block.csv      From 02_main_estimation.py
results/regional_did_size.csv       From 02_main_estimation.py
results/es_block_*.csv              From 02_main_estimation.py
results/section6_panel.csv          From 04_loan_market.py
data/registry_did.csv               For Figure 1 log-ratio computation

Outputs
-------
results/figures/figure1_trend_diagnostic.pdf  (.png)
results/figures/figure2_event_study.pdf       (.png)
results/figures/figure3_two_treatment.pdf     (.png)
results/figures/figure4_regional.pdf          (.png)
results/figures/figure5_loan_market.pdf       (.png)

Run
---
    python src/05_figures.py
"""

# =============================================================================
# 05_figures.py
# "Legal Clarity and Collateral-Setting Behavior: Evidence from Real-Estate Registries"
#
# Purpose:
#   Generate Figures 1–5 for the paper from pre-computed result CSVs.
#   This script can be run independently after 02_main_estimation.py and
#   04_loan_market_analysis.py have produced their output CSVs.
#
# Figure–content mapping (new v2 → original):
#   Figure 1  ←  original Figure 1            Pre-treatment trend diagnostic
#   Figure 2  ←  original Figure 1 + Figure 2a Extended event study (2 panels)
#   Figure 3  ←  original Figure 3            Two-treatment + C&S comparison
#   Figure 4  ←  original Figure 4            Regional heterogeneity forest plot
#   Figure 5  ←  original Figure 6.1 + 6.5   Syndicated loan substitution
#
# Note: "Figure N" labels are intentionally omitted from all figure panels.
#
# Inputs (from ../results/):
#   event_study_full.csv       Block D: detrended TWFE event study coefficients
#   estimation_results.csv     Block B/E/F summary (b_main, FD, C&S overall)
#   cs_dynamic_att.csv         Block F: Callaway–Sant'Anna dynamic ATT
#   regional_did_block.csv     Block G1: 8 geographic blocks
#   regional_did_size.csv      Block G2: 3 size strata
#   es_block_*.csv             Block G1b: block-level monthly event study paths
#   section6_panel.csv         Quarterly loan market panel (from 04)
#
# Inputs (from ../data/):
#   registry_did.csv           For Figure 1 annual log-ratio computation
#
# Outputs (to ../results/figures/):
#   figure1_trend_diagnostic.pdf / .png
#   figure2_event_study.pdf / .png
#   figure3_two_treatment.pdf / .png
#   figure4_regional_heterogeneity.pdf / .png
#   figure5_syndicated_loan.pdf / .png
#
# Required packages: pandas numpy scipy matplotlib
# =============================================================================

import os, glob, re, warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# ── Directories ───────────────────────────────────────────────────────────────
RESULTS_DIR = "../results"
FIG_DIR     = "../results/figures"
DATA_DIR    = "../data"
os.makedirs(FIG_DIR, exist_ok=True)

# ── Publication style ─────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "font.size":          11,
    "axes.titlesize":     11,
    "axes.titleweight":   "normal",
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    8.5,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   "lightgray",
    "lines.linewidth":    1.5,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.30,
    "grid.linestyle":     "--",
    "grid.color":         "#cccccc",
    "figure.dpi":         120,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
})

# ── Colour palette ────────────────────────────────────────────────────────────
C_MAIN    = "#2166AC"
C_CS      = "#762A83"
C_VERDICT = "#B2182B"
C_REFORM  = "#E08214"
C_BAND    = "#9ECAE1"
C_GRAY    = "#888888"

# ── Event-time constants ──────────────────────────────────────────────────────
VERDICT_REL  = 0
REFORM_REL   = 4   # Apr 2024 = tau=+4

# ── Block C two-treatment estimates (not written to estimation_results.csv) ───
COEF_VERDICT = -0.1865;  SE_VERDICT = 0.0226
COEF_REFORM  = -0.0670;  SE_REFORM  = 0.0159

# ── Utilities ─────────────────────────────────────────────────────────────────
def save_fig(fig, stem):
    for fmt in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"{stem}.{fmt}"))
        print(f"  saved: {stem}.{fmt}")

def stars(p):
    if pd.isna(p): return ""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""

def add_event_refs(ax, v=VERDICT_REL, r=REFORM_REL, label=True):
    ax.axvline(v, color=C_VERDICT, lw=1.1, ls=":",
               label="Verdict (Nov 2023)" if label else "_nolegend_")
    ax.axvline(r, color=C_REFORM,  lw=1.1, ls=":",
               label="Reform (Apr 2024)"  if label else "_nolegend_")
    ax.axhline(0, color="black", lw=0.8, ls="--", zorder=1)

def parse_quarter(s):
    roman = {"Ⅰ":1, "Ⅱ":2, "Ⅲ":3, "Ⅳ":4}
    m = re.match(r"(\d{4})[.\-]([IVⅠ-Ⅳ]+)", str(s))
    if m:
        yr = int(m.group(1))
        qn = roman.get(m.group(2).strip(), 1)
        return pd.Timestamp(yr, (qn - 1) * 3 + 1, 1)
    return pd.NaT

# ── Load CSVs ─────────────────────────────────────────────────────────────────
print("Loading result CSVs …")

df_es    = pd.read_csv(os.path.join(RESULTS_DIR, "event_study_full.csv"))
df_block = pd.read_csv(os.path.join(RESULTS_DIR, "regional_did_block.csv"))
df_size  = pd.read_csv(os.path.join(RESULTS_DIR, "regional_did_size.csv"))
df_est   = pd.read_csv(os.path.join(RESULTS_DIR, "estimation_results.csv"))

_cs_path = os.path.join(RESULTS_DIR, "cs_dynamic_att.csv")
df_cs    = pd.read_csv(_cs_path) if os.path.exists(_cs_path) else pd.DataFrame()

block_es = {}
for fp in sorted(glob.glob(os.path.join(RESULTS_DIR, "es_block_*.csv"))):
    bname = os.path.basename(fp).replace("es_block_","").replace(".csv","").title()
    block_es[bname] = pd.read_csv(fp)

df_s6 = pd.read_csv(os.path.join(RESULTS_DIR, "section6_panel.csv"))
df_s6["date"] = df_s6["quarter"].apply(parse_quarter)
df_s6 = df_s6.sort_values("date").reset_index(drop=True)
df_s6_est = df_s6[df_s6["date"] >= pd.Timestamp("2010-01-01")].copy()

# Key scalars
_b_row  = df_est[df_est["model"].str.contains("トレンド除去 コア", na=False)]
b_main  = float(_b_row["coef"].iloc[0]) if len(_b_row) else -0.0861
se_main = float(_b_row["se"].iloc[0])   if len(_b_row) else  0.0158

bins_df = df_es[df_es["kind"]=="bin"].dropna(subset=["coef"]).sort_values("rel_month")
mon_df  = df_es[df_es["kind"]=="monthly"].dropna(subset=["coef"]).sort_values("rel_month")
mon_pre = mon_df[mon_df["rel_month"] < 0]

VERDICT_DATE = pd.Timestamp("2024-01-01")
print("  Done.\n")


# =============================================================================
# FIGURE 1  Pre-treatment trend diagnostic
# =============================================================================
print("─"*60)
print("Figure 1  –  Pre-treatment trend diagnostic")
print("─"*60)

df_did  = pd.read_csv(os.path.join(DATA_DIR, "registry_did.csv"),
                       encoding="utf-8-sig", parse_dates=["date"])
df_pre  = df_did[df_did["post"] == 0].copy()
df_pre["log_ratio"] = df_pre["ln_mortgage"] - df_pre["ln_sale"]
annual  = (df_pre.groupby("year")["log_ratio"]
           .mean().reset_index().rename(columns={"log_ratio":"mean_ratio"}))

_x     = annual["year"].values - annual["year"].values[0]
_coef  = np.polyfit(_x, annual["mean_ratio"].values, 1)
_trend = np.polyval(_coef, _x)
_drift = _trend[-1] - _trend[0]

fig1, ax = plt.subplots(figsize=(8, 4.2))

ax.plot(annual["year"], annual["mean_ratio"],
        "o-", color=C_MAIN, lw=1.6, ms=6, zorder=4,
        label="Annual bureau-mean log ratio")
ax.plot(annual["year"], _trend,
        color=C_GRAY, lw=1.3, ls="--", zorder=3,
        label=f"Linear trend (OLS):  {_drift:+.3f} log pts over 2007–2023")
ax.fill_between(annual["year"],
                annual["mean_ratio"].values, _trend,
                where=(annual["year"] >= 2016),
                alpha=0.08, color=C_VERDICT, zorder=2)

ax.set_xlabel("Year")
ax.set_ylabel("ln(mortgage) − ln(sale transfers)\n[cross-bureau mean]")
ax.set_xlim(2006.3, 2023.7)
ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
ax.legend(loc="lower left")
ax.set_title("Pre-treatment divergence between mortgage and sale registrations,\n"
             "2007–2023  (motivating unit-specific linear detrending)")

plt.tight_layout()
save_fig(fig1, "figure1_trend_diagnostic")
plt.close(fig1)
print()


# =============================================================================
# FIGURE 2  Extended event study (two-panel)
# =============================================================================
print("─"*60)
print("Figure 2  –  Extended event study")
print("─"*60)

_pre_bin_mean = bins_df["coef"].mean()
_pre_bin_sd   = bins_df["coef"].std()
_pre_mon_mean = mon_pre["coef"].mean()
_pre_mon_sd   = mon_pre["coef"].std()

fig2, (ax_l, ax_r) = plt.subplots(
    1, 2, figsize=(13, 4.8),
    gridspec_kw={"width_ratios": [1.55, 1]}
)

# Left: pre-treatment validation
ax_l.bar(bins_df["rel_month"], bins_df["coef"],
         width=9.5, color=C_BAND, alpha=0.75, zorder=3,
         label="Annual bins  (τ ≤ −13 mo.)")
ax_l.errorbar(bins_df["rel_month"].values, bins_df["coef"].values,
              yerr=1.96*bins_df["se"].values,
              fmt="none", color=C_MAIN, capsize=3, lw=0.9, zorder=4)

ax_l.fill_between(mon_pre["rel_month"], mon_pre["ci_lo"], mon_pre["ci_hi"],
                  alpha=0.18, color=C_VERDICT, zorder=2)
ax_l.plot(mon_pre["rel_month"], mon_pre["coef"],
          "o-", color=C_VERDICT, lw=1.5, ms=4.5, zorder=5,
          label="Monthly  (τ ∈ [−12, −1])")

ax_l.axvline(-12.5, color=C_GRAY, lw=0.8, ls=":", alpha=0.6)
ax_l.axhline(0, color="black", lw=0.8, ls="--", zorder=1)

ax_l.text(0.03, 0.04,
          f"Annual bins:  mean = {_pre_bin_mean:+.3f},  SD = {_pre_bin_sd:.3f}  "
          f"(n = {len(bins_df)})\n"
          f"Monthly pre:  mean = {_pre_mon_mean:+.3f},  SD = {_pre_mon_sd:.3f}  "
          f"(n = {len(mon_pre)})",
          transform=ax_l.transAxes, fontsize=8, va="bottom",
          bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85,
                    ec="lightgray", lw=0.8))

ax_l.set_xlabel("Months relative to verdict")
ax_l.set_ylabel("Detrended DID coefficient (log)")
ax_l.set_title("(a)  Pre-treatment parallel-trends validation\n"
               "Annual bins (τ ≤ −13) and monthly (τ ∈ [−12, −1])")
ax_l.xaxis.set_major_locator(mticker.MultipleLocator(24))
ax_l.legend(loc="upper left")

# Right: post-ruling dynamic effect
mon_win = mon_df[(mon_df["rel_month"] >= -12) & (mon_df["rel_month"] <= 25)]

ax_r.fill_between(mon_win["rel_month"], mon_win["ci_lo"], mon_win["ci_hi"],
                  alpha=0.18, color=C_MAIN, zorder=2)
ax_r.plot(mon_win["rel_month"], mon_win["coef"],
          "o-", color=C_MAIN, lw=1.6, ms=5, zorder=4,
          label="Detrended TWFE")

add_event_refs(ax_r)

ax_r.text(0.04, 0.06,
          f"Pooled ATT:  δ̂ = {b_main:.3f}***\n(SE = {se_main:.4f})",
          transform=ax_r.transAxes, fontsize=8.5, va="bottom", color=C_MAIN,
          bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85,
                    ec="lightgray", lw=0.8))

ax_r.set_xlabel("Months relative to verdict")
ax_r.set_ylabel("Detrended DID coefficient (log)")
ax_r.set_title("(b)  Post-ruling dynamic effect\n"
               "Monthly resolution  (τ ∈ [−12, +25])")
ax_r.xaxis.set_major_locator(mticker.MultipleLocator(4))
ax_r.legend(loc="lower right")

plt.tight_layout()
save_fig(fig2, "figure2_event_study")
plt.close(fig2)
print()


# =============================================================================
# FIGURE 3  Two-treatment + C&S comparison (two-panel)
# =============================================================================
print("─"*60)
print("Figure 3  –  Two-treatment model and C&S comparison")
print("─"*60)

has_cs   = not df_cs.empty
_figsize = (8.5, 9.0) if has_cs else (8.5, 4.2)
_hratio  = [1, 1.5]  if has_cs else [1]

fig3 = plt.figure(figsize=_figsize)
if has_cs:
    gs    = fig3.add_gridspec(2, 1, hspace=0.45, height_ratios=_hratio)
    ax_t  = fig3.add_subplot(gs[0])
    ax_b  = fig3.add_subplot(gs[1])
else:
    ax_t  = fig3.add_subplot(1, 1, 1)
    ax_b  = None

# Top: three-estimate bar chart
_labels = ["Pooled\n(detrended TWFE)",
           "Verdict effect\n(Dec 2023 – Mar 2024)",
           "Reform effect\n(Apr 2024 onward)"]
_coefs  = [b_main,      COEF_VERDICT, COEF_REFORM]
_ses    = [se_main,     SE_VERDICT,   SE_REFORM]
_colors = [C_MAIN,      C_VERDICT,    C_REFORM]
_pvals  = [4.65e-8,     0.0,          0.0]

for i, (c, s, col, p) in enumerate(zip(_coefs, _ses, _colors, _pvals)):
    ax_t.bar(i, c, width=0.42, color=col, alpha=0.78,
             yerr=1.96*s, capsize=7,
             error_kw=dict(lw=1.5, capthick=1.5, ecolor="black"))
    st = stars(p)
    if st:
        ax_t.text(i, c * 1.08, st, ha="center", va="bottom",
                  fontsize=12, color="black", fontweight="bold")
    ax_t.text(i, c / 2, f"{c:.3f}",
              ha="center", va="center", fontsize=9.5,
              color="white", fontweight="bold")

ax_t.axhline(0, color="black", lw=0.8, ls="--")
ax_t.set_xticks([0, 1, 2])
ax_t.set_xticklabels(_labels, fontsize=9.5)
ax_t.set_ylabel("Estimated effect (log, detrended)")
ax_t.set_ylim(min(_coefs) * 1.6, 0.06)
ax_t.set_title("(a)  Pooled and two-treatment model estimates")

# Bottom: C&S vs TWFE monthly
if ax_b is not None:
    cs_trim  = df_cs[(df_cs["rel_month"] >= -24) &
                     (df_cs["rel_month"] <= 25)].dropna(subset=["att"])
    mon_trim = mon_df[(mon_df["rel_month"] >= -24) &
                      (mon_df["rel_month"] <= 25)]

    ax_b.fill_between(cs_trim["rel_month"], cs_trim["ci_lo"], cs_trim["ci_hi"],
                      alpha=0.15, color=C_CS, zorder=2)
    ax_b.plot(cs_trim["rel_month"], cs_trim["att"],
              "s--", color=C_CS, lw=1.5, ms=4.5, zorder=4,
              label="Callaway–Sant'Anna ATT")

    ax_b.fill_between(mon_trim["rel_month"], mon_trim["ci_lo"], mon_trim["ci_hi"],
                      alpha=0.15, color=C_MAIN, zorder=2)
    ax_b.plot(mon_trim["rel_month"], mon_trim["coef"],
              "o-", color=C_MAIN, lw=1.5, ms=4.5, zorder=4, alpha=0.85,
              label="Detrended TWFE (monthly)")

    add_event_refs(ax_b)
    ax_b.set_xlabel("Months relative to verdict")
    ax_b.set_ylabel("ATT / DID coefficient (log)")
    ax_b.set_title("(b)  Callaway–Sant'Anna (2021) dynamic ATT vs. "
                   "detrended TWFE\n(τ ∈ [−24, +25])")
    ax_b.xaxis.set_major_locator(mticker.MultipleLocator(4))
    ax_b.legend(loc="lower left", ncol=2)

save_fig(fig3, "figure3_two_treatment")
plt.close(fig3)
print()


# =============================================================================
# FIGURE 4  Regional heterogeneity forest plot (two-panel)
# =============================================================================
print("─"*60)
print("Figure 4  –  Regional heterogeneity")
print("─"*60)

for _df in [df_block, df_size]:
    for col in ["coef", "se", "pvalue"]:
        _df[col] = _df[col].astype(float)

df_block = df_block.sort_values("coef").reset_index(drop=True)

def blabel(row):
    return f"{row['block_name']}  ({int(row['n_bureaus'])}){stars(row['pvalue'])}"

def slabel(row):
    return f"{row['size_cat'].capitalize()}  ({int(row['n_bureaus'])} bur.)"

b_labels = [blabel(r) for _, r in df_block.iterrows()]

sz_order = ["large", "medium", "small"]
df_sz    = df_size.set_index("size_cat").loc[sz_order].reset_index()
s_labels = [slabel(r) for _, r in df_sz.iterrows()]

fig4, (ax4l, ax4r) = plt.subplots(
    1, 2, figsize=(12.5, 5.5),
    gridspec_kw={"width_ratios": [1.6, 1]}
)

# Left: geographic blocks
for i, (_, row) in enumerate(df_block.iterrows()):
    col = C_MAIN if row["pvalue"] < 0.05 else C_GRAY
    ax4l.errorbar(row["coef"], i, xerr=1.96*row["se"],
                  fmt="o", color=col, capsize=5, ms=8, lw=1.4, elinewidth=1.2)

ax4l.axvline(0,      color="black", lw=0.8, ls="--")
ax4l.axvline(b_main, color=C_MAIN,  lw=1.0, ls=":", alpha=0.7)
ax4l.set_yticks(range(len(df_block)))
ax4l.set_yticklabels(b_labels, fontsize=9.5)
ax4l.set_xlabel("Detrended DID coefficient (log)")
ax4l.set_title("(a)  By geographic block\n(sorted by point estimate)")
ax4l.legend(handles=[
    Line2D([0],[0], marker="o", color="w", markerfacecolor=C_MAIN,  ms=9, label="p < 0.05"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor=C_GRAY,  ms=9, label="p ≥ 0.05"),
    Line2D([0],[0], color=C_MAIN, lw=1, ls=":",
           label=f"Full-sample δ̂ = {b_main:.3f}"),
], loc="upper right", fontsize=8.5)

# Right: size strata
sz_colors = [C_VERDICT, C_MAIN, "#08519C"]

for i, (_, row) in enumerate(df_sz.iterrows()):
    ax4r.errorbar(row["coef"], i, xerr=1.96*row["se"],
                  fmt="D", color=sz_colors[i], capsize=5, ms=9,
                  lw=1.4, elinewidth=1.2)
    ax4r.text(row["coef"] + 0.007, i,
              f"{row['coef']:.3f}{stars(row['pvalue'])}",
              va="center", fontsize=9.5, color=sz_colors[i])

ax4r.axvline(0,      color="black", lw=0.8, ls="--")
ax4r.axvline(b_main, color=C_MAIN,  lw=1.0, ls=":", alpha=0.7)
ax4r.set_yticks(range(len(df_sz)))
ax4r.set_yticklabels(s_labels, fontsize=10)
ax4r.set_xlabel("Detrended DID coefficient (log)")
ax4r.set_title("(b)  By bureau size\n(pre-treatment mean volume)")
ax4r.annotate("← Larger effect in\n   peripheral bureaus",
              xy=(0.04, 0.14), xycoords="axes fraction",
              fontsize=8, style="italic", color="#555555")

plt.tight_layout()
save_fig(fig4, "figure4_regional_heterogeneity")
plt.close(fig4)
print()


# =============================================================================
# FIGURE 5  Syndicated loan substitution (two-panel)
# =============================================================================
print("─"*60)
print("Figure 5  –  Syndicated loan market substitution")
print("─"*60)

# Unit: 億円.  Divide by 10,000 for ¥ trillion.
SCALE = 1 / 10_000

# ITS specification matching 04_loan_market_analysis.py:
#   formula: ln(syn_amount) ~ t + Q2 + Q3 + Q4 + post + t_post
#   post     = 1 from 2024Q1 onward (first clean post-ruling quarter)
#   t_post   = 1, 2, 3 ... from 2024Q1 (slope change)
#   transition quarter (2023Q4) excluded from estimation sample
#   Q1 is the base category
try:
    import statsmodels.formula.api as smf
    _HAS_SM = True
except ImportError:
    _HAS_SM = False

VERDICT_Q  = pd.Timestamp("2023-10-01")   # 2023Q4 — transition (excluded from estimation)
POST_START = pd.Timestamp("2024-01-01")   # 2024Q1 — first clean post-ruling quarter

_s6 = df_s6_est.dropna(subset=["syn_amount_total","date"]).copy()
_s6["t"]      = np.arange(len(_s6))
_s6["post"]   = (_s6["date"] >= POST_START).astype(int)
_s6["t_post"] = np.where(_s6["post"] == 1,
                          _s6["t"] - _s6.loc[_s6["post"]==1,"t"].min() + 1, 0)
_s6["Q2"] = (_s6["date"].dt.month == 4).astype(int)
_s6["Q3"] = (_s6["date"].dt.month == 7).astype(int)
_s6["Q4"] = (_s6["date"].dt.month == 10).astype(int)
_s6["ln_y"] = np.log(_s6["syn_amount_total"])

# Estimation sample: exclude transition quarter
_samp = _s6[_s6["date"] != VERDICT_Q].dropna(subset=["ln_y"]).copy()

if _HAS_SM:
    _res = smf.ols("ln_y ~ t + Q2 + Q3 + Q4 + post + t_post", data=_samp).fit()
    _fit_all = np.exp(_res.predict(_s6)) * SCALE
    _fit_cf  = np.exp(_res.predict(_s6.assign(post=0, t_post=0))) * SCALE
else:
    # Fallback: numpy OLS
    _X = np.column_stack([np.ones(len(_samp)), _samp["t"].values,
                          _samp["Q2"].values, _samp["Q3"].values, _samp["Q4"].values,
                          _samp["post"].values, _samp["t_post"].values])
    _beta = np.linalg.lstsq(_X, _samp["ln_y"].values, rcond=None)[0]
    _X_all = np.column_stack([np.ones(len(_s6)), _s6["t"].values,
                               _s6["Q2"].values, _s6["Q3"].values, _s6["Q4"].values,
                               _s6["post"].values, _s6["t_post"].values])
    _fit_all = np.exp(_X_all @ _beta) * SCALE
    _Xcf = _X_all.copy(); _Xcf[:,5] = 0; _Xcf[:,6] = 0
    _fit_cf = np.exp(_Xcf @ _beta) * SCALE

_dates   = _s6["date"].values
_amount  = _s6["syn_amount_total"].values * SCALE
_is_post = (_s6["date"] >= POST_START).values
_is_trans = (_s6["date"] == VERDICT_Q).values

_sb    = df_s6_est.dropna(subset=["syn_outstanding_total","date"]).copy()
_out   = _sb["syn_outstanding_total"].values * SCALE
_dts_b = _sb["date"].values
_post_b = (_sb["date"] >= VERDICT_DATE).values

fig5, (ax5l, ax5r) = plt.subplots(1, 2, figsize=(13, 4.8))

# Left: origination + ITS
ax5l.bar(_dates[~_is_post & ~_is_trans], _amount[~_is_post & ~_is_trans], width=70,
         color=C_BAND, alpha=0.80, zorder=2, label="Origination (pre-ruling)")
ax5l.bar(_dates[_is_post],  _amount[_is_post],  width=70,
         color=C_MAIN, alpha=0.85, zorder=2, label="Origination (post-ruling)")
ax5l.bar(_dates[_is_trans], _amount[_is_trans],  width=70,
         color=C_REFORM, alpha=0.70, zorder=2, label="Transition quarter (2023Q4)")
ax5l.plot(_dates, _fit_all, color="black", lw=1.8, ls="-",  zorder=4, label="ITS fit")
ax5l.plot(_dates, _fit_cf,  color=C_GRAY,  lw=1.4, ls="--", zorder=3,
          label="Counterfactual")
ax5l.axvline(VERDICT_Q, color=C_VERDICT, lw=1.1, ls=":",
             label="Ruling (2023Q4)", zorder=5)

ax5l.set_xlabel("Quarter")
ax5l.set_ylabel("Syndicated loan origination (¥ trillion, flow)")
ax5l.set_title("(a)  Syndicated loan origination and ITS fit,  2010Q1–2025Q4")
ax5l.xaxis.set_major_locator(mdates.YearLocator(2))
ax5l.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.setp(ax5l.xaxis.get_majorticklabels(), rotation=30, ha="right")
ax5l.legend(loc="upper left", fontsize=8, ncol=2)

# Right: outstanding balance
ax5r.fill_between(_dts_b, _out, alpha=0.25, color=C_MAIN, zorder=2)
ax5r.plot(_dts_b[~_post_b], _out[~_post_b], color=C_MAIN, lw=1.6, zorder=3)
ax5r.plot(_dts_b[_post_b],  _out[_post_b],  color=C_MAIN, lw=2.2, zorder=4)
ax5r.axvline(VERDICT_Q, color=C_VERDICT, lw=1.1, ls=":",
             label="Ruling (2023Q4)", zorder=5)

_pre_mean  = _out[~_post_b].mean()
_post_mean = _out[_post_b].mean() if _post_b.any() else np.nan
if not np.isnan(_post_mean):
    ax5r.annotate(
        f"Post mean: ¥{_post_mean:.0f}T\nPre mean:  ¥{_pre_mean:.0f}T",
        xy=(VERDICT_Q, _post_mean * 0.95),
        xytext=(45, -45), textcoords="offset points",
        fontsize=8.5, color=C_MAIN,
        arrowprops=dict(arrowstyle="-", color=C_MAIN, lw=0.8),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.88,
                  ec="lightgray", lw=0.8))

ax5r.set_xlabel("Quarter")
ax5r.set_ylabel("Syndicated loan outstanding balance (¥ trillion)")
ax5r.set_title("(b)  Outstanding balance — secular growth context,  2010Q1–2025Q4")
ax5r.xaxis.set_major_locator(mdates.YearLocator(2))
ax5r.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.setp(ax5r.xaxis.get_majorticklabels(), rotation=30, ha="right")
ax5r.legend(loc="upper left", fontsize=9)

plt.tight_layout()
save_fig(fig5, "figure5_syndicated_loan")
plt.close(fig5)
print()


# =============================================================================
# Summary
# =============================================================================
print("="*60)
print("Output directory:", os.path.abspath(FIG_DIR))
print("="*60)
for stem in ["figure1_trend_diagnostic", "figure2_event_study",
             "figure3_two_treatment", "figure4_regional_heterogeneity",
             "figure5_syndicated_loan"]:
    for ext in ("pdf", "png"):
        p = os.path.join(FIG_DIR, f"{stem}.{ext}")
        print(f"  {'✓' if os.path.exists(p) else '✗ MISSING'}  {stem}.{ext}")
