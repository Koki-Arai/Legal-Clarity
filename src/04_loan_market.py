#!/usr/bin/env python3
"""
04_loan_market.py
=================
"Legal Clarity and Collateral-Setting Behavior: Evidence from Real-Estate Registries"
Koki Arai — JSPS KAKENHI 23K01404

Purpose
-------
Section 6 supplementary analysis: substitution effects in the loan market.
Uses Bank of Japan "Loan Market Transaction Survey" (貸出債権等の流動化) data.

Channels estimated
------------------
CH2  Syndicated loan substitution (ITS with seasonal adjustment)
CH3  Performing-loan secondary market activity
CH4  Term loan vs. commitment line composition shift
CH5  Private vs. listed borrower heterogeneity

Identification note
-------------------
This section provides supplementary mechanism evidence, NOT a fully
identified causal estimate. The Bank of Japan survey covers primarily
large financial institutions; regional banks and credit unions are not
represented. Results should be interpreted as a lower bound on
substitution for large-market borrowers (Section 6.6).

Inputs
------
data/cdab_1_.csv               BoJ loan market transaction survey
                               (貸出債権等の流動化、quarterly)

Outputs
-------
results/section6_panel.csv     Quarterly panel for Figure 5
results/section6_results.csv   ITS regression results (Table 7)

Run
---
    python src/04_loan_market.py
"""

#!/usr/bin/env python3
"""
section6_analysis.py
──────────────────────────────────────────────────────────────────────────────
第6節「Substitution Effects in the Loan Market」分析スクリプト
Google Colab 用

【分析の理論的視角】
Section 3 の理論モデルより、2023年11月判決は次の3チャネルで貸出市場に
影響を与える（Proposition 2・Hypothesis 5）：

  チャネル1「担保コスト上昇」:
    σ（登記コスト）↑ → 個別抵当権設定↓（第5節で確認済み）
    → 資金需要が消えるわけでない → 代替的信用手段へのシフト

  チャネル2「シンジケート・ローン代替仮説」:
    抵当権コスト↑ → 無担保・プール担保型のシンジケート・ローンへのシフト
    予測：判決後、シンジケート・ローン組成件数・金額が増加
    ただし Q1（1・4月）に組成が集中する強い季節性を除去する必要あり

  チャネル3「貸出債権流動化チャネル」:
    抵当権の法的不確実性↑ → 既存担保付き債権の流動化（売却・証券化）↑
    → 銀行がバランスシートから担保リスクを移転
    予測：正常債権の流動化（売却）が増加
    逆に：不良債権流動化は訴訟リスク低下で減少する可能性（予測が逆転）

  チャネル4「構成変化仮説」:
    シンジケート内のタームローン vs. コミットメントライン比率の変化
    担保に依存するタームローン（有担保型）↓、コミットメント（信用枠型）↑ の予測
    → 担保コスト上昇を反映した構成シフト

  チャネル5「民間企業vs上場企業」:
    非公開企業（担保依存度大）> 上場企業（資本市場アクセスあり）で効果大
    Proposition 3 の市場厚さ仮説の別テスト

【推計戦略】
(1) 記述統計・時系列可視化（季節調整前後）
(2) 介入前後比較（単純平均比較 + 変化率）
(3) 四半期ダミーを用いた介入効果の回帰（ITS：interrupted time series）
    - 季節調整（Q固定効果）+ 線形トレンド + 判決ダミー
    - アウトカム：ln(amount) / ln(deals)
(4) 構造変化検定（Chow test）
(5) チャネル別効果の比較（タームローン vs. コミットメント、公開 vs. 非公開）
(6) 担保シフト仮説の検証（mortgage（第5節）と syn/perf の逆相関）

入力ファイル（同フォルダに置くこと）:
    cdab_1_.csv   ← 貸出債権市場取引動向（日銀統計）

出力:
    section6_panel.csv     … 整形済みパネルデータ
    section6_results.csv   … 推計結果サマリー
    section6_figures/      … 図（PNG）
──────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
import re
import warnings
import os
warnings.filterwarnings("ignore")

# ── matplotlib / scipy ─────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
from scipy.stats import f as f_dist

# ── statsmodels ────────────────────────────────────────────────────────────
try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.stattools import durbin_watson
    HAS_SM = True
except ImportError:
    HAS_SM = False
    print("[WARNING] statsmodels not found. Regression results will be skipped.")
    print("  Run: pip install statsmodels")

# ── パス設定 ────────────────────────────────────────────────────────────────
INPUT_PATH = "cdab_1_.csv"
OUT_PANEL  = "section6_panel.csv"
OUT_RESULT = "section6_results.csv"
FIG_DIR    = "section6_figures"
os.makedirs(FIG_DIR, exist_ok=True)

SEP = "=" * 72

# 判決日（四半期）
VERDICT_Q  = pd.Timestamp(2023, 10, 1)   # 2023Q4 (2023.Ⅳ) が判決を含む最初の四半期
POST_START = pd.Timestamp(2024,  1, 1)   # 2024Q1 以降を「処置後」（純粋に判決後）
TRANSITION = pd.Timestamp(2023, 10, 1)   # 2023Q4 は移行期
# 推計用サンプル開始（安定的な期間：GFC後）
SAMPLE_START = pd.Timestamp(2010, 1, 1)

# ══════════════════════════════════════════════════════════════════════════
# 1. データ読み込み・整形
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}\n1. データ読み込み\n{SEP}")

df_raw = pd.read_csv(INPUT_PATH, encoding='CP932', header=None)

COL_MAP = {
    0:  'quarter',
    1:  'syn_deals_total',
    2:  'syn_deals_term',
    3:  'syn_deals_commit',
    4:  'syn_amount_total',
    5:  'syn_amount_term',
    6:  'syn_amount_commit',
    7:  'syn_outstanding_total',
    8:  'syn_outstanding_term',
    9:  'syn_outstanding_commit',
    10: 'perf_n_total',
    11: 'perf_n_sale',
    12: 'perf_n_trust',
    13: 'perf_n_participation',
    14: 'perf_amt_total',
    15: 'perf_amt_sale',
    16: 'perf_amt_trust',
    17: 'perf_amt_participation',
    18: 'nperf_n_total',
    19: 'nperf_n_sale',
    20: 'nperf_n_trust',
    21: 'nperf_n_participation',
    22: 'nperf_amt_total',
    23: 'nperf_amt_sale',
    24: 'nperf_amt_trust',
    25: 'nperf_amt_participation',
    26: 'syn_deals_term_listed',
    27: 'syn_deals_term_private',
    28: 'syn_deals_commit_listed',
    29: 'syn_deals_commit_private',
    30: 'syn_amt_term_listed',
    31: 'syn_amt_term_private',
    32: 'syn_amt_commit_listed',
    33: 'syn_amt_commit_private',
    34: 'syn_out_term_listed',
    35: 'syn_out_term_private',
    36: 'syn_out_commit_listed',
    37: 'syn_out_commit_private',
    38: 'perf_n_to_fi',
    39: 'perf_n_to_spc',
    40: 'perf_amt_to_fi',
    41: 'perf_amt_to_spc',
}

def parse_num(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().replace(',', '')
    try: return float(s)
    except: return np.nan

def parse_q(s):
    s = str(s).strip()
    # Unicode circled Roman numerals: Ⅰ=U+2160, Ⅱ=U+2161, Ⅲ=U+2162, Ⅳ=U+2163
    ROMAN_MAP = {'\u2160': 1, '\u2161': 2, '\u2162': 3, '\u2163': 4,
                 'I': 1, 'II': 2, 'III': 3, 'IV': 4}
    m = re.match(r'(\d{4})\.(.+)', s)
    if not m: return pd.NaT
    y = int(m.group(1))
    roman = m.group(2).strip()
    qnum = ROMAN_MAP.get(roman)
    if qnum is None: return pd.NaT
    month = {1: 1, 2: 4, 3: 7, 4: 10}[qnum]
    return pd.Timestamp(y, month, 1)

rows = []
for i in range(11, len(df_raw)):
    row = df_raw.iloc[i]
    q_str = str(row[0]).strip()
    if not re.match(r'\d{4}\.', q_str):
        continue
    d = {'quarter': q_str}
    for c, name in COL_MAP.items():
        if c > 0:
            d[name] = parse_num(row[c])
    rows.append(d)

panel = pd.DataFrame(rows)
panel['date']  = panel['quarter'].apply(parse_q)
panel['year']  = panel['date'].dt.year
panel['qnum']  = (panel['date'].dt.month - 1) // 3 + 1
panel['t']     = range(len(panel))   # 通し時間インデックス

# ── 派生変数 ──────────────────────────────────────────────────────────────

# 構成比
panel['share_term']    = panel['syn_amount_term']    / panel['syn_amount_total']
panel['share_commit']  = panel['syn_amount_commit']  / panel['syn_amount_total']
panel['share_private'] = (panel['syn_amt_term_private'] + panel['syn_amt_commit_private']) \
                         / panel['syn_amount_total']
panel['share_listed']  = (panel['syn_amt_term_listed']  + panel['syn_amt_commit_listed'])  \
                         / panel['syn_amount_total']

# 流動化：正常+不良合計
panel['trans_n_total']   = panel['perf_n_total']   + panel['nperf_n_total']
panel['trans_amt_total'] = panel['perf_amt_total']  + panel['nperf_amt_total']

# SPC向け正常債権流動化（証券化バロメータ）
panel['perf_spc_share'] = panel['perf_n_to_spc'] / panel['perf_n_total']

# ローン平均規模（1件あたり組成金額）
panel['syn_avg_size']  = panel['syn_amount_total'] / panel['syn_deals_total']
panel['perf_avg_size'] = panel['perf_amt_total']   / panel['perf_n_total']

# 対数変換
log_vars = [
    'syn_deals_total', 'syn_deals_term', 'syn_deals_commit',
    'syn_amount_total', 'syn_amount_term', 'syn_amount_commit',
    'syn_outstanding_total',
    'perf_n_total', 'perf_amt_total',
    'nperf_n_total', 'nperf_amt_total',
    'trans_n_total', 'trans_amt_total',
    'syn_deals_term_listed', 'syn_deals_term_private',
    'syn_amt_term_listed', 'syn_amt_term_private',
    'syn_avg_size', 'perf_avg_size',
]
for v in log_vars:
    panel[f'ln_{v}'] = np.log(panel[v].replace(0, np.nan))

# ── ダミー変数 ────────────────────────────────────────────────────────────
panel['post']       = (panel['date'] >= POST_START).astype(int)
panel['transition'] = (panel['date'] == TRANSITION).astype(int)
panel['post_incl_trans'] = (panel['date'] >= VERDICT_Q).astype(int)

# 四半期ダミー
for q in [1, 2, 3, 4]:
    panel[f'Q{q}'] = (panel['qnum'] == q).astype(int)

# ITS 用：判決後の時間トレンド
panel['t_post'] = np.where(panel['post'] == 1,
                           panel['t'] - panel.loc[panel['post']==1, 't'].min() + 1,
                           0)

print(f"  期間: {panel['quarter'].iloc[0]} ～ {panel['quarter'].iloc[-1]}  ({len(panel)} 四半期)")
print(f"  判決前: {(panel['post']==0).sum()} 四半期（移行期除く）")
print(f"  判決後: {(panel['post']==1).sum()} 四半期")

panel.to_csv(OUT_PANEL, index=False, encoding='utf-8-sig')
print(f"  → {OUT_PANEL} を保存")

# ══════════════════════════════════════════════════════════════════════════
# 2. 記述統計
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}\n2. 記述統計（処置前後比較）\n{SEP}")

# 推計サンプル（2010Q1以降）
samp = panel[panel['date'] >= SAMPLE_START].copy()
pre  = samp[samp['post'] == 0]
post = samp[samp['post'] == 1]
# 直近4四半期のプレ（2022Q4〜2023Q3）
pre_recent = samp[(samp['date'] >= pd.Timestamp(2022, 10, 1)) &
                  (samp['date'] <  VERDICT_Q)]

vars_desc = [
    ('syn_amount_total',    'Syndicated loan amount (100m JPY)',       'Channel 2'),
    ('syn_deals_total',     'Syndicated loan deals (no.)',             'Channel 2'),
    ('syn_avg_size',        'Avg deal size (100m JPY/deal)',           'Channel 2'),
    ('syn_amount_term',     '  Term loan amount',                      'Channel 4'),
    ('syn_amount_commit',   '  Commitment line amount',                'Channel 4'),
    ('share_term',          '  Term share (amount)',                   'Channel 4'),
    ('syn_amt_term_listed', '  Term: listed companies amount',         'Channel 5'),
    ('syn_amt_term_private','  Term: private companies amount',        'Channel 5'),
    ('perf_amt_total',      'Performing loans transferred (100m)',     'Channel 3'),
    ('perf_n_total',        'Performing loans transferred (no.)',      'Channel 3'),
    ('nperf_amt_total',     'Non-performing loans transferred (100m)', 'Channel 3'),
    ('nperf_n_total',       'Non-performing loans transferred (no.)',  'Channel 3'),
    ('perf_spc_share',      'Perf. loans to SPC share',               'Channel 3'),
]

print(f"\n  {'Variable':45s} {'Channel':10s} {'Pre mean':>12s} {'Post mean':>12s} {'Change%':>9s}  {'Pre(recent4Q)':>14s}")
print("  " + "-"*110)
for var, label, ch in vars_desc:
    if var not in pre.columns: continue
    pm  = pre[var].mean()
    pom = post[var].mean()
    prm = pre_recent[var].mean()
    chg = 100*(pom - pm)/pm if pm > 0 else np.nan
    print(f"  {label:45s} {ch:10s} {pm:>12,.1f} {pom:>12,.1f} {chg:>+8.1f}%  {prm:>14,.1f}")

# ══════════════════════════════════════════════════════════════════════════
# 3. Interrupted Time Series (ITS) 推計
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}\n3. Interrupted Time Series 回帰\n{SEP}")
print("""
モデル:
  y_t = α + β·t + γ₁·Q2 + γ₂·Q3 + γ₃·Q4
      + δ₁·Post_t + δ₂·t_post
      + ε_t

  y_t       : ln(アウトカム変数)
  t         : 通し時間トレンド（四半期）
  Q2,Q3,Q4  : 季節ダミー（Q1基準）
  Post_t    : 判決後ダミー（2024Q1以降=1）
  t_post    : 判決後の時間トレンド（slope change）
  δ₁        : 判決直後の「レベルシフト」= 主関心
  δ₂        : 判決後の追加トレンド変化
""")

if not HAS_SM:
    print("  statsmodels が必要です。スキップします。")
else:
    results_list = []

    ITS_TARGETS = [
        # (変数名, ラベル, チャネル, 理論予測の符号)
        ('ln_syn_amount_total',     'ln(Syndicated amount)',          'CH2', '+'),
        ('ln_syn_deals_total',      'ln(Syndicated deals)',           'CH2', '+'),
        ('ln_syn_amount_term',      'ln(Syn term loan amount)',       'CH4', '?'),
        ('ln_syn_amount_commit',    'ln(Syn commitment amount)',      'CH4', '+'),
        ('ln_syn_amt_term_listed',  'ln(Term: listed companies)',     'CH5', '+'),
        ('ln_syn_amt_term_private', 'ln(Term: private companies)',    'CH5', '+'),
        ('ln_perf_amt_total',       'ln(Perf. transferred amount)',   'CH3', '+'),
        ('ln_perf_n_total',         'ln(Perf. transferred deals)',    'CH3', '+'),
        ('ln_nperf_amt_total',      'ln(Non-perf. transferred amt)', 'CH3', '?'),
    ]

    for var, label, ch, expected in ITS_TARGETS:
        sub = samp[samp[var].notna() & samp['date'].notna()].copy()
        if len(sub) < 20:
            print(f"  [SKIP] {label}: n={len(sub)} (not enough obs)")
            continue

        formula = f"{var} ~ t + Q2 + Q3 + Q4 + post + t_post"
        try:
            res = smf.ols(formula, data=sub).fit(cov_type='HC3')
        except Exception as e:
            print(f"  [ERROR] {label}: {e}")
            continue

        d1 = res.params.get('post',   np.nan)
        d2 = res.params.get('t_post', np.nan)
        se1 = res.bse.get('post',   np.nan)
        se2 = res.bse.get('t_post', np.nan)
        p1  = res.pvalues.get('post',   np.nan)
        p2  = res.pvalues.get('t_post', np.nan)
        r2  = res.rsquared
        n   = int(res.nobs)
        dw  = durbin_watson(res.resid)

        def stars(p):
            if p < 0.01: return '***'
            if p < 0.05: return '**'
            if p < 0.10: return '*'
            return ''

        print(f"\n  [{ch}] {label}  (N={n}, R²={r2:.3f}, DW={dw:.2f})")
        print(f"    δ₁ (level shift) = {d1:+.4f}{stars(p1)}  SE={se1:.4f}  p={p1:.3f}  [expected: {expected}]")
        print(f"    δ₂ (slope change) = {d2:+.4f}{stars(p2)}  SE={se2:.4f}  p={p2:.3f}")

        results_list.append({
            'variable': label, 'channel': ch, 'expected': expected,
            'delta1': round(d1, 4), 'se1': round(se1, 4), 'p1': round(p1, 3),
            'delta2': round(d2, 4), 'se2': round(se2, 4), 'p2': round(p2, 3),
            'R2': round(r2, 4), 'N': n, 'DW': round(dw, 2),
            'stars1': stars(p1), 'stars2': stars(p2),
        })

    results_df = pd.DataFrame(results_list)
    results_df.to_csv(OUT_RESULT, index=False, encoding='utf-8-sig')
    print(f"\n  → {OUT_RESULT} を保存")

# ══════════════════════════════════════════════════════════════════════════
# 4. 構造変化検定（Chow Test）
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}\n4. Chow 構造変化検定\n{SEP}")
print("  H0: 判決前後で回帰係数に構造変化なし")

if HAS_SM:
    def chow_test(y, t, q2, q3, q4, split_t):
        """
        シンプル Chow 検定：split_t を境に前後に分割し、
        F統計量（制約なし vs. 制約あり）を計算。
        """
        X = sm.add_constant(np.column_stack([t, q2, q3, q4]))
        pre_idx  = t <  split_t
        post_idx = t >= split_t
        y_a, X_a = y[~np.isnan(y)], X[~np.isnan(y)]
        pre_mask  = pre_idx[~np.isnan(y)]
        post_mask = post_idx[~np.isnan(y)]
        if pre_mask.sum() < 6 or post_mask.sum() < 4:
            return np.nan, np.nan
        rss_full = (sm.OLS(y_a[pre_mask],  X_a[pre_mask]).fit().ssr +
                    sm.OLS(y_a[post_mask], X_a[post_mask]).fit().ssr)
        rss_restr = sm.OLS(y_a, X_a).fit().ssr
        k = X.shape[1]
        n = len(y_a)
        F = ((rss_restr - rss_full) / k) / (rss_full / (n - 2*k))
        p = 1 - f_dist.cdf(F, k, n - 2*k)
        return round(F, 3), round(p, 4)

    split_t = samp.loc[samp['date'] == POST_START, 't'].values
    if len(split_t) > 0:
        split_t = int(split_t[0])
        for var, label, ch, _ in ITS_TARGETS[:5]:
            y_arr = samp[var].values
            t_arr = samp['t'].values
            q2_arr = samp['Q2'].values
            q3_arr = samp['Q3'].values
            q4_arr = samp['Q4'].values
            F, p = chow_test(y_arr, t_arr, q2_arr, q3_arr, q4_arr, split_t)
            stars = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''
            print(f"  [{ch}] {label:45s}  F={F:7.3f}  p={p:.4f} {stars}")

# ══════════════════════════════════════════════════════════════════════════
# 5. 構成比変化の検定（タームローン vs. コミットメント）
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}\n5. 構成比変化の検定（チャネル4・チャネル5）\n{SEP}")
print("  担保コスト↑ → タームローン（担保依存）↓、コミットメントライン（信用枠）↑")
print("  非公開企業（担保依存）> 上場企業（資本市場）の効果差")

pre_s  = samp[samp['post'] == 0]
post_s = samp[samp['post'] == 1]

shares = [
    ('share_term',    'Term loan share (amount)',        '負'),
    ('share_commit',  'Commitment line share (amount)',  '正'),
    ('share_private', 'Private company share',          '正'),
    ('share_listed',  'Listed company share',           '負'),
]
for var, label, exp in shares:
    if var not in samp.columns: continue
    pre_v  = pre_s[var].dropna()
    post_v = post_s[var].dropna()
    t_stat, p_val = stats.ttest_ind(pre_v, post_v, equal_var=False)
    stars = '***' if p_val < 0.01 else '**' if p_val < 0.05 else '*' if p_val < 0.10 else ''
    print(f"\n  {label}")
    print(f"    Pre mean = {pre_v.mean():.4f}  Post mean = {post_v.mean():.4f}")
    print(f"    Δ = {post_v.mean()-pre_v.mean():+.4f}  [expected: {exp}]")
    print(f"    Welch t-test: t={t_stat:.3f}  p={p_val:.4f} {stars}")

# ══════════════════════════════════════════════════════════════════════════
# 6. 担保代替チャネルの統合検証
#    mortgage登記（第5節）↓ と syn/perf ↑ の同時分析
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}\n6. 代替チャネル間の相関・散布図（参照値）\n{SEP}")
print("  シンジケート・ローン組成とその季節性（Q1集中の確認）")

# Q別の平均組成額
for q in [1, 2, 3, 4]:
    sub_q = samp[samp['qnum'] == q]
    m = sub_q['syn_amount_total'].mean()
    pre_m  = sub_q[sub_q['post']==0]['syn_amount_total'].mean()
    post_m = sub_q[sub_q['post']==1]['syn_amount_total'].mean()
    print(f"  Q{q}: 全期間平均={m:>9,.0f}  前={pre_m:>9,.0f}  後={post_m:>9,.0f}  後/前={post_m/pre_m:.3f}")

# ══════════════════════════════════════════════════════════════════════════
# 7. 季節調整済みイベントスタディ（相対時間）
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}\n7. 季節調整済みイベントスタディ\n{SEP}")

if HAS_SM:
    # ベースライン：2010Q1〜2023Q3（移行期の前まで）
    pre_es = samp[samp['date'] < VERDICT_Q].copy()
    # 季節＋トレンドをフィット → 残差を取る
    def get_residuals(y_series, t_series, q2, q3, q4):
        y_arr = np.array(y_series, dtype=float)
        mask = ~np.isnan(y_arr)
        X = sm.add_constant(np.column_stack([t_series[mask], q2[mask], q3[mask], q4[mask]]))
        res = sm.OLS(y_arr[mask], X).fit()
        X_all = sm.add_constant(np.column_stack([t_series, q2, q3, q4]))
        fitted = X_all @ res.params
        resid_all = y_arr - fitted
        return resid_all, res

    print("\n  アウトカム別 残差（判決後の偏差）:")
    es_targets = [
        ('ln_syn_amount_total',  'ln(Syn amount)'),
        ('ln_syn_deals_total',   'ln(Syn deals)'),
        ('ln_perf_amt_total',    'ln(Perf transferred)'),
    ]
    for var, label in es_targets:
        if var not in samp.columns: continue
        resid, fit = get_residuals(
            samp[var].values,
            samp['t'].values,
            samp['Q2'].values, samp['Q3'].values, samp['Q4'].values
        )
        pre_resid  = resid[samp['post'].values == 0]
        post_resid = resid[samp['post'].values == 1]
        print(f"\n  [{label}]  R²(pre-fit)={fit.rsquared:.3f}")
        print(f"    Pre残差:  mean={np.nanmean(pre_resid):+.4f}  sd={np.nanstd(pre_resid):.4f}")
        print(f"    Post残差: mean={np.nanmean(post_resid):+.4f}  sd={np.nanstd(post_resid):.4f}")
        post_q_list = samp.loc[samp['post']==1, 'quarter'].values
        post_r_list = resid[samp['post'].values == 1]
        for qq, rr in zip(post_q_list, post_r_list):
            print(f"      {qq}: residual = {rr:+.4f}")

# ══════════════════════════════════════════════════════════════════════════
# 8. 図の作成
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}\n8. 図の作成\n{SEP}")

VERDICT_DATE = pd.Timestamp(2023, 11, 27)
PLOT_COLORS  = {'treat': '#1a5f9e', 'ctrl': '#c44b28', 'neutral': '#666666'}

def shade_verdict(ax, xmin=None, xmax=None, label=True):
    ax.axvline(VERDICT_Q, color='#c44b28', lw=1.5, ls='--', alpha=0.8,
               label='Supreme Court ruling (Nov. 2023)' if label else None)

def shade_post(ax):
    ylo, yhi = ax.get_ylim()
    ax.axvspan(POST_START, ax.get_xlim()[1], alpha=0.06, color='#1a5f9e')

# ── Figure 1: シンジケート・ローン組成額の推移 ──────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle('Figure 1. Syndicated Loan Market: Origination and Outstanding', fontsize=13, y=1.01)

ax = axes[0, 0]
ax.bar(samp['date'], samp['syn_amount_total']/10000, color=np.where(samp['post']==1, '#1a5f9e', '#aabbcc'), width=70, alpha=0.8)
ax.set_title('(a) Syndicated Loan Amount (¥ trillion, flow)')
ax.set_ylabel('¥ Trillion')
shade_verdict(ax)
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(3))

ax = axes[0, 1]
ax.bar(samp['date'], samp['syn_deals_total'], color=np.where(samp['post']==1, '#1a5f9e', '#aabbcc'), width=70, alpha=0.8)
ax.set_title('(b) Syndicated Loan Deals (flow)')
ax.set_ylabel('Number of deals')
shade_verdict(ax)
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(3))

ax = axes[1, 0]
ax.plot(samp['date'], samp['syn_amount_term']/10000, color='#1a5f9e', label='Term loan', lw=1.5)
ax.plot(samp['date'], samp['syn_amount_commit']/10000, color='#c44b28', label='Commitment line', lw=1.5)
ax.set_title('(c) Term Loan vs. Commitment Line Amount')
ax.set_ylabel('¥ Trillion')
ax.legend(fontsize=9)
shade_verdict(ax)
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(3))

ax = axes[1, 1]
ax.plot(samp['date'], samp['share_term']*100,   color='#1a5f9e', label='Term share (%)', lw=1.5)
ax.plot(samp['date'], samp['share_commit']*100, color='#c44b28', label='Commitment share (%)', lw=1.5)
ax.set_title('(d) Composition: Term vs. Commitment (% of amount)')
ax.set_ylabel('%')
ax.legend(fontsize=9)
shade_verdict(ax)
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(3))

for ax in axes.flat:
    ax.tick_params(axis='x', rotation=30, labelsize=8)
    ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/figure1_syndicated_loan.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  → {FIG_DIR}/figure1_syndicated_loan.png")

# ── Figure 2: 貸出債権流動化 ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle('Figure 2. Loan Transfer Market: Performing and Non-Performing', fontsize=13, y=1.01)

ax = axes[0, 0]
ax.bar(samp['date'], samp['perf_amt_total'], color=np.where(samp['post']==1, '#2d8a4e', '#99ccaa'), width=70, alpha=0.8)
ax.set_title('(a) Performing Loans Transferred (¥100m, flow)')
ax.set_ylabel('¥100m')
shade_verdict(ax)
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(3))

ax = axes[0, 1]
ax.bar(samp['date'], samp['nperf_amt_total'], color=np.where(samp['post']==1, '#c44b28', '#eebb99'), width=70, alpha=0.8)
ax.set_title('(b) Non-Performing Loans Transferred (¥100m, flow)')
ax.set_ylabel('¥100m')
shade_verdict(ax)
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(3))

ax = axes[1, 0]
ax.plot(samp['date'], samp['perf_amt_sale'], color='#1a5f9e', label='Direct sale', lw=1.5)
ax.plot(samp['date'], samp['perf_amt_trust'], color='#c44b28', label='Trust method', lw=1.5)
ax.plot(samp['date'], samp['perf_amt_participation'], color='#2d8a4e', label='Loan participation', lw=1.5, ls='--')
ax.set_title('(c) Performing Loans: by Transfer Method (¥100m)')
ax.set_ylabel('¥100m')
ax.legend(fontsize=8)
shade_verdict(ax)
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(3))

ax = axes[1, 1]
ax.plot(samp['date'], samp['perf_n_to_fi'],  color='#1a5f9e', label='To financial institutions', lw=1.5)
ax.plot(samp['date'], samp['perf_n_to_spc'], color='#c44b28', label='To SPCs (securitization)', lw=1.5)
ax.set_title('(d) Performing Loans Transferred: by Counterparty (no.)')
ax.set_ylabel('Number of transactions')
ax.legend(fontsize=9)
shade_verdict(ax)
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(3))

for ax in axes.flat:
    ax.tick_params(axis='x', rotation=30, labelsize=8)
    ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/figure2_loan_transfer.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  → {FIG_DIR}/figure2_loan_transfer.png")

# ── Figure 3: 公開 vs. 非公開企業向け（チャネル5） ───────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Figure 3. Channel 5: Syndicated Loans by Borrower Type', fontsize=12)

ax = axes[0]
ax.plot(samp['date'], samp['syn_amt_term_listed']/10000,  color='#1a5f9e', label='Listed/public companies', lw=1.8)
ax.plot(samp['date'], samp['syn_amt_term_private']/10000, color='#c44b28', label='Private companies', lw=1.8)
ax.set_title('(a) Term Loan Amount by Borrower Type (¥ trillion)')
ax.set_ylabel('¥ Trillion')
ax.legend(fontsize=9)
shade_verdict(ax)
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(3))
ax.tick_params(axis='x', rotation=30, labelsize=8)
ax.grid(axis='y', alpha=0.3)

ax = axes[1]
pvt_share = samp['syn_amt_term_private'] / (samp['syn_amt_term_listed'] + samp['syn_amt_term_private'])
ax.plot(samp['date'], pvt_share*100, color='#555555', lw=1.8)
ax.axhline(pvt_share[samp['post']==0].mean()*100, color='#1a5f9e', ls=':', lw=1.5, label='Pre mean')
ax.axhline(pvt_share[samp['post']==1].mean()*100, color='#c44b28', ls=':', lw=1.5, label='Post mean')
ax.set_title('(b) Private Company Share in Term Loan Amount (%)')
ax.set_ylabel('%')
ax.legend(fontsize=9)
shade_verdict(ax)
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(3))
ax.tick_params(axis='x', rotation=30, labelsize=8)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/figure3_borrower_type.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  → {FIG_DIR}/figure3_borrower_type.png")

# ── Figure 4: ITS フィット図（シンジケート・ローン組成額） ─────────────
if HAS_SM:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Figure 4. ITS Fit: Syndicated Loan Amount and Performing Loan Transfer',
                 fontsize=12)

    for ax_i, (var, label) in enumerate([
        ('ln_syn_amount_total', 'ln(Syndicated loan amount)'),
        ('ln_perf_amt_total',   'ln(Performing loans transferred)'),
    ]):
        ax = axes[ax_i]
        sub_fit = samp[samp[var].notna()].copy()
        formula = f"{var} ~ t + Q2 + Q3 + Q4 + post + t_post"
        res_fit = smf.ols(formula, data=sub_fit).fit()
        ax.scatter(sub_fit['date'], sub_fit[var], s=16, alpha=0.7,
                   c=np.where(sub_fit['post']==1, '#c44b28', '#1a5f9e'),
                   label='Observed (pre / post)')
        ax.plot(sub_fit['date'], res_fit.fittedvalues, color='#555555', lw=1.5,
                label='ITS fit')
        # Pre-trendの外挿（反事実）
        cf = sub_fit.copy()
        cf['post'] = 0; cf['t_post'] = 0
        ax.plot(sub_fit['date'], res_fit.predict(cf), color='#1a5f9e',
                lw=1.2, ls='--', alpha=0.7, label='Counterfactual (no ruling)')
        ax.axvline(VERDICT_Q, color='#c44b28', lw=1.5, ls='--', alpha=0.7,
                   label='Ruling (2023Q4)')
        ax.set_title(f'({chr(97+ax_i)}) {label}')
        ax.set_ylabel('Log scale')
        ax.legend(fontsize=8)
        ax.tick_params(axis='x', rotation=30, labelsize=8)
        ax.grid(axis='y', alpha=0.3)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(3))

    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/figure4_its_fit.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {FIG_DIR}/figure4_its_fit.png")

# ── Figure 5: シンジケート Outstanding 残高 ──────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
out_samp = samp.dropna(subset=['syn_outstanding_total'])
ax.fill_between(out_samp['date'],
                out_samp['syn_outstanding_term']/10000,
                label='Term loan outstanding', alpha=0.7, color='#1a5f9e')
ax.fill_between(out_samp['date'],
                out_samp['syn_outstanding_total']/10000,
                out_samp['syn_outstanding_term']/10000,
                label='Commitment line outstanding', alpha=0.7, color='#c44b28')
ax.axvline(VERDICT_Q, color='black', lw=1.5, ls='--', label='Ruling (2023Q4)')
ax.set_title('Figure 5. Syndicated Loan Outstanding Balance (¥ trillion)')
ax.set_ylabel('¥ Trillion')
ax.legend(fontsize=9)
ax.tick_params(axis='x', rotation=30, labelsize=8)
ax.grid(axis='y', alpha=0.3)
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(3))
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/figure5_outstanding.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  → {FIG_DIR}/figure5_outstanding.png")

# ══════════════════════════════════════════════════════════════════════════
# 9. 推計結果サマリー表（論文本文引用用）
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}\n9. ★ 論文 Table 推計結果サマリー ★\n{SEP}")

if HAS_SM and len(results_df) > 0:
    print("""
  Table 6.  ITS Estimates: Substitution Channels in the Loan Market
  ─────────────────────────────────────────────────────────────────────────
  Variable                            CH   δ̂₁(level) [SE]   p   δ̂₂(slope) [SE]  R²  N
  ─────────────────────────────────────────────────────────────────────────""")
    for _, row in results_df.iterrows():
        print(f"  {row['variable']:38s} {row['channel']} "
              f"  {row['delta1']:+.4f}{row['stars1']:3s}[{row['se1']:.4f}] "
              f"  p={row['p1']:.3f}"
              f"  {row['delta2']:+.4f}{row['stars2']:3s}[{row['se2']:.4f}]"
              f"  {row['R2']:.3f}  {row['N']}")

print(f"\n{SEP}\n完了\n{SEP}")
print(f"  パネルデータ  : {OUT_PANEL}")
print(f"  推計結果      : {OUT_RESULT}")
print(f"  図（{FIG_DIR}/）: figure1〜5.png")
