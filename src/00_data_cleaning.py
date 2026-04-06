#!/usr/bin/env python3
"""
00_data_cleaning.py
===================
"Legal Clarity and Collateral-Setting Behavior: Evidence from Real-Estate Registries"
Koki Arai — JSPS KAKENHI 23K01404

Purpose
-------
Parse the Ministry of Justice e-Stat CSV into analysis-ready panel datasets.

Inputs
------
data/登記統計不動産個数.csv   Raw e-Stat file (Land Registration Statistics,
                              50 Legal Affairs Bureaus, 2007-01 to 2025-12)

Outputs
-------
data/registry_panel.csv       Long panel, all registration categories
data/registry_did.csv         DiD-ready wide panel (treatment/control flags,
                              log-transformed counts, unit-specific IDs)

Run
---
    python src/00_data_cleaning.py
"""

# ============================================================
# 登記統計（不動産個数）データ クリーニング・整形スクリプト
# Google Colab 用
#
# 入力 : 登記統計不動産個数.csv  （e-Stat 形式）
# 出力 :
#   (1) registry_panel.csv     … 分析用ロング形式パネル（全登記種類）
#   (2) registry_did.csv       … DID 用整形済みデータ
#                                 処置群・対照群フラグ付き、主要変数のみ
# ============================================================

# ------ 0. ライブラリ ------
import pandas as pd
import numpy as np
import re
from pathlib import Path

# ------ 1. ファイルパス設定 ------
# Google Colab では Drive をマウントしてパスを書き換えてください
# 例: INPUT_PATH = "/content/drive/MyDrive/data/登記統計不動産個数.csv"

INPUT_PATH  = "登記統計不動産個数.csv"   # ← 適宜変更
OUT_PANEL   = "registry_panel.csv"
OUT_DID     = "registry_did.csv"


# ============================================================
# ステップ 1: 生データ読み込み
#   - e-Stat 形式: 先頭 14 行がメタ情報、15 行目以降がデータ
#   - 行 12（0-indexed: 11）に登記種類名、行 10 にコードが入っている
# ============================================================

with open(INPUT_PATH, encoding="utf-8-sig") as f:
    all_lines = f.readlines()

# 登記種類コード・名称の対応表を取得（先頭 7 列は識別子なので除外）
reg_codes = all_lines[9].strip().split(",")[7:]
reg_names = all_lines[11].strip().split(",")[7:]
code_to_name = dict(zip(reg_codes, reg_names))

# pandas でデータ部分を読み込み（行 14 がカラム行、行 15 以降がデータ）
df_raw = pd.read_csv(
    INPUT_PATH,
    encoding="utf-8-sig",
    skiprows=14,
    header=0,
    low_memory=False,
)

print(f"読み込み完了: {df_raw.shape[0]:,} 行 × {df_raw.shape[1]} 列")


# ============================================================
# ステップ 2: 識別子列の整理
# ============================================================

# 列名を簡潔に付け直す
id_col_map = {
    df_raw.columns[0]: "time_code",
    df_raw.columns[1]: "time_aux",
    df_raw.columns[2]: "yearmonth_jp",   # 例: "2023年11月"
    df_raw.columns[3]: "bureau_code",
    df_raw.columns[4]: "bureau_aux",
    df_raw.columns[5]: "bureau_name",
    df_raw.columns[6]: "item_label",
}
df_raw = df_raw.rename(columns=id_col_map)

# 登記種類列を code から正式名称に付け替え
data_cols_old = list(df_raw.columns[7:])           # "個数【個】", "個数【個】.1", ...
data_cols_new = [f"reg_{c}_{n}" for c, n in zip(reg_codes, reg_names)]
col_rename = dict(zip(data_cols_old, data_cols_new))
df_raw = df_raw.rename(columns=col_rename)


# ============================================================
# ステップ 3: 年月列を date 型に変換
#   "2023年11月" → datetime(2023, 11, 1)
# ============================================================

def parse_yearmonth(s):
    """'2023年11月' → pd.Timestamp('2023-11-01')"""
    m = re.match(r"(\d{4})年(\d{1,2})月", str(s))
    if m:
        return pd.Timestamp(int(m.group(1)), int(m.group(2)), 1)
    return pd.NaT

df_raw["date"] = df_raw["yearmonth_jp"].apply(parse_yearmonth)


# ============================================================
# ステップ 4: 局コードで集計レベルを分類
#   末尾 3 桁が "000" → 管内集計行（複数局の合計）
#   "50000"       → 全国総数
#   それ以外       → 個別法務局
# ============================================================

df_raw["bureau_code"] = df_raw["bureau_code"].astype(str).str.zfill(5)

def classify_bureau(code):
    if code == "50000":
        return "national"
    elif code.endswith("000"):
        return "region"
    else:
        return "bureau"

df_raw["bureau_level"] = df_raw["bureau_code"].apply(classify_bureau)


# ============================================================
# ステップ 5: 数値列のクリーニング
#   - "***" や "…" → NaN
#   - カンマ区切り文字列 → int / float
# ============================================================

reg_cols = [c for c in df_raw.columns if c.startswith("reg_")]

def clean_numeric(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace(",", "")
    if s in ("***", "…", "-", ""):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan

for col in reg_cols:
    df_raw[col] = df_raw[col].apply(clean_numeric)


# ============================================================
# ステップ 6: 不要列を削除してパネル完成
# ============================================================

keep_cols = ["date", "yearmonth_jp", "bureau_code", "bureau_name",
             "bureau_level"] + reg_cols

df_panel = (
    df_raw[keep_cols]
    .sort_values(["bureau_code", "date"])
    .reset_index(drop=True)
)

print(f"\n[パネル] {df_panel.shape[0]:,} 行 × {df_panel.shape[1]} 列")
print(f"  期間: {df_panel['date'].min().strftime('%Y-%m')} ～ "
      f"{df_panel['date'].max().strftime('%Y-%m')}")
print(f"  局数: {df_panel['bureau_name'].nunique()}")


# ============================================================
# ステップ 7: DID 用データセットの作成
#
#  【処置群】
#    - 抵当権の設定       (code 250)  … 論文の主要処置変数
#    - 根抵当権の設定     (code 260)
#    - 賃借権の設定又は賃借物の転貸 (code 200)
#
#  【対照群】
#    - 売買による所有権の移転     (code 150)
#    - 相続による所有権の移転     (code 410)
#    - 地上権の設定             (code 180)
#
#  集計レベル: 個別法務局（bureau_level == "bureau"）のみ使用
#  → 管内集計行は二重計上になるため除外
# ============================================================

# --- 7-1. 変数名の特定 ---
# コードから列名を逆引き
def find_col(code):
    """登記種類コードから列名を返す"""
    matches = [c for c in reg_cols if c.startswith(f"reg_{code}_")]
    return matches[0] if matches else None

var_map = {
    "mortgage":     find_col("250"),   # 抵当権の設定
    "root_mortgage": find_col("260"),  # 根抵当権の設定
    "leasehold":    find_col("200"),   # 賃借権の設定
    "sale":         find_col("150"),   # 売買
    "inheritance":  find_col("410"),   # 相続による所有権の移転（2024年4月～）
    "inheritance_old": find_col("130"),# 相続その他一般承継（～2024年3月）← 旧系列
    "surface_right": find_col("180"),  # 地上権の設定
}

print("\n[DID 変数マッピング]")
for k, v in var_map.items():
    print(f"  {k:20s} ← {v}")

# 対応する列が存在しない場合の警告
missing = [k for k, v in var_map.items() if v is None]
if missing:
    print(f"\n警告: 以下の変数が見つかりません → {missing}")


# --- 7-2. 個別局のみ抽出 ---
df_bureau = df_panel[df_panel["bureau_level"] == "bureau"].copy()

# --- 7-3. 必要列を選択 & リネーム ---
select_cols = ["date", "yearmonth_jp", "bureau_code", "bureau_name"] + \
              [v for v in var_map.values() if v is not None]

df_did = df_bureau[select_cols].copy()

rename_did = {v: k for k, v in var_map.items() if v is not None}
df_did = df_did.rename(columns=rename_did)


# --- 7-4. 時間変数の追加 ---
df_did["year"]  = df_did["date"].dt.year
df_did["month"] = df_did["date"].dt.month

# 最高裁判決: 2023年11月27日
# - 判決前   : date <= 2023-10-01  (11月は移行月として除外)
# - 移行月   : date == 2023-11-01
# - 判決後   : date >= 2023-12-01
df_did["post"] = (df_did["date"] >= pd.Timestamp("2023-12-01")).astype(int)
df_did["transition"] = (df_did["date"] == pd.Timestamp("2023-11-01")).astype(int)

# 相続登記義務化ダミー (2024年4月以降)
df_did["post_inheritance_reform"] = (
    df_did["date"] >= pd.Timestamp("2024-04-01")
).astype(int)

# 判決直後ダミー（判決後かつ義務化前: 2023-12 ～ 2024-03）
df_did["post_verdict_only"] = (
    (df_did["date"] >= pd.Timestamp("2023-12-01")) &
    (df_did["date"] <  pd.Timestamp("2024-04-01"))
).astype(int)


# --- 7-4b. 相続変数の接続処理 ---
# !! 重要 !!
#   「相続による所有権の移転」(code 410) は 2024年4月の相続登記義務化に伴い
#   新設された系列で、それ以前は全欠損。
#   「相続その他一般承継 ～2024年3月」(code 130) が旧系列で 2024年3月まで存在。
#   → 対照群として使う場合は両系列を繋いだ "inheritance_combined" を利用すること。
if "inheritance" in df_did.columns and "inheritance_old" in df_did.columns:
    df_did["inheritance_combined"] = df_did["inheritance"].fillna(
        df_did["inheritance_old"]
    )
    print("\n[注意] inheritance_combined を作成しました。")
    print("  2024年4月以降: code 410（相続による所有権の移転）")
    print("  ～2024年3月  : code 130（相続その他一般承継）で補完")
    n_filled = df_did["inheritance_combined"].notna().sum()
    print(f"  有効観測数: {n_filled:,} / {len(df_did):,}")

# --- 7-5. 対数変換列の追加 ---
count_vars = [k for k in list(var_map.keys()) + ["inheritance_combined"]
              if k in df_did.columns]

for var in count_vars:
    df_did[f"ln_{var}"] = np.log(df_did[var].replace(0, np.nan))


# --- 7-6. ロング形式への変換（オプション）---
# 処置群・対照群フラグを付けたロング形式も別途作成

treatment_vars = ["mortgage", "root_mortgage", "leasehold"]
control_vars   = ["sale", "inheritance_combined", "surface_right"]

id_vars = ["date", "year", "month", "bureau_code", "bureau_name",
           "post", "transition", "post_inheritance_reform",
           "post_verdict_only"]

df_long = pd.melt(
    df_did,
    id_vars=id_vars,
    value_vars=[v for v in count_vars if v in df_did.columns],
    var_name="reg_type",
    value_name="count",
)

df_long["treatment"] = df_long["reg_type"].isin(treatment_vars).astype(int)
df_long["ln_count"]  = np.log(df_long["count"].replace(0, np.nan))

# DID 交差項
df_long["did"] = df_long["post"] * df_long["treatment"]


# ============================================================
# ステップ 8: 基本統計量の表示
# ============================================================

print("\n" + "="*60)
print("基本統計（DID 用データ、個別法務局のみ）")
print("="*60)
print(f"  観測数           : {len(df_did):,}")
print(f"  法務局数         : {df_did['bureau_code'].nunique()}")
print(f"  期間             : {df_did['date'].min().strftime('%Y-%m')} "
      f"～ {df_did['date'].max().strftime('%Y-%m')}")
print(f"  判決前観測数     : {(df_did['post']==0).sum():,}")
print(f"  判決後観測数     : {(df_did['post']==1).sum():,}")
print()

# 欠損値の確認
print("欠損値（処置・対照変数）:")
for var in count_vars:
    if var in df_did.columns:
        n_miss = df_did[var].isna().sum()
        pct = 100 * n_miss / len(df_did)
        print(f"  {var:20s}: {n_miss:5d} ({pct:.1f}%)")

print()
print("変数別記述統計（処置前・処置後）:")
for var in count_vars:
    if var not in df_did.columns:
        continue
    pre  = df_did.loc[df_did["post"]==0, var]
    post = df_did.loc[df_did["post"]==1, var]
    print(f"\n  [{var}]")
    print(f"    判決前 mean={pre.mean():.1f}  median={pre.median():.1f}  "
          f"sd={pre.std():.1f}  N={pre.notna().sum()}")
    print(f"    判決後 mean={post.mean():.1f}  median={post.median():.1f}  "
          f"sd={post.std():.1f}  N={post.notna().sum()}")


# ============================================================
# ステップ 9: CSV 出力
# ============================================================

df_panel.to_csv(OUT_PANEL, index=False, encoding="utf-8-sig")
print(f"\n✓ パネルデータ保存: {OUT_PANEL}  ({len(df_panel):,} 行)")

df_did.to_csv(OUT_DID, index=False, encoding="utf-8-sig")
print(f"✓ DID データ保存:   {OUT_DID}  ({len(df_did):,} 行)")

# ロング形式も保存
OUT_LONG = "registry_did_long.csv"
df_long.to_csv(OUT_LONG, index=False, encoding="utf-8-sig")
print(f"✓ ロング形式保存:   {OUT_LONG}  ({len(df_long):,} 行)")


# ============================================================
# 補足: 変数の説明
# ============================================================
print("""
============================================================
出力ファイルの説明
============================================================

[registry_panel.csv]  全登記種類・全局・全期間のパネルデータ
  - date          : 年月 (YYYY-MM-DD 形式、各月1日)
  - bureau_code   : 法務局コード（5桁）
  - bureau_name   : 法務局名
  - bureau_level  : national / region / bureau
  - reg_XXX_YYY   : 登記種類 XXX（コード）YYY（名称）の個数

[registry_did.csv]  DID 分析用ワイド形式
  主要変数:
  - mortgage           : 抵当権の設定（処置群）
  - root_mortgage      : 根抵当権の設定（処置群）
  - leasehold          : 賃借権の設定（処置群）
  - sale               : 売買による所有権の移転（対照群）
  - inheritance        : 相続による所有権の移転・code 410（2024年4月以降のみ）
  - inheritance_old    : 相続その他一般承継・code 130（～2024年3月）
  - inheritance_combined: 上記2系列を接続した対照群推奨変数 ★DID で使用
  - surface_right      : 地上権の設定（対照群）
  時間ダミー:
  - post                   : 2023年12月以降 = 1
  - transition             : 2023年11月 = 1
  - post_verdict_only      : 2023年12月～2024年3月 = 1（判決後・義務化前）
  - post_inheritance_reform: 2024年4月以降 = 1（相続登記義務化後）
  - ln_*                   : 各変数の自然対数

[registry_did_long.csv]  DID 分析用ロング形式
  - reg_type    : 登記種類
  - count       : 登記個数
  - treatment   : 処置群フラグ (1=処置, 0=対照)
  - did         : post × treatment（DID 交差項）
  - ln_count    : log(count)
============================================================
""")
