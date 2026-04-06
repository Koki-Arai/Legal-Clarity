# Data

## Raw data files (not included in this repository)

Raw data files cannot be redistributed. They must be obtained independently
from the sources described below and placed in this `data/` directory before
running the pipeline.

---

### 1. `登記統計不動産個数.csv` — Land Registration Statistics

| Item | Detail |
|------|--------|
| **Source** | Ministry of Justice of Japan, via e-Stat |
| **URL** | https://www.e-stat.go.jp/stat-search/files?stat_infid=000032284107 |
| **Content** | Monthly counts of property registrations by type and Legal Affairs Bureau |
| **Coverage** | 50 Legal Affairs Bureaus (本局), January 2007 – December 2025 |
| **Format** | e-Stat CSV (UTF-8 BOM; 14-row metadata header; registration types in columns) |
| **Access** | Publicly available; free registration to e-Stat may be required |

Key registration categories used in this study:

| Column (Japanese) | English label | Role |
|---|---|---|
| 抵当権の設定 | Ordinary mortgage | Treatment group |
| 根抵当権の設定 | Blanket (root) mortgage | Treatment group |
| 売買による所有権の移転 | Sale transfer | Control group |
| 相続による所有権の移転 | Inheritance transfer | Control group |

---

### 2. `マクロ指標.csv` — Macro-Economic Controls

| Item | Detail |
|------|--------|
| **Sources** | Bank of Japan (lending rate), Ministry of Internal Affairs (CPI), Tokyo Stock Exchange (Nikkei 225) |
| **Content** | Monthly lending rate, CPI index, Nikkei 225 closing price |
| **Coverage** | January 2007 – December 2025 |
| **Format** | CSV, Japanese era dates (令和/平成), comma-separated values |
| **Access** | All series publicly available from respective statistical portals |

---

### 3. `cdab_1_.csv` — BoJ Loan Market Transaction Survey

| Item | Detail |
|------|--------|
| **Source** | Bank of Japan, "貸出債権等の流動化に関する調査" (Loan Market Transaction Survey) |
| **URL** | https://www.boj.or.jp/statistics/dl/loan/cdab/index.htm |
| **Content** | Quarterly syndicated loan origination, secondary market transactions, borrower composition |
| **Coverage** | 2010 Q1 – 2025 Q4 |
| **Format** | CSV |
| **Access** | Publicly available |

---

## Derived data files (produced by `00_data_cleaning.py`)

Once raw data are obtained, run:

```bash
python src/00_data_cleaning.py
```

This produces:

| File | Description |
|------|-------------|
| `data/registry_panel.csv` | Long panel: bureau × month × category, all registration types |
| `data/registry_did.csv` | DiD-ready wide panel: 50 bureaus × 228 months, log-transformed counts, treatment/control flags |

### `registry_did.csv` — Column Reference

| Column | Type | Description |
|--------|------|-------------|
| `bureau_code` | str | 5-digit bureau identifier (e.g., `53010` = Tokyo) |
| `bureau_name` | str | Bureau name (Japanese) |
| `date` | YYYY-MM-DD | First day of the observation month |
| `year`, `month` | int | Calendar year and month |
| `t` | int | Monotone time index (1 = Jan 2007, 228 = Dec 2025) |
| `post` | int | 1 if month ≥ December 2023 (post-ruling) |
| `excl` | int | 1 if month = November 2023 (transition month, excluded) |
| `mortgage` | float | Raw count, ordinary mortgage registrations |
| `root_mortgage` | float | Raw count, blanket mortgage registrations |
| `sale` | float | Raw count, sale transfer registrations |
| `inheritance_combined` | float | Raw count, inheritance transfer registrations |
| `ln_mortgage` | float | log(mortgage) |
| `ln_root_mortgage` | float | log(root_mortgage) |
| `ln_sale` | float | log(sale) |
| `ln_inheritance_combined` | float | log(inheritance_combined) |
| `treat` | int | 1 for mortgage/root_mortgage, 0 for sale/inheritance |
| `DID` | int | `treat × post` (main DiD dummy) |
| `block` | str | Geographic block (e.g., `kanto`, `kinki`) |
| `size_cat` | str | Bureau size category (`large`, `medium`, `small`) |
