# Legal Clarity and Collateral-Setting Behavior: Evidence from Real-Estate Registries

**Koki Arai**  
Professor of Economics, Faculty of Business Studies, Kyoritsu Women's University  
JSPS KAKENHI Grant Number 23K01404

---

## Overview

This repository contains the replication code for:

> Arai, K. (2026). "Legal Clarity and Collateral-Setting Behavior: Evidence from
> Real-Estate Registries." *European Journal of Law and Economics* (under review).

### Research question

Does a single appellate ruling — one that amends no statute — generate measurable,
real-time adjustments in a nation's secured-credit market?

### Setting

On **27 November 2023**, the Supreme Court of Japan held that a tenant cannot
assert set-off based on claims acquired *after* mortgage registration against a
mortgagee exercising subrogation over rent receivables. No statute was amended.
Within months, nationwide ordinary mortgage registration activity fell approximately
**8–14% below pre-treatment trend** — a cumulative gap of ~280,000–300,000
registrations over 25 months.

### Method

Detrended two-way fixed effects (TWFE) with unit-specific linear trends
(Autor, Donohue & Schwab 2003), across 50 Legal Affairs Bureaus × 228 months
(January 2007 – December 2025). Identification is supported by:

- Callaway–Sant'Anna (2021) doubly-robust ATT: −10.1%
- First-difference DiD (verdict-month impact): −13.9%
- Rambachan–Roth (2023) sensitivity: sign robust to M ≥ 1.1× pre-period SD

---

## Repository structure

```
.
├── src/
│   ├── 00_data_cleaning.py        Parse e-Stat CSV → analysis panel
│   ├── 01_descriptive_stats.py    Table 1 and Section 5.1 numbers
│   ├── 02_main_estimation.py      Main DiD: Blocks A–G (Tables 2, 4, 6)
│   ├── 03_robustness.py           Robustness: B1–B6 + Rambachan–Roth
│   ├── 04_loan_market.py          Section 6: loan market substitution (ITS)
│   └── 05_figures.py              Figures 1–5 (publication-quality PDF/PNG)
│
├── data/
│   ├── README.md                  Data sources and column reference
│   ├── 登記統計不動産個数.csv      ← NOT included; see data/README.md
│   ├── マクロ指標.csv              ← NOT included; see data/README.md
│   └── cdab_1_.csv                ← NOT included; see data/README.md
│
├── results/
│   ├── README.md                  Output file descriptions
│   ├── *.csv                      All estimation output CSVs
│   └── figures/                   PDF and PNG figures (regenerated)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Replication

### 1. Clone and install dependencies

```bash
git clone https://github.com/Koki-Arai/legal-clarity-collateral.git
cd legal-clarity-collateral
pip install -r requirements.txt
```

### 2. Obtain raw data

Download the three raw data files and place them in `data/`.
See [`data/README.md`](data/README.md) for exact URLs and format details.

| File | Source |
|------|--------|
| `data/登記統計不動産個数.csv` | Ministry of Justice / e-Stat |
| `data/マクロ指標.csv` | BoJ, MIC, TSE (aggregated) |
| `data/cdab_1_.csv` | Bank of Japan (Loan Market Survey) |

### 3. Run the pipeline in order

```bash
python src/00_data_cleaning.py      # → data/registry_did.csv
python src/01_descriptive_stats.py  # → results/descriptive_stats.csv
python src/02_main_estimation.py    # → results/estimation_results.csv, event studies
python src/03_robustness.py         # → results/step*.csv, RR sensitivity
python src/04_loan_market.py        # → results/section6_*.csv
python src/05_figures.py            # → results/figures/figure*.pdf
```

Each script is self-contained and can be run independently if its upstream
inputs are already present in `results/`.

### Google Colab

All scripts are compatible with Google Colab. Upload data files to your Drive,
mount Drive, and adjust the `INPUT_PATH` constants at the top of each script.

---

## Key results

| Estimator | δ̂ | SE | Source |
|-----------|------|-----|--------|
| Detrended TWFE (primary) | −0.087*** | 0.016 | Block B, `02_main_estimation.py` |
| + Macro controls | −0.087*** | 0.016 | Block B4, `03_robustness.py` |
| Two-treatment: verdict | −0.187*** | 0.023 | Block C, `02_main_estimation.py` |
| Two-treatment: reform | −0.067*** | 0.016 | Block C, `02_main_estimation.py` |
| First-difference DiD | −0.139*** | 0.034 | Block E, `02_main_estimation.py` |
| Callaway–Sant'Anna ATT | −0.101 | ≈0.014 | Block F, `02_main_estimation.py` |
| Ordinary mortgage vs. sale only | −0.085*** | 0.010 | Block B6s, `03_robustness.py` |
| Root mortgage vs. sale only | +0.024 | 0.020 | Block B6s, `03_robustness.py` |

N = 45,400 (core sample); 50 bureaus × 4 categories × 227 months  
Standard errors clustered at Legal Affairs Bureau level. *** p < 0.01.

---

## Rambachan–Roth sensitivity (Section 4.8.1)

Under the relative-magnitudes (RM) restriction with M = 1.0
(post-period violation ≤ pre-period SD = 0.079 log points):

- TWFE identified set: **[−0.166, −0.009]** (entirely negative)
- Sign reversed only at M* = **1.1** (violation > observed pre-period SD)
- FD estimate sign reversed only at M* = **1.8**

---

## Software versions

| Package | Version tested |
|---------|---------------|
| Python | 3.11.x |
| numpy | 1.26.x |
| pandas | 2.1.x |
| scipy | 1.11.x |
| linearmodels | 5.3.x |
| csdid | 0.1.4 |
| statsmodels | 0.14.x |
| matplotlib | 3.8.x |

---

## Citation

```bibtex
@article{arai2026legal,
  author  = {Arai, Koki},
  title   = {Legal Clarity and Collateral-Setting Behavior:
             Evidence from Real-Estate Registries},
  journal = {European Journal of Law and Economics},
  year    = {2026},
  note    = {Under review}
}
```

---

## Funding and disclosures

This research was supported by JSPS KAKENHI Grant Number **23K01404**.  
The author declares no competing interests.

---

## License

Code: [MIT License](LICENSE)  
Data: Raw data files are subject to the terms of their respective sources
(Ministry of Justice / e-Stat terms of use; Bank of Japan data policy).
