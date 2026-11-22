# Trader Performance vs Market Sentiment

> How does Bitcoin market sentiment (Fear/Greed Index) shape trader behavior and performance on Hyperliquid? Analysis, trader segmentation, a predictive model, behavioral clustering and an interactive dashboard.

![Python](https://img.shields.io/badge/python-3.9%2B-blue) [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://trader-performance-vs-market-sentiment-dashboard.streamlit.app/)

**Live dashboard:** https://trader-performance-vs-market-sentiment-dashboard.streamlit.app/

---

## Overview

| Part | Goal | Where |
|------|------|-------|
| **A-C. Core analysis** | Data prep, sentiment vs performance, trader segmentation | `notebooks/analysis_notebook.ipynb` |
| **Bonus 1** | Predict next-day profitability and PnL volatility | `notebooks/bonus_predictive_model.ipynb` |
| **Bonus 2** | K-Means trader archetypes | `notebooks/bonus_clustering.ipynb` |
| **Bonus 3** | Interactive Streamlit dashboard | `dashboard.py` |

## Project Structure

```
├── data/          # fear_greed_index.csv, historical_data.csv, clustering outputs
├── notebooks/     # analysis + bonus notebooks
├── charts/        # generated charts (PNG)
├── .streamlit/    # dashboard theme (config.toml)
├── dashboard.py
├── requirements.txt
└── BONUS_README.md
```

## Datasets

| File | Description |
|------|-------------|
| `data/fear_greed_index.csv` | Daily Bitcoin Fear/Greed Index (`date`, `value`, `classification`) |
| `data/historical_data.csv` | Hyperliquid trade history (`Account`, `Coin`, `Size USD`, `Side`, `Closed PnL`, `Fee`, `Timestamp IST`) |

---

## Methodology

- **Alignment:** trades and sentiment merged at daily granularity; `Timestamp IST` parsed with `dayfirst=True`; duplicate sentiment dates and unparseable timestamps dropped.
- **Daily metrics per (Account, date):** total `Closed PnL`, trade count, size, win rate, long ratio, fees.
- **Analysis:** PnL and win rate by sentiment (Mann-Whitney U, Fear vs Greed), behavioral shifts, and index-value correlations.
- **Segments:** High/Low Size (median split), Frequent/Infrequent (median split), Consistent Winners/Losers/Inconsistent (Sharpe proxy cut at -0.1 / +0.1).

## Key Insights

1. **Fear days are the most profitable.** Avg daily PnL is ~$5,300 on Fear days vs ~$3,300 on Greed days, while win rates stay stable (~60-68%). The gap comes from size and edge, not win frequency.
2. **Traders lean long in fear, and it pays.** Long ratio is 0.52-0.53 in Fear/Extreme Fear and 0.47 in Neutral/Greed/Extreme Greed.
3. **High-Size traders gain most in Extreme Fear.** Their win rate is 0.66 in Extreme Fear vs 0.46 in Greed.
4. **Frequent traders lead in Fear.** ~$8,000 per day on Fear days vs ~$5,000 on Greed days.
5. **Long bias falls as the index rises** (r = -0.066, p = 0.001), so traders cut long exposure when it has historically paid best.

## Strategy Recommendations

- **Sentiment-gated sizing:** when the index is below 40, High-Size traders should hold or increase size rather than de-risk. Low-Size traders show the opposite pattern and should not apply this.
- **Scale back in Greed:** above 60, reduce size and frequency rather than flipping aggressively short (the long-ratio correlation is real but very small).
- **Frequent traders:** raise trade frequency in Fear regimes and lower it in Greed.

## Bonus Work

Details in [BONUS_README.md](BONUS_README.md).

- **Predictive model:** Logistic Regression, Gradient Boosting and Random Forest (5-fold CV, best by AUC-ROC). Rolling win rate, today's PnL and profit streak are the strongest signals; sentiment adds roughly 5-10% AUC lift.
- **Clustering:** K-Means on 8 behavioral features gives archetypes such as Precision Snipers, High-Frequency Scalpers and Size Traders (`data/trader_archetypes.csv`).
- **Dashboard:** tabs for sentiment vs performance, behavior, segments, time series and a leaderboard, with sentiment, date and segment filters. [Open it live](https://trader-performance-vs-market-sentiment-dashboard.streamlit.app/).

## Charts

`charts/` holds `chart1` to `chart6` (PnL, behavior, scatter, segments, long-bias heatmap, time series) plus the `bonus_*` model and clustering plots.

---

**Tech stack:** Python, pandas, NumPy, SciPy, scikit-learn, Matplotlib, Seaborn, Streamlit, Jupyter
