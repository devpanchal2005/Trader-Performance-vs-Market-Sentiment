# Hyperliquid × Fear/Greed Sentiment Analysis

> Analyzing how Bitcoin market sentiment drives trader behavior and performance on Hyperliquid.

---

## Project Structure

```
hyperliquid-sentiment/
├── data/
│   ├── fear_greed_index.csv        ← Bitcoin Fear/Greed Index
│   └── historical_data.csv         ← Hyperliquid trade history
├── notebooks/
│   └── analysis.ipynb              ← Main analysis notebook
├── charts/                         ← Auto-generated output charts
│   ├── chart1_pnl_by_sentiment.png
│   ├── chart2_behavior_by_sentiment.png
│   ├── chart3_sentiment_scatter.png
│   ├── chart4_segment_x_sentiment.png
│   ├── chart5_longbias_heatmap.png
│   └── chart6_timeseries.png
└── README.md
```

---

## Setup & How to Run

### 1. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scipy jupyter
```

### 2. Place data files

Put your two CSV files in the `data/` folder:
- `data/fear_greed_index.csv`
- `data/historical_data.csv`

### 3. Run the notebook

```bash
cd notebooks
jupyter notebook analysis.ipynb
```

Run all cells top-to-bottom (`Kernel → Restart & Run All`).  
Charts save automatically to `../charts/`.

---

## Methodology

### Data Preparation (Part A)

| Step | Detail |
|------|--------|
| **Timestamp alignment** | `Timestamp IST` (dd-mm-yyyy HH:MM) parsed with `dayfirst=True`; epoch-ms column used as fallback. Both datasets merged on date at daily granularity. |
| **Sentiment encoding** | 5-class ordinal (Extreme Fear → 0, Extreme Greed → 4) + binary Fear/Greed/Neutral label. |
| **Daily metrics** | For each (Account, date) pair: sum of `Closed PnL`, trade count, avg/total size USD, win rate (closes with PnL > 0 / total closes), long ratio, total fee. |
| **Trader-level stats** | Aggregated across all days: total PnL, Sharpe proxy (mean daily PnL / std), avg trades/day, used for segmentation. |
| **Missing/duplicates** | Duplicate dates in Fear/Greed index dropped (keep first); rows with unparseable timestamps dropped from trades. |

### Analysis (Part B)

1. **PnL & win rate by sentiment** — Distribution plots + Mann-Whitney U test comparing Fear vs Greed days.
2. **Behavioral shifts** — Trade frequency, position size, long/short ratio aggregated by sentiment class.
3. **Continuous correlation** — Scatter of sentiment index value vs key metrics with linear regression.

### Segmentation (Part C)

| Segment | Split criterion |
|---------|----------------|
| **High / Low Size** | Median split on trader's average trade size in USD (leverage proxy) |
| **Frequent / Infrequent** | Median split on average trades per active day |
| **Consistent Winners / Losers / Inconsistent** | Sharpe proxy cut at –0.1 / +0.1 |

---

## Key Insights

### Insight 1 — Fear Days Boost Profitability, Not Suppress It
Average daily PnL is measurably **higher** on Fear and Extreme Fear days than on Greed days. Fear generates the highest avg daily PnL (~$5,300), while Greed is the weakest regime (~$3,300). Win rates are relatively stable across regimes (~60–68%), so the gap is driven by trade size and edge quality, not win frequency.

### Insight 2 — Traders Go More Long During Fear (Not Greed) — and It Pays Off
The long ratio peaks during Fear periods (Extreme Fear: 0.53, Fear: 0.52) and drops to 0.47 across Neutral, Greed, and Extreme Greed alike. Far from being punished, this long bias during fear is directionally correct — Fear days are the highest-PnL regime. The crowd is not chasing momentum late; it is actually positioned well when it feels worst.

### Insight 3 — High-Size Traders Thrive in Extreme Fear, Not Struggle
High-Size traders show the sharpest PnL delta between Fear and Greed days, but in the direction opposite to what intuition suggests. Their win rate is highest under Extreme Fear (0.66) and lowest under Greed (0.46). Larger positions amplify gains — not just drawdowns — when sentiment is fearful and volatility creates opportunity.

### Insight 4 — Frequent Traders Lead During Fear, Not Across All Regimes Equally
Frequent traders do not maintain flat PnL across sentiment regimes. Chart 4 shows they earn substantially more on Fear days (~$8,000) than Greed (~$5,000) or Neutral (~$4,000) days. Their advantage is most pronounced precisely when sentiment is most negative — likely because higher trade throughput captures more of the fear-driven price dislocations.

### Insight 5 — Long Bias Falls as the Fear/Greed Index Rises
The time-series and scatter charts show the long ratio and the Fear/Greed index move in **opposite** directions (r=-0.066, p=0.001). As the index rises toward Greed, traders rotate net short (0.47). Since Fear days are the highest-PnL regime, this means traders are reducing long exposure precisely when long exposure has historically been most rewarded — a systematic behavioural misalignment.

---

## Strategy Recommendations

### Strategy 1 — Sentiment-Gated Size Rule
> **"When the Fear/Greed index drops below 40 (Fear zone), High-Size traders should maintain or increase position size — not reduce it."**

- Evidence: Chart 4 shows High-Size segment PnL is most positive on Fear days (~$9,600) and weakest on Greed days (~$3,100). Chart 5 confirms their win rate peaks at 0.66 during Extreme Fear.
- Rationale: Fear-driven volatility creates pricing dislocations that larger positions are well-placed to exploit. The instinct to de-risk during fear directly works against this edge.
- Note: This applies to High-Size traders only. Low-Size traders show the opposite pattern (better performance on Greed days, ~$4,400 vs ~$2,400) and should not apply this rule.

### Strategy 2 — Reduce Long Exposure During Greed, Not Fear
> **"When the Fear/Greed index exceeds 60 (Greed zone), reduce overall activity and resist the urge to go net short — instead, simply scale back size and frequency."**

- Evidence: Chart 2 shows experienced traders already pull back organically during Greed — trade frequency falls from 133.75/day (Extreme Fear) to 76.03/day (Extreme Greed), and avg trade size drops from $8,976 to $5,372. Chart 1 confirms aggregate PnL is weakest in this regime.
- Rationale: Greed is a low-opportunity environment. The correct response is reduced activity, not contrarian shorting — Chart 3 shows the long ratio correlation with the index is real but very small (r=-0.066), so flipping aggressively short is not supported by the data.

### Rule of Thumb for Frequent Traders
> **"Increase trade frequency during Fear regimes (index below 40) and reduce it during Greed — do not treat all sentiment regimes as equivalent."**

- Evidence: Chart 4 shows Frequent traders earn ~$8,000 on Fear days versus ~$5,000 on Greed days. The gap is substantial and consistent. Frequent trading during Greed still outperforms Infrequent traders (~$3,500), but the Fear regime is where high-frequency throughput generates its strongest edge.

---

## Charts Overview

| Chart | What it shows |
|-------|--------------|
| `chart1_pnl_by_sentiment` | Violin + bar: PnL distribution and win rate per sentiment class |
| `chart2_behavior_by_sentiment` | 2×2 grid: trade count, volume, size, long ratio by sentiment |
| `chart3_sentiment_scatter` | Regression scatter: index value vs PnL, trade count, long ratio |
| `chart4_segment_x_sentiment` | Grouped bars: segment PnL under Fear / Neutral / Greed |
| `chart5_longbias_heatmap` | Long ratio by sentiment + win-rate heatmap (segment × sentiment) |
| `chart6_timeseries` | Time-series: daily aggregate PnL (colored by sentiment) vs index |
