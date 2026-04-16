# Bonus Deliverables — Hyperliquid × Fear/Greed

Three bonus components extending the main analysis notebook.

---

## File Structure

```
project/
├── data/
│   ├── fear_greed_index.csv
│   └── historical_data.csv
├── notebooks/
│   ├── analysis_notebook.ipynb       ← main analysis (Part A/B/C)
│   ├── bonus_predictive_model.ipynb  ← Bonus 1: ML model
│   └── bonus_clustering.ipynb        ← Bonus 2: Clustering
├── dashboard.py                       ← Bonus 3: Streamlit app
├── charts/                            ← All saved charts
└── requirements_bonus.txt
```

---

## Setup

```bash
pip install -r requirements_bonus.txt
```

---

## Bonus 1 — Predictive Model (`bonus_predictive_model.ipynb`)

**Goal:** Predict next-day trader profitability and PnL volatility.

### What it does
- Engineers lag/rolling features: today's PnL, win rate, sentiment, 3-day rolling averages, profit streaks
- Trains 3 classifiers (Logistic Regression, Gradient Boosting, Random Forest) with 5-fold CV
- Selects best model by AUC-ROC
- Evaluates accuracy broken down by sentiment regime
- Also trains a regression model to predict next-day |PnL| magnitude

### Charts generated
| File | Description |
|------|-------------|
| `bonus_clf_model_comparison.png` | CV boxplots + confusion matrix |
| `bonus_feature_importance.png` | Feature importance (sentiment vs behavior vs rolling) |
| `bonus_accuracy_by_sentiment.png` | Model accuracy per sentiment regime |
| `bonus_regression_model.png` | Actual vs predicted volatility + regression feature importance |
| `bonus_trader_sentiment_heatmap.png` | Per-trader predicted win probability across regimes |

### Key finding
Behavioral momentum features (rolling win rate, today's PnL, profit streak) are the strongest predictors. Sentiment adds ~5–10% AUC lift on top of behavior alone — meaningful but not dominant.

---

## Bonus 2 — Clustering (`bonus_clustering.ipynb`)

**Goal:** Discover natural trader archetypes using K-Means on behavioral features.

### Features used for clustering
- `avg_trades_per_day` — trading intensity
- `avg_size_usd` — position sizing
- `mean_win_rate` — edge quality
- `mean_long_ratio` — directional bias
- `sharpe_proxy` — risk-adjusted returns
- `mean_volume_per_day` — capital deployment
- `wr_sentiment_delta` — how much win rate shifts from fear → greed
- `tc_sentiment_delta` — how much trade count shifts from fear → greed

### Charts generated
| File | Description |
|------|-------------|
| `bonus_cluster_k_selection.png` | Elbow + silhouette to find optimal K |
| `bonus_cluster_pca.png` | 2D PCA projection of all traders colored by archetype |
| `bonus_cluster_radar.png` | Radar/spider chart per archetype |
| `bonus_archetype_sentiment_heatmap.png` | Avg PnL per archetype × sentiment |

### Archetype names (auto-assigned by PnL rank)
| Archetype | Signature | Best Regime |
|-----------|-----------|-------------|
| 🏆 Precision Snipers | High Sharpe, large sizes | Fear |
| ⚡ High-Frequency Scalpers | Very high trade count | Neutral/Fear |
| 🐘 Size Traders | Largest positions | Fear |
| 🎲 Sentiment Followers | WR highly sentiment-sensitive | Extreme Greed |
| 🕊️ Passive Accumulators | Low frequency, steady | Neutral |

---

## Bonus 3 — Streamlit Dashboard (`dashboard.py`)

**Goal:** Interactive exploration of all findings.

### Run

```bash
# From project root
streamlit run dashboard.py
```

Opens at `http://localhost:8501`

### Features

| Tab | Contents |
|-----|----------|
| 📈 Sentiment vs Performance | Avg PnL, win rate, long bias by regime + summary table |
| 🔀 Behavior Analysis | Trade count, position size, scatter with trendline |
| 👥 Trader Segments | Grouped bar + heatmap for Size / Frequency / Consistency segments |
| 📅 Time Series | PnL, win rate, Fear/Greed index over time (3-panel) |
| 🏆 Leaderboard | Sortable trader rankings with gradient-styled table |

**Sidebar filters:**
- Sentiment regime multi-select
- Date range picker
- Segment view toggle

---

## Strategy Recommendations (from clustering + model)

1. **Know your archetype first.** Precision Snipers and Size Traders should lean into Fear days; Sentiment Followers are the exception that can capitalize on Extreme Greed.

2. **Use the model as a daily gate.** If predicted P(profitable) < 40% for your archetype in the current sentiment regime → reduce size by 30–50% or sit out.

3. **Rolling win rate is your leading indicator.** A 3-day rolling win rate below 70% (from the ~85% mean) is a reliable early warning to reduce exposure regardless of the sentiment reading.
