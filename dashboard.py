"""
Hyperliquid × Fear/Greed — Interactive Dashboard
Run: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hyperliquid x Fear/Greed",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark theme CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stMetric { background-color: #1a1d2e; border-radius: 8px; padding: 12px; }
    .stMetric label { color: #a0a0b0 !important; font-size: 0.8rem !important; }
    .stMetric [data-testid="stMetricValue"] { color: #e0e0e0 !important; font-size: 1.6rem !important; }
    .stMetric [data-testid="stMetricDelta"] { font-size: 0.85rem !important; }
    h1, h2, h3 { color: #e0e0e0 !important; }
    .sidebar .sidebar-content { background-color: #1a1d2e; }
    div[data-testid="stSidebarContent"] { background-color: #1a1d2e; }
    .stSelectbox label, .stMultiSelect label, .stSlider label { color: #a0a0b0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Matplotlib dark theme ─────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1d2e",
    "axes.edgecolor":   "#3a3d52",
    "axes.labelcolor":  "#e0e0e0",
    "xtick.color":      "#a0a0b0",
    "ytick.color":      "#a0a0b0",
    "text.color":       "#e0e0e0",
    "grid.color":       "#2a2d3e",
    "grid.alpha":       0.5,
    "font.family":      "DejaVu Sans",
    "font.size":        10,
    "axes.titlesize":   12,
    "axes.titleweight": "bold",
})

SENT_COLORS = {
    "Extreme Fear": "#e74c3c",
    "Fear":         "#e67e22",
    "Neutral":      "#95a5a6",
    "Greed":        "#2ecc71",
    "Extreme Greed":"#27ae60",
}
SENT_ORDER = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
CLUSTER_COLORS = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]

# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Loading data…")
def load_data():
    base = Path(__file__).parent

    # ── Fear/Greed ────────────────────────────────────────────────────────────
    fg = pd.read_csv(base / "data" / "fear_greed_index.csv")
    fg["date"] = pd.to_datetime(fg["timestamp"], unit="s").dt.normalize()
    fg = fg.sort_values("date").drop_duplicates("date", keep="last")
    fg["sentiment_score"] = pd.Categorical(
        fg["classification"], categories=SENT_ORDER, ordered=True
    ).codes

    # ── Trades ────────────────────────────────────────────────────────────────
    hl = pd.read_csv(base / "data" / "historical_data.csv")
    hl["date"]     = pd.to_datetime(hl["Timestamp IST"], dayfirst=True).dt.normalize()
    hl["is_long"]  = (hl["Direction"].str.strip().str.lower() == "buy").astype(int)
    hl["is_win"]   = (hl["Closed PnL"] > 0).astype(int)
    hl["is_close"] = (hl["Closed PnL"] != 0).astype(int)

    daily = (
        hl.groupby(["Account", "date"])
        .agg(
            daily_pnl        = ("Closed PnL",  "sum"),
            trade_count      = ("Trade ID",    "count"),
            avg_size_usd     = ("Size USD",    "mean"),
            total_volume_usd = ("Size USD",    "sum"),
            wins             = ("is_win",      "sum"),
            closes           = ("is_close",    "sum"),
            long_trades      = ("is_long",     "sum"),
            total_fee        = ("Fee",         "sum"),
        )
        .reset_index()
    )
    daily["win_rate"]     = np.where(daily["closes"] > 0, daily["wins"] / daily["closes"], np.nan)
    daily["long_ratio"]   = daily["long_trades"] / daily["trade_count"]
    daily["pnl_positive"] = (daily["daily_pnl"] > 0).astype(int)

    daily = daily.merge(
        fg[["date", "value", "classification", "sentiment_score"]],
        on="date", how="inner"
    )

    # ── Trader summary ────────────────────────────────────────────────────────
    trader_stats = (
        daily.groupby("Account")
        .agg(
            total_pnl           = ("daily_pnl",        "sum"),
            avg_daily_pnl       = ("daily_pnl",        "mean"),
            pnl_std             = ("daily_pnl",        "std"),
            total_trades        = ("trade_count",      "sum"),
            trading_days        = ("date",             "nunique"),
            avg_size_usd        = ("avg_size_usd",     "mean"),
            mean_win_rate       = ("win_rate",         "mean"),
            mean_long_ratio     = ("long_ratio",       "mean"),
            mean_volume_per_day = ("total_volume_usd", "mean"),
        )
        .reset_index()
    )
    trader_stats["avg_trades_per_day"] = (
        trader_stats["total_trades"] / trader_stats["trading_days"]
    )
    trader_stats["sharpe_proxy"] = (
        trader_stats["avg_daily_pnl"] / (trader_stats["pnl_std"] + 1e-9)
    )

    # ── Segments ──────────────────────────────────────────────────────────────
    size_med = trader_stats["avg_size_usd"].median()
    freq_med = trader_stats["avg_trades_per_day"].median()
    pnl_med  = trader_stats["total_pnl"].median()

    trader_stats["size_segment"] = np.where(
        trader_stats["avg_size_usd"] >= size_med, "High Size", "Low Size"
    )
    trader_stats["freq_segment"] = np.where(
        trader_stats["avg_trades_per_day"] >= freq_med, "Frequent", "Infrequent"
    )
    trader_stats["consistency_segment"] = pd.cut(
        trader_stats["total_pnl"],
        bins=[-np.inf, 0, pnl_med, np.inf],
        labels=["Consistent Losers", "Inconsistent", "Consistent Winners"],
    ).astype(str)

    # ── Merge segments back to daily ──────────────────────────────────────────
    daily_seg = daily.merge(
        trader_stats[["Account", "size_segment", "freq_segment", "consistency_segment"]],
        on="Account", how="left",
    )

    # ── Sentiment-level aggregates ────────────────────────────────────────────
    sent_agg = (
        daily.groupby("classification")
        .agg(
            avg_pnl    = ("daily_pnl",        "mean"),
            median_pnl = ("daily_pnl",        "median"),
            pnl_std    = ("daily_pnl",        "std"),
            win_rate   = ("pnl_positive",     "mean"),
            avg_trades = ("trade_count",      "mean"),
            avg_vol    = ("total_volume_usd", "mean"),
            avg_size   = ("avg_size_usd",     "mean"),
            long_ratio = ("long_ratio",       "mean"),
            n_obs      = ("daily_pnl",        "count"),
        )
        .reindex([s for s in SENT_ORDER if s in daily["classification"].unique()])
        .reset_index()
    )

    return fg, hl, daily, daily_seg, trader_stats, sent_agg


fg, hl, daily, daily_seg, trader_stats, sent_agg = load_data()
all_traders = sorted(daily["Account"].unique().tolist())

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📊 Filters")

    # Sentiment filter
    selected_sentiments = st.multiselect(
        "Sentiment Regimes",
        options=SENT_ORDER,
        default=SENT_ORDER,
    )

    # Date range
    min_date = daily["date"].min().date()
    max_date = daily["date"].max().date()
    date_range = st.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    # Segment filter
    st.markdown("---")
    seg_type = st.selectbox(
        "Segment View",
        ["Size (High/Low)", "Frequency (Frequent/Infrequent)", "Consistency"],
    )

    st.markdown("---")
    st.markdown(
        "<div style='color:#666; font-size:0.75rem'>Data: Hyperliquid trades + "
        "Alternative.me Fear/Greed Index</div>",
        unsafe_allow_html=True,
    )

# ── Apply filters ─────────────────────────────────────────────────────────────
if len(date_range) == 2:
    d_start, d_end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
else:
    d_start, d_end = daily["date"].min(), daily["date"].max()

mask = (
    daily["classification"].isin(selected_sentiments)
    & daily["date"].between(d_start, d_end)
)
daily_f = daily[mask].copy()
sent_agg_f = sent_agg[sent_agg["classification"].isin(selected_sentiments)]

# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.title("📊 Hyperliquid × Fear/Greed Dashboard")
st.markdown(
    f"**{daily_f['Account'].nunique()} traders** | "
    f"**{len(daily_f):,} trader-days** | "
    f"**{daily_f['date'].nunique()} unique dates** | "
    f"Range: {d_start.date()} → {d_end.date()}"
)
st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# KPI ROW
# ═══════════════════════════════════════════════════════════════════════════════
col1, col2, col3, col4, col5 = st.columns(5)

total_pnl    = daily_f["daily_pnl"].sum()
avg_pnl      = daily_f["daily_pnl"].mean()
overall_wr   = daily_f["pnl_positive"].mean()
avg_trades   = daily_f["trade_count"].mean()
avg_size     = daily_f["avg_size_usd"].mean()

col1.metric("Total PnL", f"${total_pnl:,.0f}")
col2.metric("Avg Daily PnL", f"${avg_pnl:,.0f}")
col3.metric("Win Rate", f"{overall_wr:.1%}")
col4.metric("Avg Trades/Day", f"{avg_trades:.0f}")
col5.metric("Avg Trade Size", f"${avg_size:,.0f}")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Sentiment vs Performance",
    "🔀 Behavior Analysis",
    "👥 Trader Segments",
    "📅 Time Series",
    "🏆 Leaderboard",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SENTIMENT VS PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("How does sentiment affect trader performance?")
    c1, c2 = st.columns(2)

    with c1:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        colors = [SENT_COLORS.get(s, "#aaa") for s in sent_agg_f["classification"]]
        bars = ax.bar(sent_agg_f["classification"], sent_agg_f["avg_pnl"],
                      color=colors, edgecolor="#3a3d52", width=0.6)
        for bar, val in zip(bars, sent_agg_f["avg_pnl"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                    f"${val:,.0f}", ha="center", fontsize=8)
        ax.set_title("Avg Daily PnL by Sentiment")
        ax.set_ylabel("Avg PnL ($)")
        ax.grid(True, axis="y", alpha=0.4)
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with c2:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        x = np.arange(len(sent_agg_f))
        w = 0.35
        bars1 = ax.bar(x - w/2, sent_agg_f["win_rate"],
                        color=[SENT_COLORS.get(s, "#aaa") for s in sent_agg_f["classification"]],
                        width=w, label="Win Rate", alpha=0.9)
        ax2b = ax.twinx()
        ax2b.tick_params(colors="#e0e0e0")
        ax2b.spines["right"].set_color("#3a3d52")
        ax2b.plot(x, sent_agg_f["long_ratio"], "w--o", linewidth=2,
                  markersize=6, label="Long Ratio")
        ax2b.axhline(0.5, color="#e74c3c", linestyle=":", alpha=0.6)
        ax2b.set_ylabel("Long Ratio", color="#e0e0e0")
        ax.set_xticks(x)
        ax.set_xticklabels(sent_agg_f["classification"], rotation=15, ha="right")
        ax.set_ylabel("Win Rate")
        ax.set_ylim(0, 1)
        ax.set_title("Win Rate & Long Bias by Sentiment")
        ax.legend(loc="upper left")
        ax2b.legend(loc="upper right")
        ax.grid(True, axis="y", alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Stats table
    st.markdown("#### Summary Table")
    display_cols = ["classification", "avg_pnl", "median_pnl", "win_rate",
                    "avg_trades", "avg_vol", "avg_size", "long_ratio", "n_obs"]
    tbl = sent_agg_f[display_cols].copy()
    tbl.columns = ["Sentiment", "Avg PnL", "Median PnL", "Win Rate",
                   "Avg Trades/Day", "Avg Volume", "Avg Size", "Long Ratio", "N Obs"]
    st.dataframe(
        tbl.style.format({
            "Avg PnL": "${:,.0f}", "Median PnL": "${:,.0f}",
            "Win Rate": "{:.1%}", "Avg Trades/Day": "{:.0f}",
            "Avg Volume": "${:,.0f}", "Avg Size": "${:,.0f}",
            "Long Ratio": "{:.3f}",
        }).background_gradient(subset=["Avg PnL", "Win Rate"], cmap="RdYlGn"),
        use_container_width=True,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BEHAVIOR ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("How does trader behavior change with sentiment?")
    c1, c2 = st.columns(2)

    with c1:
        fig, axes = plt.subplots(2, 1, figsize=(7, 7))
        # Trade count
        ax = axes[0]
        bars = ax.bar(sent_agg_f["classification"], sent_agg_f["avg_trades"],
                      color=[SENT_COLORS.get(s, "#aaa") for s in sent_agg_f["classification"]],
                      edgecolor="#3a3d52", width=0.6)
        for bar, val in zip(bars, sent_agg_f["avg_trades"]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:.0f}", ha="center", fontsize=8)
        ax.set_title("Avg Trades Per Day")
        ax.set_ylabel("Trades")
        ax.grid(True, axis="y", alpha=0.4)
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

        # Avg size
        ax2 = axes[1]
        bars2 = ax2.bar(sent_agg_f["classification"], sent_agg_f["avg_size"],
                         color=[SENT_COLORS.get(s, "#aaa") for s in sent_agg_f["classification"]],
                         edgecolor="#3a3d52", width=0.6)
        for bar, val in zip(bars2, sent_agg_f["avg_size"]):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                     f"${val:,.0f}", ha="center", fontsize=8)
        ax2.set_title("Avg Trade Size (USD)")
        ax2.set_ylabel("USD")
        ax2.grid(True, axis="y", alpha=0.4)
        plt.setp(ax2.get_xticklabels(), rotation=15, ha="right")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with c2:
        fig, ax = plt.subplots(figsize=(7, 7))
        # Scatter: sentiment score vs daily pnl
        subset = daily_f[daily_f["daily_pnl"].abs() < daily_f["daily_pnl"].abs().quantile(0.97)]
        for sent, grp in subset.groupby("classification"):
            ax.scatter(grp["value"], grp["daily_pnl"],
                       color=SENT_COLORS.get(sent, "#aaa"),
                       alpha=0.35, s=12, label=sent)
        from scipy import stats as sp_stats
        m, b, r, p, _ = sp_stats.linregress(subset["value"], subset["daily_pnl"])
        xr = np.linspace(subset["value"].min(), subset["value"].max(), 100)
        ax.plot(xr, m * xr + b, "w-", linewidth=1.5,
                label=f"Trend (r={r:.3f}, p={p:.3f})")
        ax.axhline(0, color="#666", linestyle="--", alpha=0.5)
        ax.set_xlabel("Fear/Greed Index (0=Extreme Fear, 100=Extreme Greed)")
        ax.set_ylabel("Daily PnL ($)")
        ax.set_title("Sentiment Score vs Daily PnL")
        ax.legend(fontsize=8, markerscale=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — TRADER SEGMENTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Segment performance vs sentiment")

    seg_col_map = {
        "Size (High/Low)": "size_segment",
        "Frequency (Frequent/Infrequent)": "freq_segment",
        "Consistency": "consistency_segment",
    }
    seg_col = seg_col_map[seg_type]

    # Merge segment into filtered daily
    daily_seg_f = daily_f.merge(
        trader_stats[["Account", seg_col]], on="Account", how="left"
    )
    seg_sent = (
        daily_seg_f.groupby([seg_col, "classification"])["daily_pnl"]
        .mean()
        .unstack("classification")
        .reindex(columns=[s for s in SENT_ORDER if s in daily_f["classification"].unique()])
    )

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(8, 5))
        seg_list = seg_sent.index.tolist()
        n_groups = len(seg_list)
        n_sents  = len(seg_sent.columns)
        x = np.arange(n_groups)
        width = 0.8 / max(n_sents, 1)
        for i, sent in enumerate(seg_sent.columns):
            offset = (i - n_sents / 2) * width + width / 2
            bars = ax.bar(x + offset, seg_sent[sent], width=width * 0.9,
                          color=SENT_COLORS.get(sent, "#aaa"),
                          label=sent, edgecolor="#3a3d52")
        ax.set_xticks(x)
        ax.set_xticklabels(seg_list, rotation=10)
        ax.set_ylabel("Avg Daily PnL ($)")
        ax.set_title(f"{seg_type} × Sentiment — Avg Daily PnL")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, axis="y", alpha=0.4)
        ax.axhline(0, color="#666", linestyle="--", alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with c2:
        fig, ax = plt.subplots(figsize=(8, 5))
        if not seg_sent.empty and seg_sent.shape[0] > 0 and seg_sent.shape[1] > 0:
            sns.heatmap(seg_sent.round(0), annot=True, fmt=".0f",
                        cmap="RdYlGn", ax=ax, linewidths=0.5,
                        linecolor="#0f1117",
                        cbar_kws={"label": "Avg Daily PnL ($)"})
            ax.set_title(f"{seg_type} Heatmap")
            plt.setp(ax.get_xticklabels(), rotation=20)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Segment summary stats
    st.markdown("#### Segment Stats")
    seg_stats = (
        daily_seg_f.groupby(seg_col)
        .agg(
            avg_pnl      = ("daily_pnl",    "mean"),
            win_rate     = ("pnl_positive", "mean"),
            avg_trades   = ("trade_count",  "mean"),
            avg_size     = ("avg_size_usd", "mean"),
            long_ratio   = ("long_ratio",   "mean"),
            n_obs        = ("daily_pnl",    "count"),
        )
        .round(3)
        .reset_index()
    )
    st.dataframe(
        seg_stats.style.format({
            "avg_pnl":    "${:,.0f}",
            "win_rate":   "{:.1%}",
            "avg_trades": "{:.0f}",
            "avg_size":   "${:,.0f}",
            "long_ratio": "{:.3f}",
        }).background_gradient(subset=["avg_pnl", "win_rate"], cmap="RdYlGn"),
        use_container_width=True,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — TIME SERIES
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("PnL & Sentiment Over Time")

    # Aggregate by date
    daily_ts = (
        daily_f.groupby("date")
        .agg(
            total_pnl    = ("daily_pnl",    "sum"),
            trade_count  = ("trade_count",  "sum"),
            win_rate     = ("pnl_positive", "mean"),
        )
        .reset_index()
    )
    fg_ts = fg[fg["date"].between(d_start, d_end)].copy()
    merged_ts = daily_ts.merge(fg_ts[["date", "value", "classification"]], on="date", how="left")
    merged_ts["pnl_ma7"] = merged_ts["total_pnl"].rolling(7, min_periods=1).mean()

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

    # PnL
    ax = axes[0]
    ax.bar(merged_ts["date"], merged_ts["total_pnl"],
           color=[SENT_COLORS.get(s, "#555") for s in merged_ts["classification"].fillna("Neutral")],
           alpha=0.5, width=1)
    ax.plot(merged_ts["date"], merged_ts["pnl_ma7"], "white", linewidth=1.5, label="7-day MA")
    ax.axhline(0, color="#666", linestyle="--", alpha=0.5)
    ax.set_ylabel("Total PnL ($)")
    ax.set_title("Daily Aggregate PnL (colored by sentiment)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    # Win rate
    ax2 = axes[1]
    ax2.plot(merged_ts["date"], merged_ts["win_rate"].rolling(7, min_periods=1).mean(),
             color="#2ecc71", linewidth=1.5)
    ax2.fill_between(merged_ts["date"],
                      merged_ts["win_rate"].rolling(7, min_periods=1).mean(),
                      0.5, alpha=0.2,
                      where=merged_ts["win_rate"].rolling(7, min_periods=1).mean() > 0.5,
                      color="#2ecc71")
    ax2.axhline(0.5, color="white", linestyle="--", alpha=0.5)
    ax2.set_ylabel("Win Rate (7-day MA)")
    ax2.set_ylim(0, 1)
    ax2.grid(True, axis="y", alpha=0.3)

    # Fear/Greed
    ax3 = axes[2]
    ax3.plot(fg_ts["date"], fg_ts["value"], color="#3498db", linewidth=1.5)
    ax3.fill_between(fg_ts["date"], fg_ts["value"], 50,
                      where=fg_ts["value"] < 50, alpha=0.3, color="#e74c3c", label="Fear zone")
    ax3.fill_between(fg_ts["date"], fg_ts["value"], 50,
                      where=fg_ts["value"] >= 50, alpha=0.3, color="#2ecc71", label="Greed zone")
    ax3.axhline(50, color="white", linestyle="--", alpha=0.5)
    ax3.set_ylabel("Fear/Greed Index")
    ax3.set_xlabel("Date")
    ax3.legend(fontsize=8)
    ax3.grid(True, axis="y", alpha=0.3)

    patches = [mpatches.Patch(color=v, label=k) for k, v in SENT_COLORS.items()
               if k in merged_ts["classification"].dropna().unique()]
    fig.legend(handles=patches, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.02), fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — LEADERBOARD
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Trader Leaderboard")

    # Merge segment info
    lb = trader_stats.copy()
    lb["wallet_short"] = lb["Account"].apply(lambda x: f"{x[:8]}…{x[-4:]}")

    c1, c2 = st.columns([2, 1])
    with c2:
        sort_by = st.selectbox("Sort by", [
            "total_pnl", "sharpe_proxy", "mean_win_rate",
            "avg_trades_per_day", "avg_size_usd"
        ])
        top_n = st.slider("Show top N", 5, len(lb), min(20, len(lb)))

    lb_sorted = lb.sort_values(sort_by, ascending=False).head(top_n)

    with c1:
        fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.35)))
        y = np.arange(len(lb_sorted))
        colors_bar = ["#2ecc71" if v > 0 else "#e74c3c" for v in lb_sorted["total_pnl"]]
        ax.barh(y, lb_sorted["total_pnl"], color=colors_bar, edgecolor="#3a3d52")
        ax.set_yticks(y)
        ax.set_yticklabels(lb_sorted["wallet_short"], fontsize=8)
        ax.set_xlabel("Total PnL ($)")
        ax.set_title(f"Top {top_n} Traders by Total PnL")
        ax.axvline(0, color="white", linewidth=0.8, alpha=0.5)
        ax.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Table
    display_lb = lb_sorted[[
        "wallet_short", "total_pnl", "avg_daily_pnl", "sharpe_proxy",
        "mean_win_rate", "avg_trades_per_day", "avg_size_usd",
        "trading_days", "mean_long_ratio",
    ]].copy()
    display_lb.columns = [
        "Wallet", "Total PnL", "Avg Daily PnL", "Sharpe (proxy)",
        "Win Rate", "Avg Trades/Day", "Avg Size", "Days Active", "Long Ratio"
    ]
    st.dataframe(
        display_lb.style.format({
            "Total PnL":     "${:,.0f}",
            "Avg Daily PnL": "${:,.0f}",
            "Sharpe (proxy)":"  {:.3f}",
            "Win Rate":      "{:.1%}",
            "Avg Trades/Day":"{:.1f}",
            "Avg Size":      "${:,.0f}",
            "Long Ratio":    "{:.3f}",
        }).background_gradient(
            subset=["Total PnL", "Win Rate", "Sharpe (proxy)"], cmap="RdYlGn"
        ),
        use_container_width=True,
    )

    st.markdown("---")
    st.markdown(
        "**Sharpe proxy** = avg_daily_pnl / pnl_std — higher is better risk-adjusted return.  \n"
        "**Win Rate** = fraction of days with positive PnL."
    )
