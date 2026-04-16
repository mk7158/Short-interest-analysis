import os
import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from scipy.stats.mstats import winsorize

warnings.filterwarnings("ignore")

CYCLE_DIR = ""
PRICE_CACHE = "./price_cache.parquet"
OUTPUT_DIR = "./outputs"
TOP_DECILE = 0.90
HORIZONS = list(range(0, 11))
EVENT_WINDOW = 5

VANGUARD_NAVY = "#0a2463"
SUCCESS_TEAL = "#1d7874"
RISK_RED = "#c0392b"

CAP_TIER_ORDER = ["Mega (>$200B)", "Large ($10-200B)", "Mid ($2-10B)", "Small (<$2B)", "Unknown"]

def get_tier(mcap):
    if pd.isna(mcap): return "Unknown"
    if mcap >= 200e9: return "Mega (>$200B)"
    if mcap >= 10e9:  return "Large ($10-200B)"
    if mcap >= 2e9:   return "Mid ($2-10B)"
    return "Small (<$2B)"

def fetch_fundamental_data(tickers):
    data = []
    for t in tickers:
        try:
            tkr = yf.Ticker(t)
            s = tkr.info.get('sharesOutstanding')
            data.append({'ticker': t, 'shares_out': s})
        except Exception:
            continue
    return pd.DataFrame(data)

def load_cycles(directory):
    files = sorted(glob.glob(os.path.join(directory, "Bats_Short_Interest_settle_*.csv")))
    if not files:
        raise SystemExit(f"No files in {directory}")

    column_mapping = {
        "Cycle Settlement Date": "date",
        "BATS-Symbol": "ticker",
        "# Shares Net Short Current Cycle": "short_shares",
        "Cycle Avg Daily Trade Vol": "avg_volume",
        "Min # of Trade Days To Cover Shorts": "days_to_cover",
    }

    frames = []
    for f in files:
        try:
            df = pd.read_csv(f).rename(columns=column_mapping)
        except Exception:
            continue
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
        keep = [c for c in column_mapping.values() if c in df.columns]
        frames.append(df[keep])

    panel = pd.concat(frames, ignore_index=True).dropna(subset=["date", "ticker"])
    for col in ["short_shares", "avg_volume", "days_to_cover"]:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")

    panel = (panel.sort_values(["ticker", "date"])
                  .drop_duplicates(subset=["ticker", "date"], keep="last")
                  .reset_index(drop=True))

    panel = panel[
        (panel["short_shares"] >= 1000) &
        (panel["avg_volume"] >= 10000) &
        (panel["days_to_cover"].notna())
    ].copy()

    cycle_count = panel.groupby("ticker")["date"].nunique()
    panel = panel[panel["ticker"].isin(cycle_count[cycle_count >= 6].index)].reset_index(drop=True)

    panel["short_to_vol"] = panel["short_shares"] / panel["avg_volume"]
    panel = panel.sort_values(["ticker", "date"])
    panel["si_growth"] = panel.groupby("ticker")["short_shares"].pct_change()
    panel["si_growth"] = panel["si_growth"].replace([np.inf, -np.inf], np.nan).fillna(0)

    return panel

def fetch_prices(tickers, start, end, use_cache=True):
    if use_cache and Path(PRICE_CACHE).exists():
        return pd.read_parquet(PRICE_CACHE)

    tickers = sorted({str(t) for t in tickers})
    start = pd.Timestamp(start) - pd.Timedelta(days=120)
    end = pd.Timestamp(end) + pd.Timedelta(days=15)

    frames = []
    for i in range(0, len(tickers), 60):
        chunk = tickers[i:i + 60]
        try:
            data = yf.download(chunk, start=start, end=end, group_by="ticker",
                               auto_adjust=True, threads=True, progress=False)
        except Exception:
            continue
        if data is None or data.empty:
            continue

        if len(chunk) == 1:
            tkr = chunk[0]
            sub = data.reset_index()
            if not sub.empty and not sub["Close"].isna().all():
                sub["ticker"] = tkr
                frames.append(sub[["Date", "ticker", "Close", "Volume", "High"]])
        else:
            try:
                top_level = set(data.columns.get_level_values(0))
            except Exception:
                top_level = set()
            for tkr in chunk:
                if tkr not in top_level:
                    continue
                sub = data[tkr].reset_index()
                if sub.empty or sub["Close"].isna().all():
                    continue
                sub["ticker"] = tkr
                frames.append(sub[["Date", "ticker", "Close", "Volume", "High"]])

    if not frames:
        return pd.DataFrame()

    prices = pd.concat(frames, ignore_index=True).rename(columns={"Date": "date"})
    prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None)
    if use_cache:
        prices.to_parquet(PRICE_CACHE)
    return prices

def robust_filter_universe(panel, prices, min_price=5.0, min_dollar_vol=1_000_000):
    panel = panel.sort_values("date")
    prices = prices.sort_values("date")
    df = pd.merge_asof(panel, prices[['date', 'ticker', 'Close']], on="date", by="ticker", direction="backward")
    df["dollar_volume"] = df["avg_volume"] * df["Close"]
    return df[(df["Close"] >= min_price) & (df["dollar_volume"] >= min_dollar_vol)].copy()

def define_specials_robust(panel, top_decile=0.90):
    df = panel.copy()

    df["si_size_rank"] = df.groupby("date")["short_shares"].rank(pct=True)

    for col in ["days_to_cover", "si_growth"]:
        df[col] = df[col].fillna(df[col].median())
        df[col] = winsorize(df[col], limits=[0.01, 0.01])

    df["dtc_rank"] = df.groupby("date")["days_to_cover"].rank(pct=True)
    df["growth_rank"] = df.groupby("date")["si_growth"].rank(pct=True)

    df["spec_scarcity"] = (df["dtc_rank"] >= top_decile).astype(int)
    df["spec_size"] = (df["si_size_rank"] >= top_decile).astype(int)
    df["spec_momentum"] = (df["growth_rank"] >= top_decile).astype(int)

    df["spec_composite"] = (
        (df["spec_scarcity"] + df["spec_size"] + df["spec_momentum"]) >= 2
    ).astype(int)

    return df

def survival_curves(panel_with_specials, horizons=HORIZONS):
    cycles = sorted(panel_with_specials["date"].unique())
    cycle_idx = {c: i for i, c in enumerate(cycles)}
    ticker_status = {c: set(panel_with_specials[(panel_with_specials["date"] == c) & (panel_with_specials["spec_composite"] == 1)]["ticker"]) for c in cycles}

    horizon_rates = []
    for h in horizons:
        hits, totals = 0, 0
        for c in cycles:
            idx = cycle_idx[c]
            if idx + h >= len(cycles):
                continue
            future_cycle = cycles[idx + h]
            current_set = ticker_status[c]
            future_set = ticker_status[future_cycle]
            if current_set:
                hits += len(current_set & future_set)
                totals += len(current_set)
        horizon_rates.append(hits / totals if totals else np.nan)

    return pd.DataFrame({'Composite': horizon_rates}, index=horizons)

def compute_price_features(prices):
    if prices.empty:
        return prices
    prices = prices.sort_values(["ticker", "date"]).copy()
    g = prices.groupby("ticker", group_keys=False)
    prices["ret_1d"] = g["Close"].pct_change()
    prices["rv_21d"] = g["ret_1d"].rolling(21, min_periods=10).std().reset_index(level=0, drop=True)
    vol_mean = g["Volume"].rolling(63, min_periods=21).mean().reset_index(level=0, drop=True)
    vol_std = g["Volume"].rolling(63, min_periods=21).std().reset_index(level=0, drop=True)
    prices["volume_z63"] = (prices["Volume"] - vol_mean) / vol_std.replace(0, np.nan)
    return prices[["date", "ticker", "rv_21d", "volume_z63"]]

def event_study(panel_with_specials, price_features, spec_col="spec_composite", window=EVENT_WINDOW):
    df = panel_with_specials.sort_values(["ticker", "date"]).copy()
    df["spec_prev"] = df.groupby("ticker")[spec_col].shift(1)
    df["entry"] = ((df[spec_col] == 1) & (df["spec_prev"] == 0)).astype(int)

    cycles = sorted(df["date"].unique())
    cycle_idx = {c: i for i, c in enumerate(cycles)}

    cp = df[["ticker", "date", "entry"]].sort_values("date").reset_index(drop=True)
    pf = price_features.sort_values("date").reset_index(drop=True)
    df = pd.merge_asof(cp, pf, on="date", by="ticker", direction="backward", tolerance=pd.Timedelta(days=7))

    entries = df[df["entry"] == 1][["ticker", "date"]].values
    rows = []
    for ticker, entry_date in entries:
        idx = cycle_idx.get(entry_date)
        if idx is None:
            continue
        for offset in range(-window, window + 1):
            if 0 <= idx + offset < len(cycles):
                target_date = cycles[idx + offset]
                row = df[(df["ticker"] == ticker) & (df["date"] == target_date)]
                if not row.empty:
                    r = row.iloc[0]
                    rows.append({"offset": offset, "rv_21d": r.get("rv_21d"), "volume_z63": r.get("volume_z63")})

    event_df = pd.DataFrame(rows)
    return event_df.groupby("offset")[["rv_21d", "volume_z63"]].mean().reset_index() if not event_df.empty else pd.DataFrame()

def chart_revenue_persistence(survival_df, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.fill_between(survival_df.index, survival_df["Composite"],
                    color=VANGUARD_NAVY, alpha=0.1)
    ax.plot(survival_df.index, survival_df["Composite"],
            marker="o", linewidth=3, color=VANGUARD_NAVY,
            label="Composite Special")
    ax.axhline(0.5, color=RISK_RED, linestyle="--", alpha=0.5,
               label="50% Cohort Retention Reference")

    ax.set_title(
        "Exhibit 2: Persistence of Composite Specials (Survival Curve)",
        fontsize=14, fontweight="bold", loc="left",
    )
    ax.set_ylabel("Fraction of Original Cohort Still 'Special'")
    ax.set_xlabel("Settlement Cycles Since Entry (Bi-Monthly)")
    ax.legend(frameon=False, loc="upper right")
    ax.set_ylim(0, 1.05)

    final_retention = survival_df["Composite"].iloc[-1]
    one_cycle_retention = survival_df["Composite"].iloc[1]
    ax.annotate(
        f"Composite specials retain {final_retention:.0%}\n"
        f"of cohort even after 10 cycles\n"
        f"({one_cycle_retention:.0%} after just 1 cycle)",
        xy=(10, final_retention),
        xytext=(6.5, 0.80),
        fontsize=10,
        ha="left",
        arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=6),
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f7",
                  edgecolor="#ccc", linewidth=0.5),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

def compute_concentration(panel_specs):
    cycles = sorted(panel_specs["date"].unique())
    rows = []

    for c in cycles:
        snap = panel_specs[panel_specs["date"] == c]
        total_si = snap["short_shares"].sum()
        total_names = len(snap)

        if total_si == 0 or total_names == 0:
            continue

        spec_snap = snap[snap["spec_composite"] == 1]
        spec_si = spec_snap["short_shares"].sum()
        spec_names = len(spec_snap)

        rows.append({
            "date": c,
            "specials_pct_names": spec_names / total_names,
            "specials_pct_short_interest": spec_si / total_si,
            "total_si": total_si,
            "spec_si": spec_si,
        })

    return pd.DataFrame(rows)


def chart_concentration(concentration_df, output_path):
    avg_names_pct = concentration_df["specials_pct_names"].mean()
    avg_si_pct = concentration_df["specials_pct_short_interest"].mean()
    concentration_ratio = avg_si_pct / avg_names_pct

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    ax1 = axes[0]
    categories = ["% of Names", "% of Total\nShort Interest"]
    values = [avg_names_pct * 100, avg_si_pct * 100]
    colors = [GREY, VANGUARD_NAVY]
    bars = ax1.bar(categories, values, color=colors, edgecolor="white",
                   linewidth=1, width=0.5)

    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.5, f"{val:.1f}%",
                 ha="center", fontsize=11, fontweight="bold", color="#222")

    ax1.set_title(f"Composite Specials Hold {avg_si_pct:.0%} of Total Short Interest",
                  fontsize=11.5, color="#1a1a1a", pad=10)
    ax1.set_ylabel("Share (%)")
    ax1.set_ylim(0, max(values) * 1.25)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.text(0.5, -0.18,
             f"Concentration ratio: {concentration_ratio:.1f}x\n"
             f"(specials punch {concentration_ratio:.1f}x above their weight)",
             transform=ax1.transAxes, ha="center", fontsize=9,
             color="#444", style="italic")

    ax2 = axes[1]
    ax2.fill_between(concentration_df["date"],
                     concentration_df["specials_pct_short_interest"] * 100,
                     color=VANGUARD_NAVY, alpha=0.2)
    ax2.plot(concentration_df["date"],
             concentration_df["specials_pct_short_interest"] * 100,
             linewidth=2, color=VANGUARD_NAVY, label="% of Short Interest")
    ax2.axhline(concentration_df["specials_pct_names"].mean() * 100,
                color=GREY, linestyle="--", linewidth=1,
                label=f"% of Names ({avg_names_pct:.0%}) — reference")
    ax2.set_title("Stability of Concentration Over Time",
                  fontsize=11.5, color="#1a1a1a", pad=10)
    ax2.set_ylabel("% of Total Short Interest in Specials")
    ax2.set_xlabel("Settlement Cycle")
    ax2.legend(frameon=False, loc="upper right", fontsize=9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.tick_params(axis="x", rotation=30)

    fig.suptitle("Exhibit 1: Concentration of Short Interest in Composite Specials",
                 fontsize=14, fontweight="bold", x=0.05, ha="left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

def chart_scarcity_utilization(panel_specs, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))

    plot_data = panel_specs[panel_specs["cap_segment"] != "Unknown"].copy()
    plot_data["cap_segment"] = pd.Categorical(
        plot_data["cap_segment"],
        categories=[c for c in CAP_TIER_ORDER if c != "Unknown"],
        ordered=True,
    )

    sns.boxplot(
        x="spec_composite", y="short_to_vol", hue="cap_segment",
        data=plot_data, ax=ax,
        hue_order=[c for c in CAP_TIER_ORDER if c != "Unknown"],
    )
    ax.set_yscale("log")
    ax.set_title(
        "Exhibit 3: Short-to-Volume Distribution by Market Cap Tier",
        fontsize=14, fontweight="bold", loc="left",
    )
    ax.set_xticklabels(["General Collateral", "Composite Specials"])
    ax.set_ylabel("Short-to-Volume Ratio (Log Scale)")
    ax.set_xlabel("")
    plt.legend(title="Market Cap", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

def diagnose_definition_overlap(panel_specs):
    cycles = sorted(panel_specs["date"].unique())
    pairs = [
        ("spec_scarcity", "spec_size", "Scarcity vs Size"),
        ("spec_scarcity", "spec_momentum", "Scarcity vs Momentum"),
        ("spec_size", "spec_momentum", "Size vs Momentum"),
    ]
    print("\n Definition Overlap Diagnostic")
    for a, b, label in pairs:
        jacs = []
        for c in cycles:
            snap = panel_specs[panel_specs["date"] == c]
            sa = set(snap[snap[a] == 1]["ticker"])
            sb = set(snap[snap[b] == 1]["ticker"])
            union = len(sa | sb)
            jacs.append(len(sa & sb) / union if union else 0)
        print(f"  {label}: {np.mean(jacs):.1%}")

if __name__ == "__main__":
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    panel = load_cycles(CYCLE_DIR)
    all_tickers = panel["ticker"].unique().tolist()

    prices_raw = fetch_prices(all_tickers, panel["date"].min(), panel["date"].max())
    shares_df = fetch_fundamental_data(all_tickers)

    panel_filtered = robust_filter_universe(panel, prices_raw)

    latest_prices = prices_raw.sort_values("date").groupby("ticker").last()["Close"].reset_index()
    mkt_cap_map = pd.merge(shares_df, latest_prices, on="ticker")
    mkt_cap_map["mkt_cap"] = mkt_cap_map["shares_out"] * mkt_cap_map["Close"]
    mkt_cap_map["cap_segment"] = mkt_cap_map["mkt_cap"].apply(get_tier)

    panel_enriched = pd.merge(panel_filtered, mkt_cap_map[['ticker', 'cap_segment']], on='ticker', how='left')
    panel_specs = define_specials_robust(panel_enriched)

    diagnose_definition_overlap(panel_specs)

    survival_df = survival_curves(panel_specs)
    chart_revenue_persistence(survival_df, os.path.join(OUTPUT_DIR, "exhibit1_persistence.png"))

    chart_scarcity_utilization(panel_specs, os.path.join(OUTPUT_DIR, "exhibit3_utilization.png"))

    concentration_df = compute_concentration(panel_specs)
    chart_concentration(concentration_df, os.path.join(OUTPUT_DIR, "exhibit2_concentration.png"))

    print("\n Concentration Summary")
    print(f"Avg % of names that are composite specials: "
          f"{concentration_df['specials_pct_names'].mean():.1%}")
    print(f"Avg % of total short interest in composite specials: "
          f"{concentration_df['specials_pct_short_interest'].mean():.1%}")
    print(f"Concentration ratio: "
          f"{concentration_df['specials_pct_short_interest'].mean() / concentration_df['specials_pct_names'].mean():.2f}x")

