"""
Hidden Markov Model for Financial Regime Detection
===================================================
Detects market regimes (Bull, Bear, High-Volatility) from financial time series
using a Gaussian HMM on log-returns.

Dependencies:
    pip install numpy pandas matplotlib seaborn hmmlearn yfinance scipy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from hmmlearn import hmm
from scipy import stats
import warnings
import sys

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1.  DATA LOADING
# ─────────────────────────────────────────────

def load_data(source: str = "synthetic", ticker: str = "SPY",
              start: str = "2010-01-01", end: str = "2024-01-01") -> pd.DataFrame:
    """
    Load financial data.

    Parameters
    ----------
    source  : 'synthetic' or 'yfinance'
    ticker  : Yahoo Finance ticker (used when source='yfinance')
    start   : ISO date string
    end     : ISO date string

    Returns
    -------
    pd.DataFrame with columns ['Close', 'Log_Return']
    """
    if source == "yfinance":
        try:
            import yfinance as yf
            df = yf.download(ticker, start=start, end=end, progress=False)
            df = df[["Close"]].dropna()
            df.index = pd.to_datetime(df.index)
            print(f"[✓] Downloaded {len(df)} rows for {ticker} from Yahoo Finance.")
        except Exception as exc:
            print(f"[!] yfinance failed ({exc}). Falling back to synthetic data.")
            return load_data("synthetic")
    else:
        print("[✓] Generating synthetic multi-regime market data …")
        df = _generate_synthetic_data()

    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    df.dropna(inplace=True)
    return df


def _generate_synthetic_data(n_days: int = 3_500, seed: int = 42) -> pd.DataFrame:
    """
    Simulate a price series that visits three distinct volatility/drift regimes.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start="2010-01-01", periods=n_days)

    # Regime parameters: (mean daily return, daily volatility, avg duration days)
    regimes = {
        "bull":       (0.0008, 0.008, 300),
        "bear":       (-0.0006, 0.015, 120),
        "high_vol":   (0.0001, 0.025, 60),
    }
    regime_seq = ["bull", "bear", "bull", "high_vol", "bull",
                  "bear", "high_vol", "bull", "bear", "bull",
                  "high_vol", "bear", "bull"]

    returns = []
    for name in regime_seq:
        mu, sigma, dur = regimes[name]
        n = int(rng.normal(dur, dur * 0.2))
        n = max(n, 20)
        returns.extend(rng.normal(mu, sigma, n).tolist())

    returns = np.array(returns[:n_days])
    price = 100 * np.exp(np.cumsum(returns))
    df = pd.DataFrame({"Close": price}, index=dates[:len(price)])
    return df


# ─────────────────────────────────────────────
# 2.  MODEL
# ─────────────────────────────────────────────

class RegimeHMM:
    """
    Gaussian HMM wrapper that fits to log-returns and assigns
    human-readable regime labels based on posterior statistics.
    """

    LABEL_COLORS = {
        "Bull Market":       "#2ecc71",
        "Bear Market":       "#e74c3c",
        "High Volatility":   "#f39c12",
        "Unknown":           "#95a5a6",
    }

    def __init__(self, n_regimes: int = 3, n_iter: int = 200,
                 covariance_type: str = "full", random_state: int = 42):
        self.n_regimes       = n_regimes
        self.n_iter          = n_iter
        self.covariance_type = covariance_type
        self.random_state    = random_state
        self.model           = None
        self.state_labels    = {}          # state_id -> readable label
        self.hidden_states   = None        # array of predicted states

    # ── fitting ──────────────────────────────

    def fit(self, returns: np.ndarray) -> "RegimeHMM":
        X = returns.reshape(-1, 1)
        self.model = hmm.GaussianHMM(
            n_components     = self.n_regimes,
            covariance_type  = self.covariance_type,
            n_iter           = self.n_iter,
            random_state     = self.random_state,
        )
        self.model.fit(X)
        self.hidden_states = self.model.predict(X)
        self._assign_labels()
        print(f"[✓] HMM converged: log-likelihood = {self.model.score(X):.2f}")
        return self

    def _assign_labels(self):
        """
        Label each latent state by comparing its mean return and volatility
        to the overall median.  Three canonical regimes are recognised.
        """
        means = self.model.means_.flatten()
        vols  = np.sqrt(self.model.covars_.flatten())

        # Rank states: highest mean → Bull; lowest mean → Bear; highest vol → High-Vol
        sorted_by_mean = np.argsort(means)   # ascending
        sorted_by_vol  = np.argsort(vols)    # ascending

        # Provisional assignment
        labels = {}
        labels[sorted_by_mean[-1]] = "Bull Market"
        labels[sorted_by_mean[0]]  = "Bear Market"
        # The remaining state(s) get High-Volatility
        for s in range(self.n_regimes):
            if s not in labels:
                labels[s] = "High Volatility"

        self.state_labels = labels

    # ── inference ────────────────────────────

    def predict(self, returns: np.ndarray) -> np.ndarray:
        return self.model.predict(returns.reshape(-1, 1))

    def get_regime_series(self, index) -> pd.Series:
        labels = [self.state_labels.get(s, "Unknown") for s in self.hidden_states]
        return pd.Series(labels, index=index, name="Regime")

    def get_state_stats(self, returns: np.ndarray) -> pd.DataFrame:
        rows = []
        for state in range(self.n_regimes):
            mask  = self.hidden_states == state
            r     = returns[mask]
            label = self.state_labels.get(state, f"State {state}")
            rows.append({
                "State":          state,
                "Label":          label,
                "Mean Return":    r.mean(),
                "Volatility":     r.std(),
                "Sharpe (annlzd)": (r.mean() / r.std()) * np.sqrt(252) if r.std() > 0 else 0,
                "% of Time":      mask.mean() * 100,
                "HMM Mean":       self.model.means_[state, 0],
                "HMM Sigma":      np.sqrt(self.model.covars_[state].flatten()[0]),
            })
        return pd.DataFrame(rows).set_index("Label")

    @property
    def transition_matrix(self) -> pd.DataFrame:
        labels = [self.state_labels.get(s, f"S{s}") for s in range(self.n_regimes)]
        return pd.DataFrame(self.model.transmat_, index=labels, columns=labels)


# ─────────────────────────────────────────────
# 3.  ANALYTICS
# ─────────────────────────────────────────────

def compute_regime_performance(price: pd.Series, regimes: pd.Series) -> pd.DataFrame:
    """Annualised return & max-drawdown per regime."""
    rows = []
    df   = pd.DataFrame({"price": price, "regime": regimes}).dropna()
    for label in df["regime"].unique():
        mask  = df["regime"] == label
        sub   = df.loc[mask, "price"]
        if len(sub) < 2:
            continue
        log_r = np.log(sub / sub.shift(1)).dropna()
        ann_r = log_r.mean() * 252
        ann_v = log_r.std() * np.sqrt(252)
        cum   = (1 + log_r).cumprod()
        roll_max = cum.cummax()
        drawdown = (cum - roll_max) / roll_max
        rows.append({
            "Regime":             label,
            "Ann. Return":        ann_r,
            "Ann. Volatility":    ann_v,
            "Max Drawdown":       drawdown.min(),
            "# Observations":     int(mask.sum()),
        })
    return pd.DataFrame(rows).set_index("Regime")


# ─────────────────────────────────────────────
# 4.  VISUALISATION
# ─────────────────────────────────────────────

def plot_all(df: pd.DataFrame, model: RegimeHMM):
    """Produce a four-panel diagnostic figure."""
    regimes  = model.get_regime_series(df.index)
    returns  = df["Log_Return"].values
    colors   = RegimeHMM.LABEL_COLORS

    fig = plt.figure(figsize=(18, 14), facecolor="#0d1117")
    fig.suptitle("Hidden Markov Model — Market Regime Detection",
                 color="white", fontsize=18, fontweight="bold", y=0.98)

    ax_style = dict(facecolor="#161b22", tick_params=dict(colors="#8b949e"),
                    spine_color="#30363d")

    def style_ax(ax):
        ax.set_facecolor(ax_style["facecolor"])
        for sp in ax.spines.values():
            sp.set_color(ax_style["spine_color"])
        ax.tick_params(colors=ax_style["tick_params"]["colors"])
        ax.xaxis.label.set_color("#8b949e")
        ax.yaxis.label.set_color("#8b949e")
        ax.title.set_color("white")

    gs = fig.add_gridspec(4, 2, hspace=0.45, wspace=0.3,
                          top=0.93, bottom=0.05, left=0.06, right=0.97)

    # ── Panel 1: Price with regime shading ──────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df.index, df["Close"], color="#58a6ff", linewidth=0.9, zorder=3)

    current_regime = None
    start_idx      = df.index[0]
    for i, (idx, reg) in enumerate(regimes.items()):
        if reg != current_regime:
            if current_regime is not None:
                ax1.axvspan(start_idx, idx,
                            color=colors.get(current_regime, "#95a5a6"),
                            alpha=0.25, zorder=1)
            current_regime = reg
            start_idx      = idx
    ax1.axvspan(start_idx, df.index[-1],
                color=colors.get(current_regime, "#95a5a6"), alpha=0.25, zorder=1)

    patches = [mpatches.Patch(color=c, alpha=0.6, label=l)
               for l, c in colors.items() if l != "Unknown"]
    ax1.legend(handles=patches, loc="upper left",
               facecolor="#161b22", edgecolor="#30363d",
               labelcolor="white", fontsize=9)
    ax1.set_title("Asset Price with Detected Regimes")
    ax1.set_ylabel("Price")
    style_ax(ax1)

    # ── Panel 2: Regime over time ────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :])
    numeric_regime = regimes.map({v: k for k, v in model.state_labels.items()})
    ax2.plot(df.index, numeric_regime.values, color="#f0883e", linewidth=0.8)
    ax2.set_yticks(list(model.state_labels.keys()))
    ax2.set_yticklabels([model.state_labels[s] for s in model.state_labels],
                        color="#c9d1d9", fontsize=8)
    ax2.set_title("Latent State Sequence")
    style_ax(ax2)

    # ── Panel 3: Return distribution by regime ───────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    for label, grp in pd.DataFrame({"ret": returns, "regime": regimes.values}).groupby("regime"):
        grp["ret"].hist(ax=ax3, bins=60, alpha=0.55,
                        color=colors.get(label, "#95a5a6"),
                        label=label, density=True)
    ax3.set_title("Return Distributions by Regime")
    ax3.set_xlabel("Log Return")
    ax3.legend(facecolor="#161b22", edgecolor="#30363d",
               labelcolor="white", fontsize=8)
    style_ax(ax3)

    # ── Panel 4: Transition matrix heatmap ──────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    trans = model.transition_matrix
    sns.heatmap(trans, annot=True, fmt=".3f", ax=ax4,
                cmap="YlOrRd", linewidths=0.5,
                cbar_kws={"shrink": 0.8},
                annot_kws={"color": "black", "size": 9})
    ax4.set_title("Regime Transition Matrix")
    ax4.tick_params(colors="#c9d1d9", labelsize=8)
    ax4.set_facecolor("#161b22")

    # ── Panel 5: Posterior probabilities ────────────────────────────────────
    ax5 = fig.add_subplot(gs[3, :])
    posteriors = model.model.predict_proba(returns.reshape(-1, 1))
    for state in range(model.n_regimes):
        label = model.state_labels.get(state, f"S{state}")
        ax5.fill_between(df.index, posteriors[:, state],
                         alpha=0.6, label=label,
                         color=colors.get(label, "#95a5a6"))
    ax5.set_title("Posterior State Probabilities")
    ax5.set_ylabel("P(state | data)")
    ax5.legend(facecolor="#161b22", edgecolor="#30363d",
               labelcolor="white", fontsize=8, loc="upper right")
    style_ax(ax5)

    plt.savefig("hmm_regimes.png",
                dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print("[✓] Figure saved → hmm_regimes.png")


# ─────────────────────────────────────────────
# 5.  REPORTING
# ─────────────────────────────────────────────

def print_report(df: pd.DataFrame, model: RegimeHMM):
    sep = "═" * 62
    print(f"\n{sep}")
    print("  HMM REGIME DETECTION — SUMMARY REPORT")
    print(sep)

    print("\n▸  STATE STATISTICS (from fitted model)\n")
    stats_df = model.get_state_stats(df["Log_Return"].values)
    print(stats_df.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n▸  REGIME PERFORMANCE (empirical)\n")
    perf_df = compute_regime_performance(df["Close"], model.get_regime_series(df.index))
    print(perf_df.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n▸  TRANSITION MATRIX\n")
    print(model.transition_matrix.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\n{sep}\n")


# ─────────────────────────────────────────────
# 6.  MAIN
# ─────────────────────────────────────────────

def main():
    # ── Config ──────────────────────────────────────────────────────────────
    DATA_SOURCE  = "synthetic"   # "synthetic" | "yfinance"
    TICKER       = "SPY"
    START        = "2010-01-01"
    END          = "2024-01-01"
    N_REGIMES    = 3            # number of hidden states

    # Override source if a ticker is supplied via CLI
    if len(sys.argv) > 1:
        TICKER      = sys.argv[1]
        DATA_SOURCE = "yfinance"
        print(f"[i] Ticker override: {TICKER}")

    # ── Pipeline ─────────────────────────────────────────────────────────────
    print("\n── Step 1 / 4  Load data ──")
    df = load_data(DATA_SOURCE, TICKER, START, END)
    print(f"    Period : {df.index[0].date()} → {df.index[-1].date()}")
    print(f"    Points : {len(df)}")

    print("\n── Step 2 / 4  Fit HMM ──")
    model = RegimeHMM(n_regimes=N_REGIMES, n_iter=300)
    model.fit(df["Log_Return"].values)

    print("\n── Step 3 / 4  Report ──")
    print_report(df, model)

    print("── Step 4 / 4  Plots ──")
    plot_all(df, model)

    print("Done ✓\n")


if __name__ == "__main__":
    main()