"""
Report Plots — AcRL (Nonlinear Dynamics)
=========================================
Compact 3+1 figure result section.  Reads CSVs + trace npz files produced
by ``generalization.py``.

Figures  (→ results/plots/)
---------------------------
  Fig 1.  albany_ood_summary.png      — Albany OOD test: 3 heatmaps (MAE,
          comfort-in-band, energy) with rows = algorithms, cols = seasons
  Fig 2.  generalization_gap.png      — One heatmap: % change Syracuse → Albany
  Fig 3.  best_policy_traces.png      — Winter + Summer episodes for best algo
  Fig 4.  training_convergence.png    — (optional) learning curves

Usage
-----
  python -m evaluation.report_plots              # all 4 figures
  python -m evaluation.report_plots --best SAC_nl  # override best algo
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES_DIR  = _ROOT / "results"
PLOT_DIR = RES_DIR / "plots"

# ── Constants ─────────────────────────────────────────────────────────────────
ALGOS       = ["PPO_nl", "A2C_nl", "SAC_nl", "TD3_nl"]
ALGO_LABELS = {"PPO_nl": "PPO", "A2C_nl": "A2C",
                "SAC_nl": "SAC", "TD3_nl": "TD3"}
ALGO_COLORS = {"PPO_nl": "#1f77b4", "A2C_nl": "#ff7f0e",
                "SAC_nl": "#2ca02c", "TD3_nl": "#d62728"}
ALGO_STYLES = {"PPO_nl": "-", "A2C_nl": "--",
                "SAC_nl": "-.", "TD3_nl": ":"}

SEASONS = ["Winter", "Spring", "Summer", "Fall"]

# Main-text metrics only
MAIN_METRICS = ["mae", "comfort_in_band_pct", "energy_kwh"]
MAIN_LABELS  = {
    "mae":                 "MAE (°C)",
    "comfort_in_band_pct": "Time within ±0.5°C (%)",
    "energy_kwh":          "Energy (kWh/day)",
}
# For the gap figure: positive = worse for MAE & energy, negative = worse for comfort
_HIGHER_IS_WORSE = {"mae": True, "comfort_in_band_pct": False, "energy_kwh": True}

# Per-metric display formatting
FMT = {
    "mae":                 "{:.2f}",
    "comfort_in_band_pct": "{:.0f}",
    "energy_kwh":          "{:.1f}",
}

# Fixed color ranges for Figure 1 (stable across reruns)
RANGES = {
    "mae":                 (0.0, 1.5),
    "comfort_in_band_pct": (70.0, 100.0),
    "energy_kwh":          (0.0, 60.0),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save(fig, name: str):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    p = PLOT_DIR / name
    fig.savefig(str(p), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {p}")


def _load_csvs() -> pd.DataFrame:
    """Load all generalization CSVs; add comfort_in_band_pct column."""
    frames = []
    for algo in ALGOS:
        for city_tag, city_label in [
            ("albany (test)",    "Albany"),
            ("syracuse (train)", "Syracuse"),
        ]:
            path = RES_DIR / algo / f"generalization_stats_{algo.lower()}_{city_tag}.csv"
            if not path.exists():
                print(f"  [skip] {path.name}")
                continue
            df = pd.read_csv(path)
            df["algo"] = algo
            df["city"] = city_label
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["comfort_in_band_pct"] = 100.0 - df["violations_pct"]
    return df


def _load_convergence() -> dict:
    data = {}
    for algo in ALGOS:
        path = RES_DIR / algo / "evaluations.npz"
        if not path.exists():
            continue
        npz = np.load(path)
        data[algo] = {
            "timesteps": npz["timesteps"],
            "results":   npz["results"],
            "mean":      npz["results"].mean(axis=1),
        }
    return data


def _load_traces(algo: str, city_tag: str):
    path = RES_DIR / algo / f"generalization_traces_{algo.lower()}_{city_tag}.npz"
    if not path.exists():
        return None
    return dict(np.load(path, allow_pickle=False))


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — Albany OOD Test Summary  (3 side-by-side annotated heatmaps)
# ═══════════════════════════════════════════════════════════════════════════════

def _fig_albany_summary(df: pd.DataFrame) -> None:
    albany = df[df["city"] == "Albany"]
    avail  = [a for a in ALGOS if not albany[albany["algo"] == a].empty]
    if not avail:
        print("  [skip] no Albany data")
        return

    col       = "mae"
    cmap_name = "Reds"
    n_algos   = len(avail)
    n_seasons = len(SEASONS)

    # Same size as the generalization gap figure
    fig, ax = plt.subplots(figsize=(8, 0.9 * n_algos + 2))

    mat  = np.full((n_algos, n_seasons), np.nan)
    text = [[""] * n_seasons for _ in range(n_algos)]
    for ai, algo in enumerate(avail):
        for si, s in enumerate(SEASONS):
            vals = albany[(albany["algo"] == algo) &
                          (albany["season"] == s)][col].values
            if len(vals) == 0:
                continue
            med = float(np.median(vals))
            mat[ai, si]  = med
            text[ai][si] = FMT[col].format(med)

    mn, mx = RANGES[col]
    normed = np.where(np.isnan(mat), 0.3,
                      np.clip((mat - mn) / max(mx - mn, 1e-9), 0.0, 1.0))
    cmap = plt.get_cmap(cmap_name)

    for ai in range(n_algos):
        for si in range(n_seasons):
            rgba = cmap(0.15 + 0.7 * normed[ai, si])
            ax.add_patch(plt.Rectangle((si, ai), 1, 1, facecolor=rgba,
                                        edgecolor="white", linewidth=2))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            tc  = "white" if lum < 0.55 else "black"
            ax.text(si + 0.5, ai + 0.5, text[ai][si],
                    ha="center", va="center", fontsize=12,
                    fontweight="bold", color=tc)

    ax.set_xlim(0, n_seasons)
    ax.set_ylim(0, n_algos)
    ax.set_xticks([i + 0.5 for i in range(n_seasons)])
    ax.set_xticklabels(SEASONS, fontsize=10)
    ax.set_yticks([i + 0.5 for i in range(n_algos)])
    ax.set_yticklabels([ALGO_LABELS[a] for a in avail], fontsize=11)
    ax.invert_yaxis()
    ax.tick_params(length=0)
    ax.set_frame_on(False)
    ax.set_title("Albany — MAE by Season (°C)\n"
                 "(median over 25 episodes per season)",
                 fontsize=12, fontweight="bold", pad=14)

    fig.tight_layout()
    _save(fig, "albany_ood_summary.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — Generalization Gap  (single annotated heatmap)
# ═══════════════════════════════════════════════════════════════════════════════

def _fig_generalization_gap(df: pd.DataFrame) -> None:
    avail = []
    mat = []

    # Column labels: positive always means worse on Albany
    col_labels = ["MAE change (%)", "Comfort change (%)", "Energy change (%)"]

    for algo in ALGOS:
        syr = df[(df["algo"] == algo) & (df["city"] == "Syracuse")]
        alb = df[(df["algo"] == algo) & (df["city"] == "Albany")]
        if syr.empty or alb.empty:
            continue

        row = []
        for m in MAIN_METRICS:
            s_ref = syr[m].median()
            a_ref = alb[m].median()
            raw_pct = 0.0 if abs(s_ref) < 1e-9 else (a_ref - s_ref) / abs(s_ref) * 100.0

            # positive = worse on Albany for every metric
            disp_pct = raw_pct if _HIGHER_IS_WORSE[m] else -raw_pct
            row.append(disp_pct)

        avail.append(algo)
        mat.append(row)

    if not avail:
        print("  [skip] need both cities for gap figure")
        return

    mat = np.array(mat)
    row_labels = [ALGO_LABELS[a] for a in avail]

    fig, ax = plt.subplots(figsize=(8, 0.9 * len(avail) + 2))
    max_abs = max(np.abs(mat).max(), 1.0)

    im = ax.imshow(mat, cmap="RdYlGn_r", aspect="auto",
                   vmin=-max_abs, vmax=max_abs)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=11)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=12)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            tc = "white" if abs(v) > max_abs * 0.55 else "black"
            ax.text(j, i, f"{v:+.1f}%", ha="center", va="center",
                    fontsize=12, fontweight="bold", color=tc)

    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.03)
    cbar.set_label("Sign-adjusted relative change (%)", fontsize=10)

    ax.set_title("Generalization Gap (Syracuse \u2192 Albany)",
                 fontsize=12, fontweight="bold", pad=14)

    fig.tight_layout()
    _save(fig, "generalization_gap.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — Best-Policy Behavior Traces  (Winter + Summer, Albany)
# ═══════════════════════════════════════════════════════════════════════════════

def _fig_best_traces(df: pd.DataFrame, best_algo: str) -> None:
    city_tag = "albany (test)"
    csv_path = RES_DIR / best_algo / f"generalization_stats_{best_algo.lower()}_{city_tag}.csv"
    traces   = _load_traces(best_algo, city_tag)
    if traces is None or not csv_path.exists():
        print(f"  [skip] traces or CSV missing for {best_algo}")
        return

    ep_df = pd.read_csv(csv_path)
    label = ALGO_LABELS.get(best_algo, best_algo)

    setpoint  = 22.0
    max_power = float(np.nanmax(np.abs(traces["power"])))

    show_seasons = ["Winter", "Summer"]
    season_colors = {"Winter": "#1f77b4", "Summer": "#d62728"}

    fig, (ax_t, ax_p) = plt.subplots(1, 2, figsize=(14, 4.5),
                                      gridspec_kw={"wspace": 0.28})
    fig.suptitle(f"Winter vs Summer on Albany ({label})",
                  fontsize=13, fontweight="bold")

    months = traces["month"]

    for season in show_seasons:
        sub = ep_df[ep_df["season"] == season]
        if sub.empty:
            continue

        # Pick the episode closest to median MAE
        med      = sub["mae"].median()
        best_idx = int(sub.loc[(sub["mae"] - med).abs().idxmin(), "episode"])

        time_arr = traces["time"][best_idx]
        in_arr   = traces["indoor"][best_idx]
        out_arr  = traces["outdoor"][best_idx]
        pwr_arr  = traces["power"][best_idx]
        rec      = ep_df[ep_df["episode"] == best_idx].iloc[0]
        comfort_pct = 100.0 - rec["violations_pct"]
        clr = season_colors[season]

        # ── Temperature panel ─────────────────────────────────────────────
        ax_t.plot(time_arr, in_arr,  color=clr, lw=2.2,
                  label=f"{season} indoor (MAE {rec['mae']:.2f}°C, {comfort_pct:.0f}% comfort)")
        ax_t.plot(time_arr, out_arr, color=clr, lw=1.2, ls="--", alpha=0.5,
                  label=f"{season} outdoor")

        # ── HVAC Power panel ──────────────────────────────────────────────
        ax_p.plot(time_arr, pwr_arr, color=clr, lw=1.8,
                  label=f"{season} ({rec['energy_kwh']:.1f} kWh/day)")

    # Temperature axis formatting
    ax_t.axhline(setpoint, color="green", lw=1.5, ls=":", label="Setpoint")
    ax_t.fill_between([0, 24], setpoint - 0.5, setpoint + 0.5,
                      color="green", alpha=0.10, label="±0.5°C band")
    ax_t.set_ylabel("Temperature (°C)", fontsize=10)
    ax_t.set_title("Temperature Tracking", fontsize=11, fontweight="bold")
    ax_t.grid(alpha=0.3)
    ax_t.set_xlim(0, 24)
    ax_t.set_xticks(range(0, 25, 4))
    ax_t.legend(fontsize=8, loc="best")
    ax_t.set_xlabel("Hour of Day")

    # HVAC Power axis formatting
    ax_p.axhline(0, color="black", lw=0.5)
    ax_p.set_ylim(-1.05 * max_power, 1.05 * max_power)
    ax_p.set_ylabel("HVAC Power (W)", fontsize=10)
    ax_p.set_title("HVAC Power", fontsize=11, fontweight="bold")
    ax_p.grid(alpha=0.3)
    ax_p.set_xlim(0, 24)
    ax_p.set_xticks(range(0, 25, 4))
    ax_p.legend(fontsize=8, loc="best")
    ax_p.set_xlabel("Hour of Day")

    _save(fig, "best_policy_traces.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 4 (optional) — Training Convergence
# ═══════════════════════════════════════════════════════════════════════════════

def _fig_convergence(conv: dict) -> None:
    if not conv:
        print("  [skip] no evaluations.npz data")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for algo in ALGOS:
        if algo not in conv:
            continue
        d    = conv[algo]
        ts   = d["timesteps"]
        raw  = d["results"]          # (n_evals, n_eval_episodes)
        mean = d["mean"]

        # Faint scatter of individual eval episodes
        for j in range(raw.shape[1]):
            ax.scatter(ts, raw[:, j], color=ALGO_COLORS[algo],
                       alpha=0.12, s=8, zorder=1, linewidths=0)

        ax.plot(ts, mean,
                color=ALGO_COLORS[algo], linestyle=ALGO_STYLES[algo],
                linewidth=2.2, label=ALGO_LABELS[algo], zorder=3)

    ax.set_xlabel("Environment Timesteps", fontsize=11)
    ax.set_ylabel("Episode Reward", fontsize=11)
    ax.set_title("Training Convergence", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(alpha=0.25)
    ax.ticklabel_format(axis="x", style="scientific", scilimits=(0, 0))
    fig.tight_layout()
    _save(fig, "training_convergence.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINTS
# ═══════════════════════════════════════════════════════════════════════════════

def generate_report(best_algo: str = "TD3_nl"):
    """Generate the compact 3+1 figure result section."""
    print("Loading data …")
    df   = _load_csvs()
    conv = _load_convergence()
    if df.empty:
        print("  No CSVs found — run test scripts first.")
        return
    print(f"  {len(df)} rows — "
          f"{df['algo'].nunique()} algos × {df['city'].nunique()} cities\n")

    print("Fig 1: Albany OOD Test Summary")
    _fig_albany_summary(df)

    print("Fig 2: Generalization Gap (Syracuse → Albany)")
    _fig_generalization_gap(df)

    print(f"Fig 3: Best-Policy Traces ({ALGO_LABELS.get(best_algo, best_algo)})")
    _fig_best_traces(df, best_algo)

    print("Fig 4: Training Convergence (optional)")
    _fig_convergence(conv)

    print(f"\nAll figures saved to {PLOT_DIR}/")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="AcRL Compact Report Plots")
    p.add_argument("--best", type=str, default="TD3_nl",
                   help="Algorithm for the behaviour trace figure (default: TD3_nl)")
    args = p.parse_args()
    generate_report(best_algo=args.best)
