"""Shared 100-episode generalisation test engine.

Called by each ``test_<algo>.py`` wrapper.  The only per-algorithm inputs are
the SB3 algorithm class and its string name.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs import make_test_env

# ── Season helpers ────────────────────────────────────────────────────────────
_M2S = {12: "Winter", 1: "Winter", 2: "Winter",
         3: "Spring", 4: "Spring", 5: "Spring",
         6: "Summer", 7: "Summer", 8: "Summer",
         9: "Fall",  10: "Fall",  11: "Fall"}
_SEASON_ORDER = ["Winter", "Spring", "Summer", "Fall"]
_COLORS = {"Winter": "#4a90d9", "Spring": "#50c878",
           "Summer": "#f4a460", "Fall":   "#cd853f"}
_METRICS = [("mae",            "MAE (C)"),
            ("rmse",           "RMSE (C)"),
            ("violations_pct", "Comfort Violations (%)"),
            ("energy_kwh",     "Energy (kWh / day)")]


def _fmt(s: pd.Series) -> str:
    return f"{s.mean():.3f} +/- {s.std():.3f}"


# ── Main entry point ─────────────────────────────────────────────────────────

def run_generalization_test(algo_cls, algo_name: str, *, n_episodes: int = 100):
    """Run a stratified seasonal evaluation and produce stats + figures.

    Parameters
    ----------
    algo_cls : type
        SB3 algorithm class (e.g. ``PPO``, ``SAC``).
    algo_name : str
        Upper-case label used for directory names and titles.
    n_episodes : int
        Total episodes (cycled W -> Sp -> Su -> F).
    """
    _ROOT   = Path(__file__).resolve().parent.parent
    LOG_DIR = _ROOT / "results" / algo_name

    # ── Load model ────────────────────────────────────────────────────────
    model_path = LOG_DIR / "best_model.zip"
    if not model_path.exists():
        model_path = LOG_DIR / "final_model.zip"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No model in {LOG_DIR}. Run train_{algo_name.lower()}.py first."
        )

    print(f"Loading model: {model_path}")

    env = DummyVecEnv([make_test_env])

    stats_path = LOG_DIR / "vecnormalize.pkl"
    if stats_path.exists():
        try:
            env = VecNormalize.load(str(stats_path), env)
            env.training    = False
            env.norm_reward = False
            print("Normalisation stats loaded.")
        except Exception as e:
            print(f"Warning: could not load vecnormalize.pkl ({e}).")
    else:
        print("Warning: vecnormalize.pkl not found -- running without obs normalisation.")

    model = algo_cls.load(str(model_path), env=env)

    # ── Run episodes ──────────────────────────────────────────────────────
    records = []
    traces  = []

    obs = env.reset()

    raw_env   = env.unwrapped.envs[0]
    sim       = raw_env.simulator
    setpoint  = sim.setpoint_temp
    max_power = sim.max_hvac_power
    dt_ctrl   = sim.control_timestep
    n_steps   = sim.max_steps

    for ep in range(n_episodes):
        init     = sim.get_state()
        ep_month = int(init["month"])
        ep_doy   = int(init["doy"])

        t_arr   = np.empty(n_steps)
        in_arr  = np.empty(n_steps)
        out_arr = np.empty(n_steps)
        pwr_arr = np.empty(n_steps)
        ghi_arr = np.empty(n_steps)
        wnd_arr = np.empty(n_steps)
        rew_arr = np.empty(n_steps)

        for i in range(n_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, rews, dones, infos = env.step(action)
            info         = infos[0]
            t_arr[i]     = info["elapsed_hours"]
            in_arr[i]    = info["indoor_temp"]
            out_arr[i]   = info["outdoor_temp"]
            pwr_arr[i]   = info["hvac_power"]
            ghi_arr[i]   = info["GHI"]
            wnd_arr[i]   = info["wind_speed"]
            rew_arr[i]   = rews[0]

        mae    = float(np.mean(np.abs(in_arr - setpoint)))
        rmse   = float(np.sqrt(np.mean((in_arr - setpoint) ** 2)))
        viol   = float(np.mean(np.abs(in_arr - setpoint) > 0.5) * 100)
        energy = float(np.sum(np.abs(pwr_arr)) * dt_ctrl / 3.6e6)
        tot_r  = float(rew_arr.sum())
        season = _M2S[ep_month]

        records.append(dict(episode=ep, season=season, month=ep_month, doy=ep_doy,
                            mae=mae, rmse=rmse, violations_pct=viol,
                            energy_kwh=energy, reward=tot_r))
        traces.append(dict(time=t_arr.copy(), indoor=in_arr.copy(),
                           outdoor=out_arr.copy(), power=pwr_arr.copy(),
                           ghi=ghi_arr.copy(), wind=wnd_arr.copy(),
                           month=ep_month, doy=ep_doy, season=season))

        if (ep + 1) % 25 == 0 or ep == 0:
            print(f"  Ep {ep+1:3d}/{n_episodes}  {season:6s}  "
                  f"MAE {mae:.3f}  Viol {viol:5.1f}%  Energy {energy:.2f} kWh")

    env.close()
    df = pd.DataFrame(records)

    # ── Save CSV ──────────────────────────────────────────────────────────
    csv_path = LOG_DIR / f"generalization_stats_{algo_name.lower()}.csv"
    df.to_csv(str(csv_path), index=False)
    print(f"\nStats CSV -> {csv_path}")

    # ── Console summary ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  {algo_name} Generalisation Test -- Albany NY  ({n_episodes} episodes)")
    print(f"{'='*70}")
    print(f"  Overall  (n = {len(df)})")
    print(f"  {'-'*50}")
    print(f"    MAE           : {_fmt(df['mae'])} C")
    print(f"    RMSE          : {_fmt(df['rmse'])} C")
    print(f"    Violations    : {_fmt(df['violations_pct'])} %")
    print(f"    Energy        : {_fmt(df['energy_kwh'])} kWh/day")
    print(f"    Reward        : {_fmt(df['reward'])}")
    print()
    print(f"  Per-Season Breakdown")
    print(f"  {'-'*66}")
    print(f"  {'Season':<8s}  {'n':>3s}  {'MAE(C)':>14s}  {'RMSE(C)':>14s}  "
          f"{'Viol(%)':>14s}  {'Energy(kWh)':>14s}")
    for s in _SEASON_ORDER:
        sub = df[df["season"] == s]
        if sub.empty:
            continue
        print(f"  {s:<8s}  {len(sub):>3d}  {_fmt(sub['mae']):>14s}  "
              f"{_fmt(sub['rmse']):>14s}  {_fmt(sub['violations_pct']):>14s}  "
              f"{_fmt(sub['energy_kwh']):>14s}")
    print(f"  {'-'*66}\n")

    # ── Figure 1: boxplots ────────────────────────────────────────────────
    fig1, axes1 = plt.subplots(2, 2, figsize=(12, 9))
    fig1.suptitle(f"{algo_name} -- Generalisation on Albany ({n_episodes} episodes)",
                  fontsize=13, fontweight="bold")

    for ax, (col, label) in zip(axes1.flat, _METRICS):
        data = [df.loc[df["season"] == s, col].values for s in _SEASON_ORDER]
        bp = ax.boxplot(data, labels=_SEASON_ORDER, patch_artist=True, widths=0.55)
        for patch, s in zip(bp["boxes"], _SEASON_ORDER):
            patch.set_facecolor(_COLORS[s])
            patch.set_alpha(0.7)
        ax.set_ylabel(label, fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    fig1.tight_layout(rect=[0, 0, 1, 0.94])
    p1 = LOG_DIR / f"generalization_boxplots_{algo_name.lower()}.png"
    fig1.savefig(str(p1), dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"Figure saved -> {p1}")

    # ── Figure 2: representative episode per season ───────────────────────
    fig2, axes2 = plt.subplots(4, 2, figsize=(14, 14),
                                gridspec_kw={"hspace": 0.45, "wspace": 0.28})
    fig2.suptitle(f"{algo_name} -- Representative Episodes by Season (Albany)",
                  fontsize=13, fontweight="bold")

    for row, s in enumerate(_SEASON_ORDER):
        sub = df[df["season"] == s]
        if sub.empty:
            for c in range(2):
                axes2[row, c].set_visible(False)
            continue
        med      = sub["mae"].median()
        best_idx = int(sub.loc[(sub["mae"] - med).abs().idxmin(), "episode"])
        tr  = traces[best_idx]
        rec = records[best_idx]

        ax_t = axes2[row, 0]
        ax_p = axes2[row, 1]
        time = tr["time"]

        # -- Temperature --
        ax_t.plot(time, tr["indoor"],  color="crimson",   lw=2,   label="Indoor")
        ax_t.plot(time, tr["outdoor"], color="royalblue", lw=1.5, ls="--", label="Outdoor")
        ax_t.axhline(setpoint, color="green", lw=1.5, ls=":", label="Setpoint")
        ax_t.fill_between(time, setpoint - 0.5, setpoint + 0.5,
                          color="green", alpha=0.10)
        ax_t.set_ylabel("Temp (C)", fontsize=9)
        ax_t.set_title(f"{s}  (month {tr['month']}, DOY {tr['doy']})  "
                       f"MAE {rec['mae']:.2f} C  Viol {rec['violations_pct']:.1f}%",
                       fontsize=9)
        ax_t.grid(alpha=0.3)
        ax_t.set_xlim(0, 24)
        ax_t.set_xticks(range(0, 25, 4))
        if row == 0:
            ax_t.legend(fontsize=7, loc="upper right")
        if row == 3:
            ax_t.set_xlabel("Hour of Day")

        # -- HVAC Power --
        pwr = tr["power"]
        ax_p.fill_between(time, 0, pwr, where=(pwr >= 0),
                          interpolate=True, color="deepskyblue", alpha=0.7, label="Cool")
        ax_p.fill_between(time, 0, pwr, where=(pwr <= 0),
                          interpolate=True, color="orangered",   alpha=0.7, label="Heat")
        ax_p.axhline(0, color="black", lw=0.5)
        ax_p.set_ylim(-max_power * 1.05, max_power * 1.05)
        ax_p.set_ylabel("HVAC (W)", fontsize=9)
        ax_p.set_title(f"Energy {rec['energy_kwh']:.2f} kWh", fontsize=9)
        ax_p.grid(alpha=0.3)
        ax_p.set_xlim(0, 24)
        ax_p.set_xticks(range(0, 25, 4))
        if row == 0:
            ax_p.legend(fontsize=7, loc="upper right")
        if row == 3:
            ax_p.set_xlabel("Hour of Day")

    p2 = LOG_DIR / f"seasonal_profiles_{algo_name.lower()}.png"
    fig2.savefig(str(p2), dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Figure saved -> {p2}")
