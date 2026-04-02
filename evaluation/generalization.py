"""
Evaluation Engine — AcRL (Nonlinear Dynamics)
==============================================
Data-collection only.  Runs the trained model, saves CSVs + trace npz files.
No figures are generated here — see ``report_plots.py`` for all plotting.

Per-algorithm evaluation (called by test_<algo>.py wrappers):
    from evaluation.generalization import run_train_vs_test
    run_train_vs_test(SAC, "SAC_nl")

Outputs per city  (written to results/<ALGO>/)
----------------------------------------------
  generalization_stats_<algo>_<city>.csv    — episode-level metrics
  generalization_traces_<algo>_<city>.npz   — per-step time-series (for trajectory plots)
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs import make_test_env

RES_DIR  = _ROOT / "results"

# ── Season helpers ────────────────────────────────────────────────────────────
_M2S = {12: "Winter", 1: "Winter", 2: "Winter",
         3: "Spring", 4: "Spring", 5: "Spring",
         6: "Summer", 7: "Summer", 8: "Summer",
         9: "Fall",  10: "Fall",  11: "Fall"}
_SEASON_ORDER = ["Winter", "Spring", "Summer", "Fall"]


# ── Shared helpers ────────────────────────────────────────────────────────────
def _fmt(s: pd.Series) -> str:
    return f"{s.mean():.3f} +/- {s.std():.3f}"


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 1 — Per-Algorithm Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def run_generalization_test(algo_cls, algo_name: str, *,
                            n_episodes: int = 100,
                            env_factory=None,
                            city_name: str = "Albany"):
    """Run a stratified seasonal evaluation, save CSV + trace npz.

    Parameters
    ----------
    algo_cls : type
        SB3 algorithm class (e.g. ``PPO``, ``SAC``).
    algo_name : str
        Upper-case label used for directory names and titles.
    n_episodes : int
        Total episodes (cycled W -> Sp -> Su -> F).
    env_factory : callable or None
        Factory function returning a Gymnasium env.  Defaults to
        ``make_test_env`` (Albany).
    city_name : str
        Human-readable city label used in titles and file names.
    """
    from envs import make_test_env as _default_factory
    if env_factory is None:
        env_factory = _default_factory

    LOG_DIR = RES_DIR / algo_name

    # ── Load model ────────────────────────────────────────────────────────
    model_path = LOG_DIR / "best_model.zip"
    if not model_path.exists():
        model_path = LOG_DIR / "final_model.zip"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No model in {LOG_DIR}. Run train_{algo_name.lower()}.py first."
        )

    print(f"Loading model: {model_path}")

    env = DummyVecEnv([env_factory])

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
        act_arr = np.empty(n_steps)
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
            act_arr[i]   = float(action.flatten()[0])
            ghi_arr[i]   = info["GHI"]
            wnd_arr[i]   = info["wind_speed"]
            rew_arr[i]   = rews[0]

        # ── Per-episode metrics ───────────────────────────────────────────
        errors = np.abs(in_arr - setpoint)
        mae    = float(np.mean(errors))
        rmse   = float(np.sqrt(np.mean(errors ** 2)))
        viol05 = float(np.mean(errors > 0.5) * 100)
        viol10 = float(np.mean(errors > 1.0) * 100)
        energy = float(np.sum(np.abs(pwr_arr)) * dt_ctrl / 3.6e6)
        tot_r  = float(rew_arr.sum())
        season = _M2S[ep_month]
        peak_w = float(np.max(np.abs(pwr_arr)))
        slew   = float(np.mean(np.abs(np.diff(act_arr))))

        records.append(dict(
            episode=ep, season=season, month=ep_month, doy=ep_doy,
            mae=mae, rmse=rmse,
            violations_pct=viol05, within_1c_pct=100.0 - viol10,
            energy_kwh=energy, reward=tot_r,
            mean_slew=slew, peak_power_w=peak_w,
        ))
        traces.append(dict(time=t_arr.copy(), indoor=in_arr.copy(),
                           outdoor=out_arr.copy(), power=pwr_arr.copy(),
                           action=act_arr.copy(),
                           ghi=ghi_arr.copy(), wind=wnd_arr.copy(),
                           month=ep_month, doy=ep_doy, season=season))

        if (ep + 1) % 25 == 0 or ep == 0:
            print(f"  Ep {ep+1:3d}/{n_episodes}  {season:6s}  "
                  f"MAE {mae:.3f}  Viol {viol05:5.1f}%  Energy {energy:.2f} kWh")

    env.close()
    df = pd.DataFrame(records)

    # ── Save CSV ──────────────────────────────────────────────────────────
    city_tag  = city_name.lower()
    csv_path = LOG_DIR / f"generalization_stats_{algo_name.lower()}_{city_tag}.csv"
    df.to_csv(str(csv_path), index=False)
    print(f"\nStats CSV -> {csv_path}")

    # ── Console summary ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  {algo_name} Generalisation Test -- {city_name}  ({n_episodes} episodes)")
    print(f"{'='*70}")
    print(f"  Overall  (n = {len(df)})")
    print(f"  {'-'*50}")
    print(f"    MAE           : {_fmt(df['mae'])} C")
    print(f"    RMSE          : {_fmt(df['rmse'])} C")
    print(f"    Violations    : {_fmt(df['violations_pct'])} %")
    print(f"    Energy        : {_fmt(df['energy_kwh'])} kWh/day")
    print(f"    Mean slew     : {_fmt(df['mean_slew'])}")
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

    # ── Save traces npz (for trajectory plots in report_plots.py) ───────
    trace_arrays = {
        "time":    np.array([t["time"]    for t in traces]),
        "indoor":  np.array([t["indoor"]  for t in traces]),
        "outdoor": np.array([t["outdoor"] for t in traces]),
        "power":   np.array([t["power"]   for t in traces]),
        "action":  np.array([t["action"]  for t in traces]),
        "ghi":     np.array([t["ghi"]     for t in traces]),
        "wind":    np.array([t["wind"]    for t in traces]),
        "month":   np.array([t["month"]   for t in traces]),
        "doy":     np.array([t["doy"]     for t in traces]),
    }
    npz_path = LOG_DIR / f"generalization_traces_{algo_name.lower()}_{city_tag}.npz"
    np.savez_compressed(str(npz_path), **trace_arrays)
    print(f"Traces  -> {npz_path}")

    return df


# ── Train vs Test comparison ──────────────────────────────────────────────────

def run_train_vs_test(algo_cls, algo_name: str, *, n_episodes: int = 100):
    """Evaluate on both cities, save CSVs + traces.  No plots."""
    from envs import make_train_env, make_test_env

    print("\n" + "=" * 70)
    print(f"  {algo_name}  --  Training city (Syracuse)")
    print("=" * 70)
    df_train = run_generalization_test(
        algo_cls, algo_name,
        n_episodes=n_episodes,
        env_factory=make_train_env,
        city_name="Syracuse (train)",
    )

    print("\n" + "=" * 70)
    print(f"  {algo_name}  --  Test city (Albany)")
    print("=" * 70)
    df_test = run_generalization_test(
        algo_cls, algo_name,
        n_episodes=n_episodes,
        env_factory=make_test_env,
        city_name="Albany (test)",
    )

    # ── Console gap summary ───────────────────────────────────────────────
    metrics = [
        ("mae",            "MAE (°C)"),
        ("rmse",           "RMSE (°C)"),
        ("violations_pct", "Comfort Violations (%)"),
        ("energy_kwh",     "Energy (kWh/day)"),
    ]
    print(f"\n{'='*70}")
    print(f"  Generalisation Gap  (Albany − Syracuse)  /  Syracuse  × 100")
    print(f"  {'Metric':<24s}  {'Overall gap':>12s}")
    print(f"  {'-'*40}")
    for col, label in metrics:
        t_mean = df_train[col].mean()
        g_mean = df_test[col].mean()
        gap    = (g_mean - t_mean) / t_mean * 100 if t_mean != 0 else float("nan")
        sign   = "+" if gap >= 0 else ""
        print(f"  {label:<24s}  {sign}{gap:>10.1f}%")
    print(f"{'='*70}\n")

