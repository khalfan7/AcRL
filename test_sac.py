"""
Daily Episode Profile — SAC Agent (Testing Phase)
Runs the trained SAC agent on a single 24-hour day and plots:
  Top:    Indoor Temp  vs  Outdoor Temp  vs  Setpoint
  Bottom: HVAC Power (Action) over time

Simulator sign convention  (thermal_simulator.py):
    delta_T = (dt/C) * [ U·(T_out − T_in)  −  hvac_power ]
    hvac_power = action × max_hvac_power

    ▸ action > 0  →  positive power  →  COOLS the zone  (removes heat)
    ▸ action < 0  →  negative power  →  HEATS the zone  (adds heat)

Night windows : 00-06 and 18-24
Afternoon heat: 12-18
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')          # non-interactive backend — avoids Qt crash
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from hvac_environment import HVACControlEnv

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE   = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(_HERE, "results", "SAC")

model_path = os.path.join(LOG_DIR, "best_model.zip")
if not os.path.exists(model_path):
    model_path = os.path.join(LOG_DIR, "final_model.zip")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found in {LOG_DIR}. Run train_sac.py first.")

print(f"Loading model: {model_path}")

# ── Environment ───────────────────────────────────────────────────────────────
def make_env():
    return HVACControlEnv()

env = DummyVecEnv([make_env])

stats_path = os.path.join(LOG_DIR, "vecnormalize.pkl")
if os.path.exists(stats_path):
    env = VecNormalize.load(stats_path, env)
    env.training    = False
    env.norm_reward = False
    print("Normalisation stats loaded.")
else:
    print("Warning: vecnormalize.pkl not found — running without obs normalisation.")

# ── Load model ────────────────────────────────────────────────────────────────
model = SAC.load(model_path, env=env)

# ── Simulation loop ───────────────────────────────────────────────────────────
obs     = env.reset()
raw_env = env.unwrapped.envs[0]
sim     = raw_env.simulator

setpoint  = sim.setpoint_temp
max_power = sim.max_hvac_power
dt_ctrl   = sim.control_timestep
n_steps   = sim.max_steps

outdoor_profile = sim.outdoor_temp_array.copy()

# Pre-allocate arrays
time_arr    = np.empty(n_steps)
indoor_arr  = np.empty(n_steps)
outdoor_arr = np.empty(n_steps)
power_arr   = np.empty(n_steps)

step_count = 0
for i in range(n_steps):
    action, _ = model.predict(obs, deterministic=True)
    obs, rewards, dones, infos = env.step(action)

    info = infos[0]
    time_arr[i]    = info["elapsed_hours"]
    indoor_arr[i]  = info["indoor_temp"]
    power_arr[i]   = info["hvac_power"]          # +ve = cooling, −ve = heating
    outdoor_arr[i] = outdoor_profile[min(i, n_steps - 1)]
    step_count     = i + 1

    if dones[0]:
        break

# Trim to actual episode length
time_arr    = time_arr[:step_count]
indoor_arr  = indoor_arr[:step_count]
outdoor_arr = outdoor_arr[:step_count]
power_arr   = power_arr[:step_count]

# ── Metrics ───────────────────────────────────────────────────────────────────
mae  = np.mean(np.abs(indoor_arr - setpoint))
rmse = np.sqrt(np.mean((indoor_arr - setpoint) ** 2))

outside_band = np.abs(indoor_arr - setpoint) > 0.5
comfort_viol = outside_band.sum() / step_count * 100

cool_mask = power_arr > 0
heat_mask = power_arr < 0

peak_cool = power_arr[cool_mask].max() if cool_mask.any() else 0.0
peak_heat = np.abs(power_arr[heat_mask].min()) if heat_mask.any() else 0.0

total_energy_wh = np.sum(np.abs(power_arr)) * dt_ctrl / 3600.0

print(f"\n── Test-Day Performance ──────────────────────────────")
print(f"  MAE  from setpoint : {mae:.3f} °C")
print(f"  RMSE from setpoint : {rmse:.3f} °C")
print(f"  Comfort violations : {comfort_viol:.1f}% of steps outside ±0.5 °C")
print(f"  Peak cooling power : {peak_cool:,.0f} W  ({peak_cool/max_power*100:.1f}% of capacity)")
print(f"  Peak heating power : {peak_heat:,.0f} W  ({peak_heat/max_power*100:.1f}% of capacity)")
print(f"  Total energy used  : {total_energy_wh:,.0f} Wh  ({total_energy_wh/1000:.2f} kWh)")
print(f"─────────────────────────────────────────────────────\n")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 8))
fig.suptitle(
    f"SAC Agent — 24-Hour Episode   |   "
    f"MAE {mae:.2f} °C    RMSE {rmse:.2f} °C    Energy {total_energy_wh/1000:.2f} kWh",
    fontsize=13, fontweight="bold",
)
gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[2, 1], hspace=0.35)

# ── Top: temperatures ──────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])

for ns, ne in [(0, 6), (18, 24)]:
    ax1.axvspan(ns, ne, alpha=0.08, color="navy", zorder=0)
ax1.axvspan(12, 18, alpha=0.07, color="orange", zorder=0)

ax1.plot(time_arr, indoor_arr,  color="crimson",   linewidth=2,   label="Indoor Temp")
ax1.plot(time_arr, outdoor_arr, color="royalblue", linewidth=1.5, label="Outdoor Temp", linestyle="--")
ax1.axhline(setpoint, color="green", linewidth=1.8, linestyle=":",
            label=f"Setpoint ({setpoint:.0f} °C)")
ax1.fill_between(time_arr, setpoint - 0.5, setpoint + 0.5,
                 color="green", alpha=0.12, label="±0.5 °C comfort band")

y_lo, _ = ax1.get_ylim()
ax1.text( 3, y_lo + 0.3, "Night",            ha="center", fontsize=8, color="navy",       style="italic")
ax1.text(15, y_lo + 0.3, "Afternoon\nheat",  ha="center", fontsize=8, color="darkorange", style="italic")
ax1.text(21, y_lo + 0.3, "Night",            ha="center", fontsize=8, color="navy",       style="italic")

ax1.set_ylabel("Temperature (°C)", fontsize=11)
ax1.set_xlim(0, 24)
ax1.set_xticks(range(0, 25, 2))
ax1.legend(loc="upper right", fontsize=9)
ax1.grid(alpha=0.3)

# ── Bottom: HVAC power ─────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1], sharex=ax1)

for ns, ne in [(0, 6), (18, 24)]:
    ax2.axvspan(ns, ne, alpha=0.08, color="navy", zorder=0)
ax2.axvspan(12, 18, alpha=0.07, color="orange", zorder=0)

ax2.fill_between(time_arr, 0, power_arr, where=(power_arr >= 0),
                 interpolate=True, color="deepskyblue", alpha=0.7, label="Cooling (+)")
ax2.fill_between(time_arr, 0, power_arr, where=(power_arr <= 0),
                 interpolate=True, color="orangered",   alpha=0.7, label="Heating (−)")
ax2.plot(time_arr, power_arr, color="black", linewidth=0.6, alpha=0.5)
ax2.axhline(0, color="black", linewidth=0.8)
ax2.axhline( max_power, color="grey", linewidth=0.8, linestyle=":", alpha=0.6)
ax2.axhline(-max_power, color="grey", linewidth=0.8, linestyle=":", alpha=0.6)
ax2.text(24.15,  max_power, f"+{max_power:.0f}", va="center", fontsize=7, color="grey")
ax2.text(24.15, -max_power, f"−{max_power:.0f}", va="center", fontsize=7, color="grey")

ax2.set_ylabel("HVAC Power (W)\n[+ cooling  /  − heating]", fontsize=10)
ax2.set_xlabel("Hour of Day", fontsize=11)
ax2.set_xlim(0, 24)
ax2.set_ylim(-max_power * 1.05, max_power * 1.05)
ax2.set_xticks(range(0, 25, 2))
ax2.legend(loc="upper right", fontsize=9)
ax2.grid(alpha=0.3)

out_path = os.path.join(LOG_DIR, "daily_episode_profile_sac.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Figure saved \u2192 {out_path}")
