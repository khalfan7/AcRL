# DRL-Based HVAC Control for a Single-Zone Smart Building

Deep reinforcement learning agents (PPO, A2C, SAC, TD3) that control indoor
temperature in a physics-based thermal zone using **real EPA hourly weather
data** and **real New York monthly electricity prices**, with **per-episode
domain randomisation** of building parameters. The agent balances comfort,
energy cost, and actuator smoothness.

The simulator uses **nonlinear dynamics**: a Carnot-based HVAC coefficient of
performance (COP) that degrades with temperature lift, and a three-term
infiltration model (conduction + power-law wind + buoyancy stack effect).

---

## Project Layout

```
AcRL/
├── envs/                # Simulator + Gymnasium environment + data loaders
├── training/            # Train PPO / A2C / SAC / TD3 with Stable-Baselines3
├── evaluation/          # Generalisation tests, CSV/NPZ logging, report plots
├── Data/                # Raw weather + pricing inputs (see "Data" below)
└── results/             # Generated artefacts (gitignored)
```

---

## Data

All training inputs live under `Data/`. Two kinds of files are used.

### 1. Weather — EPA PRZM `.hNN` hourly archives

| Folder | Station | WBAN | Files | Role |
|--------|---------|------|-------|------|
| `Data/Data_Syracuse_train/` | Syracuse NY | 14771 | `w14771.h89`, `w14771.h90`, `w14771.txt` (header) | Training |
| `Data/Data_Albany_test/`    | Albany NY   | 14735 | `w14735.h89`, `w14735.h90`, `w14735.txt` (header) | Out-of-distribution test |

Each `.hNN` file is fixed-width ASCII, one row per hour, two years per station
(1989 + 1990) ≈ 17,520 rows total. The parser (`envs/weather.py`) reads only
the columns it needs:

| Column (1-indexed) | Field | Unit | Used as |
|---|---|---|---|
| 2 – 11  | Date `yyyy-mm-dd` | — | timestamp + `month`, `doy` |
| 12 – 14 | Hour 1–24 (25 = daily total, skipped) | — | `hour_of_day` |
| 30 – 34 | Global Horizontal Irradiance | Wh/m² | $GHI$ → solar gain |
| 65 – 69 | Dry-bulb temperature | °C | $T_{out}$ → envelope heat exchange |
| 96 – 100 | Wind speed at 10 m | m/s | $v$ → wind-driven infiltration |

Missing values (`---`) are forward- then back-filled, GHI and wind are clipped
to ≥ 0. The resulting DataFrame is consumed by `ThermalZoneSimulator`, which
indexes 24-hour windows from it for each episode.

### 2. Pricing — New York monthly residential rates

`Data/Pricing/newyork_monthly.txt` is a tab-separated table:

```
                2025  2024  2023 ...
January         25.3  23.5  23.6 ...
February        26.2  24.3  24.2 ...
...
```

`envs/pricing.py` parses this into a `(12,)` NumPy array (¢/kWh, 0=Jan…11=Dec).
The default uses 2025 rates; older years remain available as a fallback. The
weather data is from 1989–90 but is paired with modern rates because the
agent needs to learn against today's price signal.

### How the data flows into the simulation

```
┌────────────────────┐    ┌─────────────────────┐
│ .hNN weather files │    │ newyork_monthly.txt │
└─────────┬──────────┘    └──────────┬──────────┘
          │ load_hnn_multi()         │ load_monthly_prices(year=2025)
          ▼                          ▼
   pd.DataFrame                 np.ndarray (12,) ¢/kWh
   (T_out, GHI, wind,                  │
    month, doy, hour)                  │
          │                            │
          ▼                            │
ThermalZoneSimulator                   │
  ├─ pre-buckets indices by season     │
  ├─ on reset(): draws (C, U), picks   │
  │   season, slices a 24 h window     │
  └─ on step(a): integrates the ODE    │
          │                            │
          ▼                            ▼
        HVACControlEnv ◄───── monthly_prices[month-1] enters
        (Gymnasium API)        the observation AND the reward
```

For every step the simulator looks up its current hour-of-archive row and
uses $T_{out}$, $GHI$, $v$ to advance the indoor temperature. The pricing
array is indexed by the current month so the reward and the observation
always reflect the season-appropriate electricity rate.

---

## Equations

### Thermal dynamics (Euler-forward, $\Delta t = 300$ s)

$$
T_{in}^{k+1} = T_{in}^{k} + \frac{\Delta t}{C}\Big[\,U_{eff}\,(T_{out} - T_{in}) + \alpha\,GHI - Q_{hvac}\,\Big]
$$

with the **nonlinear effective conductance**:

$$
U_{eff} = U_{cond} + k_{wind}\,v^{\,n_{wind}} + k_{stack}\sqrt{|T_{in} - T_{out}|}
$$

| Symbol | Meaning | Default / Range |
|--------|---------|-----------------|
| $C$ | Thermal capacitance (J/°C) | $\mathcal{U}(2{\times}10^{5},\;5{\times}10^{5})$, randomised per episode |
| $U_{cond}$ | Envelope conductance (W/°C) | $\mathcal{U}(30,\;80)$, randomised per episode |
| $k_{wind},\;n_{wind}$ | Wind infiltration | $4.45,\;0.65$ |
| $k_{stack}$ | Stack-effect infiltration | $2.6$ |
| $\alpha$ | Solar aperture (m²) | $0.5$ |
| $Q_{hvac}$ | HVAC power (W) | $a \times 3000,\; a \in [-1, 1]$ |
| $T_{out},\,GHI,\,v$ | From weather data | per hour |

**Sign convention:** $Q_{hvac} > 0$ removes heat (cooling); $Q_{hvac} < 0$
adds heat (heating).

### HVAC efficiency (Carnot COP)

$$
COP_{cool} = \eta\,\frac{T_{in,K}}{\max(T_{out,K} - T_{in,K},\;\varepsilon)}, \quad
COP_{heat} = \eta\,\frac{T_{in,K}}{\max(T_{in,K} - T_{out,K},\;\varepsilon)}
$$

with $\eta = 0.4$, clipped to $[0.8,\;10]$. Mode is selected by
$\text{sign}(a)$.

### Reward

$$
R_t = -\Big(\,w_c\,|T_{in} - T_{set}|
            + w_e\,\tfrac{|a_t|}{COP_t}\,\tfrac{p_t}{p_{\max}}
            + w_s\,|a_t - a_{t-1}|\,\Big)
$$

Defaults: $w_c = 1.0,\; w_e = 0.1,\; w_s = 0.05$. The energy term is both
COP- and price-scaled, so the agent learns *when* to run HVAC, not just *how
much*.

### Observation (14-dim float32)

| Idx | Feature | Idx | Feature |
|-----|---------|-----|---------|
| 0 | $T_{in} - T_{set}$ | 7 | $GHI / 1200$ |
| 1 | $(T_{out} - 20) / 20$ | 8 | $v / 20$ |
| 2 | $\sin(2\pi h / 24)$ | 9 | $C / C_{nom}$ |
| 3 | $\cos(2\pi h / 24)$ | 10 | $U_{cond} / U_{cond,nom}$ |
| 4 | $\sin(2\pi\,doy / 365)$ | 11 | previous action |
| 5 | $\cos(2\pi\,doy / 365)$ | 12 | $COP / 10$ |
| 6 | $p_t / p_{\max}$ | 13 | $U_{eff} / 150$ |

### Action

A single continuous value $a \in [-1, 1]$, mapped linearly to
$Q_{hvac} = 3000\,a$ W (negative = heat, positive = cool).

### Episode

288 steps × 5 minutes = 24 hours. On each `reset()`:

1. Draw $C \sim \mathcal{U}(C_{\min}, C_{\max})$ and
   $U_{cond} \sim \mathcal{U}(U_{\min}, U_{\max})$.
2. Advance the season cycle (Winter → Spring → Summer → Fall, round-robin).
3. Pick a uniformly random 24 h window inside that season's index bucket.
4. Randomise $T_{in}$ within $T_{set} \pm 4\,°\text{C}$.

This guarantees equal seasonal coverage across episodes and prevents the
agent from overfitting to the building parameters or the weather window.

---

## Setup

```bash
pip install gymnasium "stable-baselines3[extra]" numpy pandas matplotlib
```

Python ≥ 3.9 required.

## Usage

```bash
# Train one or more agents (≈ 30–60 min each, 500 k steps)
python training/train_ppo.py
python training/train_a2c.py
python training/train_sac.py
python training/train_td3.py

# Evaluate on Albany (out-of-distribution city)
python evaluation/test_ppo.py
python evaluation/test_a2c.py
python evaluation/test_sac.py
python evaluation/test_td3.py

# Generate the report figures
python -m evaluation.report_plots
```

Each training script writes `best_model.zip`, `final_model.zip`,
`vecnormalize.pkl`, and `evaluations.npz` to `results/<ALGO>_nl/`. Each test
script writes `generalization_stats_<algo>_<city>.csv` and
`generalization_traces_<algo>_<city>.npz` to the same folder.
`report_plots.py` writes four PNGs to `results/plots/`.

---

## Module Reference

### `envs/`

| Module | Public API |
|--------|------------|
| `weather.py` | `load_hnn(path)`, `load_hnn_multi(paths)` → DataFrame with `T_out`, `GHI`, `wind_speed`, `month`, `doy`, `hour_of_day` |
| `pricing.py` | `load_monthly_prices(path, year=2025)` → `np.ndarray` shape `(12,)`, ¢/kWh |
| `simulator.py` | `ThermalZoneSimulator` with `reset(seed)`, `step(action)`, `get_state()` |
| `environment.py` | `HVACControlEnv` (Gymnasium env), `make_train_env()` (Syracuse), `make_test_env()` (Albany) |

### `training/`

| File | Algorithm | Vec env | Reward norm | Notes |
|------|-----------|---------|-------------|-------|
| `train_ppo.py` | PPO | SubprocVecEnv × 8 | yes | n_steps=2048, batch=64, lr=3e-4, γ=0.98 |
| `train_a2c.py` | A2C | SubprocVecEnv × 8 | yes | n_steps=5, RMSProp, CPU |
| `train_sac.py` | SAC | DummyVecEnv × 1 | no | replay 1 M, batch 256, auto entropy |
| `train_td3.py` | TD3 | DummyVecEnv × 1 | no | replay 1 M, NormalActionNoise σ=0.1 |
| `callbacks.py` | `SyncNormCallback` keeps eval-env normalisation stats in sync with the training env |

All trainers run for 500 k environment steps with `EvalCallback` saving the
best model every 10 k steps.

### `evaluation/`

| Module | Purpose |
|--------|---------|
| `generalization.py` | `run_generalization_test(...)` and `run_train_vs_test(...)` — load the trained policy + VecNormalize stats, run 100 deterministic 24 h episodes (25 per season) on a chosen city, save CSV + trace NPZ |
| `report_plots.py` | Reads CSVs/NPZs and writes the four report figures: Albany OOD heatmap, Syracuse → Albany gap, best-policy traces (winter + summer), training convergence |
| `test_<algo>.py` | Thin wrapper around `run_train_vs_test` for that algorithm |

CSV columns:
`episode, season, month, doy, mae, rmse, violations_pct, within_1c_pct, energy_kwh, reward, mean_slew, peak_power_w`.

NPZ trace arrays (per episode):
`time, indoor, outdoor, power, action, ghi, wind, month, doy`.

---

## Reproducing the Reported Numbers

Train all four algorithms, then run all four test scripts, then
`python -m evaluation.report_plots`. Expected Albany medians:

| Agent | MAE (°C) | Comfort (%) | Energy (kWh/day) |
|-------|----------|-------------|------------------|
| PPO   | 0.45 | 64.9 | 21.7 |
| A2C   | 0.16 | 99.7 | 18.5 |
| SAC   | 0.17 | 99.0 | 21.0 |
| **TD3** | **0.12** | **99.7** | **16.9** |
| PID baseline | 1.01 | 63.2 | 42.7 |

> Trained models, CSVs, NPZ traces and TensorBoard event files are excluded
> by `.gitignore`. Re-run the commands above to regenerate them.

---

## References

- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [Gymnasium](https://gymnasium.farama.org/)
- [EPA SAMSON / PRZM Meteorological Data](https://www.epa.gov/ceam/meteorological-data-samson-and-related-files)
