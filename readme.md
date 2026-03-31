# DRL-Based HVAC Control for a Single-Zone Smart Building

Deep reinforcement learning agents (PPO, SAC, TD3, A2C) that learn to regulate
indoor temperature of a physics-based thermal zone using **real EPA hourly
weather data**, **electricity pricing**, and **domain randomisation** — balancing
thermal comfort, energy cost, and actuator smoothness.

---

## End-to-End Pipeline

The diagram below shows how every component connects, from raw data to a
trained, evaluated policy.

```
┌─────────────────────────── DATA INGESTION ───────────────────────────┐
│                                                                      │
│  EPA .hNN files                  NY Rate Table                       │
│  (Syracuse / Albany)             newyork_monthly.txt                  │
│        │                                │                            │
│        ▼                                ▼                            │
│  ┌──────────────┐              ┌──────────────────┐                  │
│  │ weather.py   │              │   pricing.py     │                  │
│  │ load_hnn_    │              │ load_monthly_    │                  │
│  │ multi()      │              │ prices()         │                  │
│  └──────┬───────┘              └────────┬─────────┘                  │
│         │  DataFrame: T_out,            │  ndarray (12,): ¢/kWh     │
│         │  GHI, wind, month, doy        │                            │
└─────────┼───────────────────────────────┼────────────────────────────┘
          │                               │
          ▼                               ▼
┌─────────────────────────── SIMULATOR ────────────────────────────────┐
│                                                                      │
│  ThermalZoneSimulator  (envs/simulator.py)                           │
│  ─────────────────────────────────────────                           │
│  • Stores full 2-year weather archive                                │
│  • Pre-builds season index buckets (W / Sp / Su / F)                 │
│  • On reset():                                                       │
│      1. Draw C ~ U(200k, 500k),  U ~ U(30, 80)   [domain random.]   │
│      2. Pick season via round-robin cycle          [stratified]       │
│      3. Random 24h window within that season       [episode]          │
│      4. Randomise T_in ±4 °C around setpoint                        │
│  • On step(action):                                                  │
│      Euler forward:                                                  │
│        T_in += (Δt/C)·[U_eff·(T_out−T_in) + α·GHI − Q_hvac]       │
│      where U_eff = U·(1 + k_w·wind)                                 │
│                                                                      │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────── GYMNASIUM ENV ────────────────────────────┐
│                                                                      │
│  HVACControlEnv  (envs/environment.py)                               │
│  ─────────────────────────────────────                               │
│  • Wraps simulator, adds pricing                                     │
│  • Builds 12-dim observation   (see Observation Space below)         │
│  • Computes 3-term reward      (see Reward Function below)           │
│  • Factories: make_train_env() / make_test_env()                     │
│                                                                      │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
┌─────── TRAINING ────────┐          ┌──────── EVALUATION ─────────┐
│ training/train_*.py     │          │ evaluation/test_*.py        │
│                         │          │                             │
│ 1. VecEnv + VecNorm     │          │ 1. Load best_model.zip      │
│ 2. SB3 algorithm        │  save    │ 2. Load vecnormalize.pkl    │
│ 3. 500k steps           │───────►  │ 3. 100 episodes on Albany   │
│ 4. EvalCallback saves   │  .zip    │    (25/season, stratified)  │
│    best_model.zip       │  .pkl    │ 4. Per-season stats + CSV   │
│    vecnormalize.pkl     │          │ 5. Boxplots + season traces │
└─────────────────────────┘          └─────────────────────────────┘
```

### Pipeline Steps (what happens when you run the project)

| Step | What runs | What it does |
|------|-----------|--------------|
| **0 — Setup** | `pip install ...` | Install Gymnasium, Stable-Baselines3, NumPy, pandas, matplotlib |
| **1 — Parse weather** | `envs/weather.py` | Reads EPA fixed-width `.hNN` files → DataFrame with `T_out`, `GHI`, `wind_speed`, `month`, `doy`, `hour_of_day`. Two years (h89 + h90) concatenated = ~17,520 rows |
| **2 — Load pricing** | `envs/pricing.py` | Reads `newyork_monthly.txt` → shape-(12,) array of monthly electricity rates (¢/kWh) |
| **3 — Build simulator** | `envs/simulator.py` | Wraps the weather DataFrame into a physics engine: RC thermal ODE with solar gain, wind effects, domain randomisation, and stratified seasonal sampling |
| **4 — Wrap as Gym env** | `envs/environment.py` | Adds observation engineering (12-dim), cost-aware reward, Gymnasium API. Factory functions wire everything together |
| **5 — Train** | `training/train_*.py` | Vectorised envs + VecNormalize + SB3 algorithm. 500k steps with eval callbacks. Saves `best_model.zip` + `vecnormalize.pkl` |
| **6 — Evaluate** | `evaluation/test_*.py` | Loads trained model, runs **100 episodes** (25 per season via stratified round-robin) on **Albany** (unseen city). Reports overall + per-season stats, saves CSV, boxplots, and representative 24h traces |
| **7 — Compare** | `evaluation/plot_convergence.py` | Reads `evaluations.npz` from each algorithm, plots learning curves side-by-side |

---

## Step 1 — Designing the Simulator

The simulator (`envs/simulator.py`) is the most critical component — the RL
agent's **entire understanding of the world** comes from interacting with it.
Getting the physics and episode design right determines whether the policy
transfers to reality.

### 1.1  Governing Equation

First-law lumped-capacitance (RC) ODE with solar gain and wind-modified
conductance, solved with Euler forward integration at each 5-minute control
step ($\Delta t = 300$ s):

$$\frac{dT_{in}}{dt} = \frac{1}{C}\bigl[U_{eff}(T_{out} - T_{in}) + \alpha \cdot GHI - Q_{hvac}\bigr]$$

where $U_{eff} = U \cdot (1 + k_w \cdot v_{wind})$.

| Symbol | Parameter | Default / Range | Physical meaning |
|--------|-----------|-----------------|------------------|
| $C$ | `thermal_capacitance` | $\sim\mathcal{U}(200\,000,\;500\,000)$ J/°C | Thermal mass of the zone (walls, air, furniture) |
| $U$ | `heat_transfer_coeff` | $\sim\mathcal{U}(30,\;80)$ W/°C | Envelope conductance (randomised per episode) |
| $U_{eff}$ | effective conductance | — | Wind-modified: higher wind → more infiltration / convection |
| $k_w$ | `wind_coeff` | 0.05 s/m | Wind speed amplification factor |
| $\alpha$ | `solar_gain_coeff` | 0.5 m² | Effective aperture × absorptance |
| $GHI$ | Global Horizontal Irradiance | from weather (Wh/m²) | Solar heat gain through glazing |
| $Q_{hvac}$ | `action × max_hvac_power` | ±3 000 W | Controlled input from HVAC |
| $T_{out}$ | dry-bulb temperature | from weather (°C) | Disturbance signal |
| $T_{in}$ | `indoor_temp` | randomised ±4 °C | Controlled variable |

**Why these choices?**
- The RC model is simple enough for fast simulation (~1 μs per step) but
  captures the dominant thermal dynamics of a single zone.
- Solar gain and wind effects add realism without requiring a full EnergyPlus
  co-simulation — the agent must learn that sunny afternoons cause overheating
  and that windy nights accelerate heat loss.
- Sign convention: $Q_{hvac} > 0$ = cooling, $Q_{hvac} < 0$ = heating.

**Discretised step** (what the code computes each call to `step()`):

$$T_{in}^{k+1} = T_{in}^{k} + \frac{\Delta t}{C}\bigl[U_{eff}(T_{out}^{k} - T_{in}^{k}) + \alpha \cdot GHI^{k} - Q_{hvac}^{k}\bigr]$$

### 1.2  Real Weather Data

Hourly observations from EPA PRZM `.hNN` fixed-width files, parsed by
`envs/weather.py`:

| Field | EPA Column | Unit | Use in model |
|-------|-----------|------|--------------|
| Dry-bulb temperature | cols 65–69 | °C | $T_{out}$ — drives envelope heat exchange |
| Global Horizontal Irradiance | cols 30–34 | Wh/m² | $GHI$ — solar gain through windows |
| Wind speed (10 m) | cols 96–100 | m/s | Modifies $U_{eff}$ — wind increases heat loss |

Each archive spans **two full years** (1989–1990), giving ~17,520 hourly
rows per station. The parser concatenates both `.h89` and `.h90` into a single
DataFrame that the simulator indexes by row number.

### 1.3  Domain Randomisation

To make the policy robust to building-parameter uncertainty the simulator
draws fresh values of $C$ and $U$ at the start of **every episode**:

$$C \sim \mathcal{U}(C_{min},\;C_{max}) \qquad U \sim \mathcal{U}(U_{min},\;U_{max})$$

The agent observes normalised ratios $C/C_{nominal}$ and $U/U_{nominal}$ in
its state vector — it can *see* the current building characteristics and
adapt. This is the sim-to-real transfer mechanism: if the real building's C
and U fall within the trained range, the policy should generalise.

### 1.4  Stratified Seasonal Sampling

Weather is seasonal — temperature ranges in January (−20 °C) are nothing
like July (+35 °C). Purely random episode sampling would produce a roughly
uniform distribution across months, but clustering can still leave seasonal
gaps in a finite training budget.

**Solution — round-robin season cycling.**  At initialisation the simulator
pre-computes four index buckets:

| Quartile | Months | Typical bucket size |
|----------|--------|---------------------|
| **Winter** (Q0) | Dec, Jan, Feb | ~4,300 valid start indices |
| **Spring** (Q1) | Mar, Apr, May | ~4,400 |
| **Summer** (Q2) | Jun, Jul, Aug | ~4,400 |
| **Fall** (Q3) | Sep, Oct, Nov | ~4,400 |

On each `reset()` the simulator:

1. Selects the **next season** in a deterministic cycle: W → Sp → Su → F → W → …
2. Picks a **random starting hour** within that season's bucket.
3. Extracts the 24-hour window starting there.

This guarantees the agent sees all four seasons **equally** over any block of
four consecutive episodes, while still randomising the specific day within
each season. The result: the agent cannot overfit to summer-dominated data or
miss rare winter extremes.

### 1.5  Episode Lifecycle (what `reset` + `step` do)

```
reset(seed)
  │
  ├─ 1. Create episode-local RNG (default_rng(seed))
  ├─ 2. Draw C, U from uniform bounds → store C_ratio, U_ratio
  ├─ 3. Season cycle: pick season quartile (round-robin)
  ├─ 4. Random start within season bucket → slice 24h from archive
  ├─ 5. Randomise T_in ∈ [setpoint ± 4 °C]
  └─ return T_in
          │
          ▼
step(action)   ← called 288 times per episode (24h ÷ 5min)
  │
  ├─ 1. Clip action to [-1, 1]
  ├─ 2. Q_hvac = action × 3000 W
  ├─ 3. Look up weather for current step (zero-order hold within hour)
  ├─ 4. U_eff = U × (1 + k_w × wind_speed)
  ├─ 5. Q_solar = α × GHI
  ├─ 6. Euler step: T_in += (Δt/C) × [U_eff×(T_out−T_in) + Q_solar − Q_hvac]
  ├─ 7. Increment step counter; check if done (step ≥ 288)
  └─ return (T_in, weather_dict, done)
```

---

## Step 2 — Designing the RL Environment

The Gymnasium wrapper (`envs/environment.py`) sits between the simulator and
the RL algorithm.  It has three jobs: **build the observation**, **compute the
reward**, and **satisfy the Gymnasium API**.

### 2.1  Observation Space (12-dimensional)

The observation is a dense `float32` vector assembled from the simulator state
plus pricing data.  Every feature is either normalised to a bounded range or
already unitless:

| Idx | Feature | Source | Range | Why included |
|-----|---------|--------|-------|--------------|
| 0 | `temp_error` | $T_{in} - T_{set}$ | \[−50, 50\] °C | Primary error signal |
| 1 | `outdoor_norm` | $(T_{out} - 20) / 20$ | \[−2, 2\] | Current heat load |
| 2 | `sin_hour` | $\sin(2\pi h / 24)$ | \[−1, 1\] | Cyclic time-of-day |
| 3 | `cos_hour` | $\cos(2\pi h / 24)$ | \[−1, 1\] | (avoids discontinuity at midnight) |
| 4 | `sin_year` | $\sin(2\pi \cdot doy / 365)$ | \[−1, 1\] | Cyclic time-of-year |
| 5 | `cos_year` | $\cos(2\pi \cdot doy / 365)$ | \[−1, 1\] | (season awareness) |
| 6 | `price_norm` | $p_{month} / p_{max}$ | \[0, 1\] | Electricity cost signal |
| 7 | `solar_norm` | $GHI / 1200$ | \[0, 1\] | Anticipate solar gain |
| 8 | `wind_norm` | $v_{wind} / 20$ | \[0, 1\] | Anticipate wind-driven loss |
| 9 | `C_ratio` | $C / C_{nominal}$ | \[≈0.57, ≈1.43\] | Building thermal mass |
| 10 | `U_ratio` | $U / U_{nominal}$ | \[≈0.55, ≈1.45\] | Building conductance |
| 11 | `prev_action` | last HVAC command | \[−1, 1\] | Enables slew penalty |

**Design rationale:**

- **Cyclic time encoding** (sin/cos pairs) eliminates the 23:59 → 00:00
  discontinuity that would confuse a neural network if raw hours were used.
- **Price in the observation** lets the agent learn price-responsive control —
  it can reduce power during expensive months and compensate during cheap ones.
- **C_ratio / U_ratio** inform the agent about the building it is currently
  controlling (varies per episode via domain randomisation).
- **prev_action** gives the agent memory of its last command so it can
  minimise the slew penalty.

### 2.2  Action Space

A single continuous value in $[-1, 1]$ (shape `(1,)`), mapped linearly to
HVAC power:

$$Q_{hvac} = a_t \times 3\,000 \;\text{W}$$

- $a_t = +1.0$ → full cooling (3 kW removed)
- $a_t = 0.0$ → HVAC off
- $a_t = -1.0$ → full heating (3 kW added)

### 2.3  Reward Function (cost-aware, 3-term)

The reward is the **only learning signal** seen by the RL algorithm.  It
is computed every step inside `HVACControlEnv._compute_reward()`:

$$\boxed{R_t = -\bigl(w_c \cdot \underbrace{|T_{in} - T_{set}|}_{\text{comfort}} + w_e \cdot \underbrace{|a_t| \cdot \tfrac{p_t}{p_{max}}}_{\text{energy cost}} + w_s \cdot \underbrace{|a_t - a_{t-1}|}_{\text{slew}}\bigr)}$$

| Term | Weight | Range | Purpose |
|------|--------|-------|---------|
| **Comfort penalty** | $w_c = 1.0$ | unbounded | Every °C of deviation is penalised |
| **Energy cost** | $w_e = 0.1$ | $[0, 1]$ | Price-scaled: $\|a_t\|$ × normalised electricity price. Higher price month → higher penalty for same power |
| **Action slew** | $w_s = 0.05$ | $[0, 2]$ | Penalises rapid actuator swings → smoother control |

**How pricing enters the reward:**  The energy term multiplies the action
magnitude by $p_t / p_{max}$ where $p_t$ is the current month's NY
residential electricity rate.  The agent receives $p_t / p_{max}$ as
observation feature `price_norm` (index 6), so it *knows* the current price
and can decide to trade a small comfort loss for significant energy savings
during expensive months.

**Weight tuning intuition:**

| Configuration | Learned behaviour |
|---|---|
| $w_e=0,\; w_s=0$ (comfort only) | Bang-bang: full power whenever any error exists |
| $w_e=0.1,\; w_s=0.05$ (default) | Proportional, smooth, price-responsive |
| $w_e \gg 1$ (energy dominant) | Near-zero action; comfort sacrificed |

**Training vs. testing:**

| Context | Uses reward? | What is reported |
|---------|-------------|------------------|
| **Training** | Yes — sole learning signal. Maximises $G_t = \sum \gamma^k R_{t+k}$ | Return (reward sum) |
| **Evaluation** | No — reward is discarded | Physical metrics: MAE, RMSE, violations, energy (Wh), peak power |

The evaluation metrics map directly to the reward terms:

| Physical metric | Corresponds to |
|---|---|
| MAE / RMSE (°C) | Comfort term |
| Total energy (Wh) | Energy cost term |
| Comfort violations (% steps outside ±0.5 °C) | Comfort term |
| Peak cooling / heating (W) | Energy + slew terms |

---

## Step 3 — Training

### 3.1  Algorithm Selection

| Algorithm | Type | # Envs | VecEnv | Reward norm | Key trait |
|-----------|------|--------|--------|-------------|-----------|
| **PPO** | On-policy | 8 | SubprocVecEnv | ✓ | Robust default; clipped surrogate objective |
| **A2C** | On-policy | 8 | SubprocVecEnv | ✓ | Synchronous A3C; fast on CPU |
| **SAC** | Off-policy | 1 | DummyVecEnv | ✗ | Entropy-regularised; auto entropy tuning |
| **TD3** | Off-policy | 1 | DummyVecEnv | ✗ | Deterministic policy; Gaussian noise σ = 0.1 |

**On-policy (PPO / A2C):** use 8 parallel environments for sample throughput
and enable reward normalisation so the critic value targets stay well-scaled.

**Off-policy (SAC / TD3):** use 1 environment (replay buffer provides
diversity); reward normalisation disabled because the buffer already
stabilises target statistics.

### 3.2  Training Script Pattern

Every `training/train_*.py` follows the same sequence.  The shared
`SyncNormCallback` lives in `training/callbacks.py` so that each script only
contains its algorithm-specific configuration:

```python
from envs import make_train_env                # Syracuse weather + NY 2025 pricing
from training.callbacks import SyncNormCallback # shared callback

# ① Vectorised training env with observation + reward normalisation
env = make_vec_env(make_train_env, n_envs=N, seed=42)
env = VecNormalize(env, norm_obs=True, norm_reward=<True/False>)

# ② Separate eval env (training=False prevents running-stat drift)
eval_env = VecNormalize(make_vec_env(...), training=False)

# ③ Callbacks
#    SyncNormCallback — copies obs running stats from train env to eval env
#                       (also syncs ret_rms when reward normalisation is on)
#    EvalCallback     — evaluates every 10k steps, saves best_model.zip
callbacks = [SyncNormCallback(env, eval_env), EvalCallback(eval_env, ...)]

# ④ Train
model = PPO("MlpPolicy", env, ...)
model.learn(total_timesteps=500_000, callback=callbacks)

# ⑤ Save artifacts
model.save("results/PPO/final_model")
env.save("results/PPO/vecnormalize.pkl")       # needed at test time
```

Similarly, the test scripts are thin wrappers that call the shared engine in
`evaluation/generalization.py`:

```python
from stable_baselines3 import PPO
from evaluation.generalization import run_generalization_test

run_generalization_test(PPO, "PPO")   # loads model, runs 100 episodes, saves outputs
```

### 3.3  Training Budget

- **500,000 environment steps** at 288 steps/episode ≈ **1,736 episodes**
- With stratified sampling (4 seasons), that's ~434 episodes per season
- Each episode sees a fresh (C, U) draw → the agent trains across ~1,736
  different "buildings" in ~1,736 different weather days

---

## Step 4 — Evaluation

### 4.1  Generalisation Test on Albany

The test scripts load the saved policy and VecNormalize stats, then run **100
deterministic 24-hour episodes** on **Albany NY** — a city the agent has
**never seen during training**.  The stratified round-robin sampling ensures
exactly **25 episodes per season** (Winter → Spring → Summer → Fall cycling),
so performance is measured uniformly across the full annual weather range.

```bash
python evaluation/test_ppo.py
```

### 4.2  Reported Metrics

Each episode produces per-episode statistics.  The console prints overall and
per-season summaries (mean ± std):

| Metric | Unit | What it tells you |
|--------|------|-------------------|
| MAE | °C | Average temperature tracking error |
| RMSE | °C | Penalises large deviations more than MAE |
| Comfort violations | % | Fraction of steps where $\|T_{in}-T_{set}\| > 0.5$ °C |
| Energy | kWh/day | Total electricity consumed per 24 h episode |
| Reward | — | Sum of step rewards (for algorithm comparison) |

### 4.3  Outputs

Each test script writes three artifacts to `results/<ALGO>/`:

| File | Description |
|------|-------------|
| `generalization_stats_<algo>.csv` | Per-episode metrics (100 rows) for cross-algorithm comparison |
| `generalization_boxplots_<algo>.png` | 2×2 boxplots of MAE, RMSE, violations%, and energy by season |
| `seasonal_profiles_<algo>.png` | 4×2 grid: representative 24h trace per season (temp + HVAC power), selected as the episode closest to the season's median MAE |

The **boxplot figure** is the primary generalization report — it shows how
performance varies by season and how tight the distribution is.  Wide boxes
or outliers indicate weather conditions the agent struggles with.

The **seasonal profiles** figure gives qualitative insight — you can see how
the agent responds to winter cold snaps vs summer heat waves, and whether it
uses heating, cooling, or both appropriately.

### 4.4  Convergence Comparison

```bash
python evaluation/plot_convergence.py
```

Reads `evaluations.npz` from each algorithm directory and plots mean eval
reward ± std over training, saved to `results/training_convergence.png`.

---

## Data

### Weather — EPA PRZM .hNN Format

| Split | Station | WBAN | Files | Purpose |
|-------|---------|------|-------|---------|
| **Train** | Syracuse NY | 14771 | `Data/Data_Syracuse_train/w14771.h89`, `.h90` | 2-year training archive |
| **Test** | Albany NY | 14735 | `Data/Data_Albany_test/w14735.h89`, `.h90` | Generalisation benchmark |

Both stations are in upstate New York (~250 km apart), sharing a climate zone
but with different micro-climates — making Albany a meaningful out-of-
distribution test.

### Electricity Pricing

`Data/Pricing/newyork_monthly.txt` — New York State 2025 residential rates
(~25–27 ¢/kWh). The weather data spans 1989–1990, but real pricing for those
years is unavailable, so 2025 rates are used as a modern proxy. Used in two
places:

1. **Observation** (index 6): agent sees current normalised price
2. **Reward**: energy penalty is scaled by normalised price

---

## Project Structure

```
AcRL/
├── readme.md
│
├── envs/                              # Environment package
│   ├── __init__.py                    #   exports HVACControlEnv, make_train_env, make_test_env
│   ├── environment.py                 #   Gymnasium env: 12-dim obs, cost-aware reward
│   ├── simulator.py                   #   RC thermal model, domain randomisation, stratified sampling
│   ├── weather.py                     #   EPA .hNN hourly weather parser
│   └── pricing.py                     #   Monthly electricity price loader
│
├── training/                          # Training scripts
│   ├── __init__.py
│   ├── callbacks.py                   #   SyncNormCallback (shared by all trainers)
│   ├── train_ppo.py
│   ├── train_sac.py
│   ├── train_td3.py
│   └── train_a2c.py
│
├── evaluation/                        # Testing & visualisation
│   ├── __init__.py
│   ├── generalization.py              #   100-episode evaluation engine (shared by all testers)
│   ├── test_ppo.py
│   ├── test_sac.py
│   ├── test_td3.py
│   ├── test_a2c.py
│   └── plot_convergence.py
│
├── Data/
│   ├── Data_Syracuse_train/           # EPA .hNN files (1989–1990)
│   ├── Data_Albany_test/              # EPA .hNN files (1989–1990)
│   └── Pricing/
│       └── newyork_monthly.txt        # NY residential electricity rates
│
└── results/                           # Generated during training/evaluation
    ├── PPO/                           #   best_model.zip, vecnormalize.pkl, evaluations.npz,
    │                                  #   generalization_stats_ppo.csv,
    │                                  #   generalization_boxplots_ppo.png,
    │                                  #   seasonal_profiles_ppo.png
    ├── SAC/
    ├── TD3/
    └── A2C/
```

---

## Setup & Usage

### Prerequisites

```
Python >= 3.9
gymnasium
stable-baselines3[extra]   # includes tensorboard, matplotlib
numpy
pandas
matplotlib
```

```bash
pip install gymnasium "stable-baselines3[extra]" numpy pandas matplotlib
```

### Quick Start

```bash
# 1. Train (pick any algorithm)
python training/train_ppo.py

# 2. Evaluate on Albany
python evaluation/test_ppo.py

# 3. Compare all algorithms
python evaluation/plot_convergence.py

# 4. TensorBoard (optional)
tensorboard --logdir results/
```

---

## References

- [Stable-Baselines3 docs](https://stable-baselines3.readthedocs.io/)
- [Gymnasium docs](https://gymnasium.farama.org/)
- [EPA Meteorological Data (SAMSON/PRZM format)](https://www.epa.gov/ceam/meteorological-data-samson-and-related-files)
