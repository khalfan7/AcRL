# DRL-Based HVAC Control for a Single-Zone Smart Building

Reinforcement learning agents (PPO, SAC, TD3, A2C) trained to regulate the
indoor temperature of a lumped-capacitance thermal zone against a sinusoidal
outdoor temperature profile, balancing **thermal comfort** and **energy efficiency**.

---

## Requirements Table

| Requirement | Implementation |
|---|---|
| **Thermal Model** | Lumped-capacitance (RC) model — `thermal_simulator.py` |
| **RL Environment** | Gymnasium-compatible `HVACControlEnv` — `hvac_environment.py` |
| **State Space** | 4-dim: `[temp_error, outdoor_norm, sin_time, cos_time]` |
| **Action Space** | Continuous ∈ \[-1, 1\] → scales HVAC power ±3 000 W |
| **Reward Function** | `R = −|e_T| − λ·|a|`  (comfort + energy penalty) |
| **Algorithms** | PPO, SAC, TD3, A2C (all support continuous actions) |
| **Weather Profile** | Sinusoidal diurnal cycle, peak at 15:00 |

---

## Thermal Model — `ThermalZoneSimulator`

`thermal_simulator.py` is the **physics engine** for the entire project.  Every
RL algorithm interacts with the real world only through this class — it is the
environment's ground truth.

### Governing Equation

First-law lumped-capacitance (RC) ODE, solved with Euler forward integration
at each 5-minute control step ($\Delta t = 300$ s):

$$\frac{dT_{in}}{dt} = \frac{1}{C}\bigl[U(T_{out} - T_{in}) - Q_{hvac}\bigr]$$

| Symbol | Parameter | Default | Physical meaning |
|--------|-----------|---------|------------------|
| $C$ | `thermal_capacitance` | 300 000 J/°C | Thermal mass of the zone (walls, air, furniture) |
| $U$ | `heat_transfer_coeff` | 50 W/°C | Heat leakage through envelope |
| $Q_{hvac}$ | `action × max_hvac_power` | ±3 000 W | Net power delivered by HVAC unit |
| $T_{out}$ | outdoor temp array | sinusoidal | Disturbance driving heat gain/loss |
| $T_{in}$ | `indoor_temp` | randomised | Controlled variable |

**Sign convention** — $Q_{hvac} > 0$ removes heat (cooling); $Q_{hvac} < 0$ adds heat (heating).

**Discretised step** (what the code actually computes):

$$T_{in}^{k+1} = T_{in}^{k} + \frac{\Delta t}{C}\bigl[U(T_{out}^{k} - T_{in}^{k}) - Q_{hvac}^{k}\bigr]$$

### Outdoor Temperature Profile

Configured via `outdoor_temp_profile` (default `'sinusoidal'`):

| Mode | Behaviour |
|------|-----------|
| `'sinusoidal'` | $T_{out}(t) = T_{base} + A\sin\bigl(\tfrac{2\pi t}{24h} + \phi\bigr)$, peak at **15:00** |
| `'constant'` | Fixed at $T_{base}$ — good for unit tests |
| `'step'` | Steps up by $A$ at the episode midpoint — stress test |

### Key Simulator Internals

| Attribute / Method | Role |
|---|---|
| `max_steps = duration / dt` | Total control steps per episode (288 for 24 h @ 5 min) |
| `reset(seed)` | Creates a new `np.random.default_rng(seed)` (episode-local, no global state mutation); randomises $T_{in,0}\in[18,28]$ °C; regenerates outdoor profile |
| `step(action)` | Clips action to $[-1,1]$; sets $Q_{hvac}$; runs one Euler step; increments `current_step`; returns `(T_{in}, T_{out}, done)` |
| `get_state()` | Returns a dict `{indoor_temp, outdoor_temp, setpoint_temp, temp_error, hvac_power, current_step, elapsed_hours}` — used as `info` by the Gymnasium env |
| `elapsed_hours` (property) | `current_step × dt / 3600` — convenience accessor |

---

## Reward Function

Defined in `HVACControlEnv.step()` — the **only learning signal** seen by
the RL algorithm during training.

$$\boxed{R_t = -\underbrace{|T_{in} - T_{set}|}_{\text{comfort term}} - \lambda\underbrace{|a_t|}_{\text{energy term}}}$$

### Term-by-term breakdown

| Term | Symbol | Default weight | Purpose |
|------|--------|----------------|---------|
| Comfort penalty | $-\|T_{in}-T_{set}\|$ | 1.0 | Penalises every °C of deviation from setpoint; the primary objective |
| Energy penalty | $-\lambda\|a_t\|$ | $\lambda=0.1$ | Penalises HVAC power magnitude; discourages bang-bang control |

### Role in training vs. testing

**Training** — the reward is the *sole* feedback signal.  The algorithm
(PPO/SAC/TD3/A2C) maximises discounted cumulative return
$G_t = \sum_{k=0}^{\infty}\gamma^k R_{t+k}$ by adjusting its policy.
With $\lambda=0$ the agent maximises comfort at any energy cost.  With
$\lambda=0.1$ it learns smoother, proportional responses instead of
saturating the actuator at ±3 000 W whenever there is even a small error.

**Testing** — `test_*.py` receives the reward from `env.step()` but
**never reads it**.  Evaluation instead uses physical metrics that directly
measure the two objectives the reward was designed to proxy:

| Physical metric | What it measures | Corresponds to |
|---|---|---|
| MAE / RMSE (°C) | Temperature tracking accuracy | Comfort term |
| Total energy (Wh) | Electricity consumed | Energy term |
| Comfort violations (%) | Steps outside ±0.5 °C band | Comfort term |
| Peak cooling / heating (W) | Actuator saturation | Energy term |

### Why two terms?

| Reward shape | Learnt behaviour |
|---|---|
| $\lambda=0$ (comfort only) | Bang-bang: full cooling/heating whenever any error exists |
| $\lambda=0.1$ (default) | Proportional: power scales with error magnitude |
| $\lambda\gg 1$ (energy dominant) | Near-zero action; comfort sacrificed |

$\lambda=0.1$ keeps the energy penalty as a **regulariser** — it shapes the
action distribution without overriding the thermal comfort objective.  In a
real building this trades off electricity cost against occupant satisfaction.

---

## Observation Space  (4-dimensional)

| Index | Feature | Range | Description |
|-------|---------|-------|-------------|
| 0 | `temp_error` | \[−50, 50\] °C | $T_{in} - T_{set}$ |
| 1 | `outdoor_norm` | \[−1, 1\] | $(T_{out} - T_{base}) / A$ |
| 2 | `sin_time` | \[−1, 1\] | $\sin(2\pi \cdot h / 24)$ |
| 3 | `cos_time` | \[−1, 1\] | $\cos(2\pi \cdot h / 24)$ |

The cyclic time encoding lets the agent infer the time of day without clipping
artefacts:

| Time | sin | cos | Meaning |
|------|-----|-----|---------|
| 00:00 | 0 | +1 | midnight |
| 06:00 | +1 | 0 | dawn |
| 12:00 | 0 | −1 | noon |
| 18:00 | −1 | 0 | dusk |

---

## Project Structure

```
AcRL/
├── thermal_simulator.py     # Physics engine (RC thermal model)
├── hvac_environment.py      # Gymnasium environment wrapper
├── train_ppo.py             # PPO training script
├── train_sac.py             # SAC training script
├── train_td3.py             # TD3 training script
├── train_a2c.py             # A2C training script
├── test_ppo.py              # PPO evaluation + daily profile plot
├── test_sac.py              # SAC evaluation + daily profile plot
├── test_td3.py              # TD3 evaluation + daily profile plot
├── test_a2c.py              # A2C evaluation + daily profile plot
├── plot_convergence.py      # Side-by-side convergence comparison
├── results/
│   ├── PPO/                 # best_model.zip, vecnormalize.pkl, evaluations.npz
│   ├── SAC/
│   ├── TD3/
│   └── A2C/
└── readme.md
```

---

## Prerequisites

```
Python >= 3.9
gymnasium
stable-baselines3[extra]   # includes tensorboard, matplotlib
numpy
matplotlib
```

Install with:

```bash
pip install gymnasium "stable-baselines3[extra]" numpy matplotlib
```

---

## Usage

### 1 — Train an agent

```bash
python train_ppo.py    # or train_sac.py / train_td3.py / train_a2c.py
```

Each script trains for 500 000 environment steps, saves the best checkpoint
(by mean eval reward) to `results/<ALGO>/best_model.zip`, and exports
`vecnormalize.pkl` for inference-time observation normalisation.

### 2 — Evaluate (24-hour episode plot)

```bash
python test_ppo.py    # loads best_model or final_model automatically
```

Prints MAE, RMSE, comfort violations, peak power, and total energy, then saves
`results/<ALGO>/daily_episode_profile_<algo>.png`.

### 3 — Compare convergence across algorithms

```bash
python plot_convergence.py
```

Requires at least one `results/<ALGO>/evaluations.npz` to be present.
Saves `results/training_convergence.png`.

### 4 — TensorBoard

```bash
tensorboard --logdir results/
```

---

## Algorithm Notes

| Algorithm | Type | Envs | Reward norm | Notes |
|-----------|------|------|-------------|-------|
| **PPO** | On-policy | 8 × SubprocVecEnv | ✓ | Robust, good first choice |
| **A2C** | On-policy | 8 × SubprocVecEnv | ✓ | Fast CPU training; lower sample efficiency than PPO |
| **SAC** | Off-policy | 1 × DummyVecEnv | ✗ | Entropy-regularised; automatic entropy tuning |
| **TD3** | Off-policy | 1 × DummyVecEnv | ✗ | Deterministic; Gaussian exploration noise σ = 0.1 |

**On-policy (PPO / A2C):** reward normalisation is enabled so the critic value
targets stay well-scaled.  An in-training `SyncNormCallback` keeps the eval
env's observation running stats aligned with the training env.

**Off-policy (SAC / TD3):** reward normalisation is disabled; the replay buffer
already stabilises learning.  Observation normalisation is still used.

---

## Code Reference

### `thermal_simulator.py` — `ThermalZoneSimulator`

Self-contained physics engine.  The Gymnasium environment owns one instance
and delegates all physical state updates to it.

```
HVACControlEnv
    └── ThermalZoneSimulator   ← physics
            ├── indoor_temp        (controlled variable)
            ├── outdoor_temp_array (disturbance profile)
            ├── hvac_power         (action × max_hvac_power)
            └── current_step       (episode clock)
```

| Method | Description |
|--------|-------------|
| `__init__()` | Stores all RC parameters; pre-computes `max_steps = simulation_duration / control_timestep` |
| `reset(seed)` | Creates episode-local `np.random.default_rng(seed)` (no global seed mutation); draws $T_{in,0}\in[18,28]$ °C; calls `_generate_outdoor_temp_profile()` |
| `_generate_outdoor_temp_profile()` | Builds `outdoor_temp_array[0..max_steps]` — sinusoidal (peak 15:00), constant, or step-change |
| `step(action)` | Clips action; computes $Q_{hvac}$; one Euler step of the ODE; increments `current_step`; returns `(T_{in}, T_{out}, done)` |
| `get_state()` | Snapshot dict: `indoor_temp`, `outdoor_temp`, `setpoint_temp`, `temp_error`, `hvac_power`, `current_step`, `elapsed_hours` — passed through as Gymnasium `info` |
| `elapsed_hours` | Property: `current_step × dt / 3600` |

### `hvac_environment.py` — `HVACControlEnv`

Thin Gymnasium wrapper around `ThermalZoneSimulator`.  Its three
responsibilities are: **observation engineering**, **reward computation**,
and **Gymnasium API compliance**.

#### Observation pipeline (`_get_observation`)

```
Simulator state
  ├─ indoor_temp, setpoint_temp  →  temp_error = T_in − T_set          [-50..50 °C]
  ├─ outdoor_temp_array[step]    →  outdoor_norm = (T_out−T_base)/A    [-1..1]
  └─ elapsed_hours               →  angle = 2π·h/24
                                      sin_time = sin(angle)             [-1..1]
                                      cos_time = cos(angle)             [-1..1]
```

The four features give the agent all information it needs:
- **temp_error** — how far off-target it is right now
- **outdoor_norm** — current heat load from outside
- **sin/cos time** — where it is in the diurnal cycle (anticipatory control)

#### Reward computation (`step`)

```python
# hvac_environment.py  —  HVACControlEnv.step()
temperature_error = abs(indoor_temp - self.simulator.setpoint_temp)
energy_penalty    = self.lambda_weight * abs(action_value)
reward            = -temperature_error - energy_penalty
```

This is the **only** value the RL algorithm uses to update its weights.
The test scripts receive it from `env.step()` but discard it — they report
physical metrics (MAE, RMSE, Wh) instead.

| Method | Description |
|--------|-------------|
| `reset(seed)` | Calls `simulator.reset(seed)`; returns `(obs_4d, info_dict)` |
| `step(action)` | Steps simulator; computes two-term reward; returns Gymnasium 5-tuple |
| `_get_observation()` | Returns `np.float32` array `[temp_error, outdoor_norm, sin_time, cos_time]` |
| `_get_info()` | Returns `simulator.get_state()` dict |
| `render()` | One-line console summary when `render_mode='human'` |

### Training scripts `train_*.py`

All four follow the same pattern:

```python
# 1. Create vectorised training env + VecNormalize
env = make_vec_env(make_env, n_envs=N, seed=42)
env = VecNormalize(env, norm_obs=True, norm_reward=<True/False>)

# 2. Eval env (training=False prevents stat drift during evaluation)
eval_env = VecNormalize(make_vec_env(...), norm_obs=True, norm_reward=False, training=False)

# 3. Callbacks: sync normalisation stats + periodic evaluation
callbacks = [SyncNormCallback(env, eval_env), EvalCallback(eval_env, ...)]

# 4. Train and save
model.learn(total_timesteps=500_000, callback=callbacks)
model.save(...)
env.save("vecnormalize.pkl")
```

### Test scripts `test_*.py`

Load `best_model.zip` (or `final_model.zip`), run one deterministic 24-hour
episode, print statistics, and save a two-panel figure:

- **Top:** indoor temp vs outdoor temp vs setpoint (with ±0.5 °C comfort band)
- **Bottom:** HVAC power over time (blue = cooling, red = heating)

Note: the reward is received from `env.step()` but **not used** — evaluation
reports physical metrics that directly measure what the reward was designed to
proxy (see [Reward Function](#reward-function) above).

---

## References

- [Stable-Baselines3 docs](https://stable-baselines3.readthedocs.io/)
- [Gymnasium docs](https://gymnasium.farama.org/)
