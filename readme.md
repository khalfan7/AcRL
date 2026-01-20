# DRL-Based HVAC Control for a Single-Zone Smart Building

### ML709 Project Requirements

| Requirement | Implementation |
|-------------|----------------|
| **Thermal Model** | Lumped-capacitance (RC network) model in `thermal_simulator.py` |
| **RL Environment** | Gymnasium-compatible `HVACControlEnv` in `hvac_environment.py` |
| **State Space** | Indoor temp error, outdoor temp (normalized) |
| **Action Space** | Continuous HVAC power level [-1, 1] |
| **Reward Function** | Negative absolute temperature error (comfort penalty) |
| **RL Algorithms** | PPO, SAC, TD3, A2C (continuous action algorithms) |
| **Weather Patterns** | Configurable outdoor temp profiles (sinusoidal diurnal cycle) |

---

## Thermal Model

Uses a **lumped-capacitance model** (1st law of thermodynamics):

$$\frac{dT_{in}}{dt} = \frac{1}{mC} \left[ K(T_{out} - T_{in}) - Q_{hvac} \right]$$

| Symbol | Variable | Default |
|--------|----------|---------|
| $mC$ | `thermal_capacitance` | 300,000 J/°C |
| $K$ | `heat_transfer_coeff` | 50 W/°C |
| $Q_{hvac}$ | `max_hvac_power` | ±3000 W |

---

## Project Structure

```
HVAC-RL/
├── thermal_simulator.py   # Physics engine
├── hvac_environment.py    # Gymnasium environment
├── train_ppo.py           # PPO training
├── train_sac.py           # SAC training
├── train_td3.py           # TD3 training
├── train_a2c.py           # A2C training
└── readme.md
```

---

## Code Explanation

### `thermal_simulator.py`

**ThermalZoneSimulator** - Physics engine for a single thermal zone.

| Method | Description |
|--------|-------------|
| `__init__()` | Initialize thermal parameters and compute `max_steps = simulation_duration / control_timestep` |
| `reset(seed)` | Reset indoor temp to setpoint, regenerate outdoor temp profile, reset step counter |
| `_generate_outdoor_temp_profile()` | Create sinusoidal outdoor temp array: `T_base + amplitude * sin(2π * hour/24)` with noise |
| `step(action)` | Apply Euler integration of thermal ODE. Action ∈ [-1,1] scales HVAC power. Returns (indoor_temp, outdoor_temp, done) |
| `get_state()` | Return dict with all simulation state variables |

**Key Physics (step method):**
```python
dT = (K * (T_out - T_in) - hvac_power) / thermal_capacitance * dt
T_in += dT
```

### `hvac_environment.py`

**HVACControlEnv** - Gymnasium wrapper for RL training.

| Component | Specification |
|-----------|---------------|
| `action_space` | `Box(-1, 1, shape=(1,))` - continuous HVAC control |
| `observation_space` | `Box([-50,-1], [50,1], shape=(2,))` - [temp_error, outdoor_norm] |
| Reward | `-abs(indoor_temp - setpoint)` - minimize comfort deviation |
| Episode length | 288 steps (24h at 5-min intervals) |

| Method | Description |
|--------|-------------|
| `reset(seed)` | Reset simulator, return (observation, info) |
| `step(action)` | Step simulator, compute reward, return (obs, reward, terminated, truncated, info) |
| `_get_obs()` | Compute observation: [temp_error, normalized_outdoor_temp] |
| `_get_info()` | Return simulator state dict |
| `render()` | Print current state if render_mode='human' |

**Observation Construction:**
```python
error = indoor_temp - setpoint_temp
outdoor_norm = (outdoor_temp - outdoor_base) / amplitude
obs = [error, clip(outdoor_norm, -1, 1)]
```

### Training Scripts (`train_*.py`)

All training scripts follow the same pattern:

1. **Environment Setup**: Create vectorized env with `SubprocVecEnv` (parallel)
2. **Normalization**: Wrap with `VecNormalize` for observation/reward scaling
3. **Callbacks**: `EvalCallback` saves best model during training
4. **Training**: Run for specified timesteps with TensorBoard logging
5. **Save**: Export final model and normalization stats

```python
env = SubprocVecEnv([lambda: HVACControlEnv() for _ in range(n_envs)])
env = VecNormalize(env, norm_obs=True, norm_reward=True)
model = PPO("MlpPolicy", env, tensorboard_log="results/")
model.learn(total_timesteps=100000, callback=eval_callback)
```

---

## Installation

```bash
pip install stable-baselines3 gymnasium numpy
```

## Usage

### Training

```bash
python train_ppo.py   # PPO
python train_sac.py   # SAC
python train_td3.py   # TD3
python train_a2c.py   # A2C
```

### Monitor Training

```bash
tensorboard --logdir results/
```

### Using the Environment

```python
from hvac_environment import HVACControlEnv

env = HVACControlEnv()
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

### Load Trained Model

```python
from stable_baselines3 import PPO
from hvac_environment import HVACControlEnv

env = HVACControlEnv()
model = PPO.load("results/PPO/final_model")
obs, info = env.reset()

for _ in range(288):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        break
```

---

## Results

Training outputs saved to `results/<algorithm>/`:
- `final_model.zip` - Trained model
- `best_model.zip` - Best model (from EvalCallback)
- `vecnormalize.pkl` - Normalization stats

---

## References

- [StableBaselines3](https://stable-baselines3.readthedocs.io/)
- [Gymnasium](https://gymnasium.farama.org/)
