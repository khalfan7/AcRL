"""
HVACControlEnv — Gymnasium wrapper for ThermalZoneSimulator
============================================================
Uses real EPA hourly weather data (Syracuse for training, Albany for testing)
with domain-randomised building parameters per episode.

Observation  (14-dim, float32)
──────────────────────────────
  [0]  temp_error    °C        T_in − T_set
  [1]  outdoor_norm            (T_out − 20) / 20  clipped ∈ [−2, 2]
  [2]  sin_hour                sin(2π · h / 24)
  [3]  cos_hour                cos(2π · h / 24)
  [4]  sin_year                sin(2π · doy / 365)
  [5]  cos_year                cos(2π · doy / 365)
  [6]  price_norm              price / max_price          ∈ [0, 1]
  [7]  solar_norm              GHI / 1200                 ∈ [0, 1]
  [8]  wind_norm               wind_speed / 20            ∈ [0, 1]
  [9]  C_ratio                 C / C_nominal              ∈ [≈0.57, ≈1.43]
  [10] U_ratio                 U / U_nominal              ∈ [≈0.55, ≈1.45]
  [11] prev_action             previous HVAC command      ∈ [−1, 1]
  [12] COP_norm                COP / 10.0                 ∈ [0.08, 1.0]
  [13] U_eff_norm              U_eff / 150.0              ∈ [0.0, 1.0]

Reward  (cost-aware, COP-adjusted)
──────────────────────────────────
  R = −(w_c · |T_in − T_set|
        + w_e · (|a_t| / COP) · (price / max_price)
        + w_s · |a_t − a_{t−1}|)

  Default weights: w_c = 1.0, w_e = 0.1, w_s = 0.05
  • comfort term  dominates  (°C error)
  • energy term   is a price-scaled regulariser on action magnitude
  • slew   term   penalises rapid actuator changes (smoother control)

Data splits
───────────
  Training   : Syracuse NY  (Data/Data_Syracuse_train/w14771.h89 + .h90)
  Generalisation test : Albany NY   (Data/Data_Albany_test/w14735.h89 + .h90)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.simulator import ThermalZoneSimulator
from envs.weather import load_hnn_multi
from envs.pricing import load_monthly_prices

# ── Data paths (relative to project root) ────────────────────────────────────
_ROOT       = Path(__file__).resolve().parent.parent
_TRAIN_DIR  = _ROOT / "Data" / "Data_Syracuse_train"
_TEST_DIR   = _ROOT / "Data" / "Data_Albany_test"
_PRICE_FILE = _ROOT / "Data" / "Pricing" / "newyork_monthly.txt"

_TRAIN_FILES = sorted(_TRAIN_DIR.glob("*.h[0-9][0-9]"))
_TEST_FILES  = sorted(_TEST_DIR.glob("*.h[0-9][0-9]"))


# ── Environment ──────────────────────────────────────────────────────────────
class HVACControlEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        simulator: ThermalZoneSimulator,
        monthly_prices: np.ndarray,          # shape (12,), ¢/kWh
        comfort_weight: float = 1.0,
        energy_weight:  float = 0.1,
        slew_weight:    float = 0.05,
        render_mode:    Optional[str] = None,
    ):
        super().__init__()
        self.simulator      = simulator
        self._prices        = monthly_prices
        self._max_price     = float(monthly_prices.max())
        self.comfort_weight = comfort_weight
        self.energy_weight  = energy_weight
        self.slew_weight    = slew_weight
        self.render_mode    = render_mode

        self._prev_action:   float = 0.0
        self._current_price: float = float(monthly_prices.mean())

        # Normalisation constants
        self._solar_max = 1_200.0    # Wh/m²  (typical clear-sky peak at mid-latitude)
        self._wind_max  =    20.0    # m/s

        # ── Action space ─────────────────────────────────────────────────
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # ── Observation space (14-dim) ───────────────────────────────────
        #         [err, out, sin_h, cos_h, sin_y, cos_y, price, solar, wind, C, U, prev_a, cop, u_eff]
        low  = np.array([-50, -2, -1, -1, -1, -1,  0,  0,  0,  0.2, 0.2, -1,  0.0, 0.0], dtype=np.float32)
        high = np.array([ 50,  2,  1,  1,  1,  1,  1,  1,  1,  1.8, 1.8,  1,  1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        self.simulator.reset(seed=seed)
        self._prev_action = 0.0
        state = self.simulator.get_state()
        self._current_price = float(self._prices[state["month"] - 1])
        return self._get_observation(), self._get_info()

    def step(self, action):
        a = float(np.clip(action[0], -1.0, 1.0))
        indoor_temp, weather, done = self.simulator.step(a)
        self._current_price = float(self._prices[weather["month"] - 1])
        reward = self._compute_reward(indoor_temp, a)
        obs    = self._get_observation()
        self._prev_action = a
        return obs, reward, done, False, self._get_info()

    def render(self):
        if self.render_mode == "human":
            s = self.simulator.get_state()
            print(
                f"t={s['elapsed_hours']:.2f}h | "
                f"T_in={s['indoor_temp']:.2f}°C | "
                f"err={s['temp_error']:+.2f}°C | "
                f"HVAC={s['hvac_power']:.0f}W | "
                f"GHI={s['GHI']:.0f}Wh/m² | "
                f"wind={s['wind_speed']:.1f}m/s | "
                f"price={self._current_price:.1f}¢/kWh | "
                f"C×{s['C_ratio']:.2f}  U×{s['U_ratio']:.2f}"
            )

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_reward(self, indoor_temp: float, action: float) -> float:
        comfort = abs(indoor_temp - self.simulator.setpoint_temp)
        energy  = (abs(action) / self.simulator.current_cop) * (self._current_price / self._max_price)
        slew    = abs(action - self._prev_action)                         # ∈ [0, 2]
        return -(
            self.comfort_weight * comfort
            + self.energy_weight  * energy
            + self.slew_weight   * slew
        )

    def _get_observation(self) -> np.ndarray:
        sim   = self.simulator
        state = sim.get_state()

        temp_error   = sim.indoor_temp - sim.setpoint_temp
        outdoor_norm = float(np.clip((state["outdoor_temp"] - 20.0) / 20.0, -2.0, 2.0))

        # Time-of-day cyclic encoding
        h        = state["elapsed_hours"] % 24.0
        sin_hour = float(np.sin(2.0 * np.pi * h / 24.0))
        cos_hour = float(np.cos(2.0 * np.pi * h / 24.0))

        # Time-of-year cyclic encoding
        doy      = state["doy"]
        sin_year = float(np.sin(2.0 * np.pi * doy / 365.0))
        cos_year = float(np.cos(2.0 * np.pi * doy / 365.0))

        price_norm = float(self._current_price / self._max_price)
        solar_norm = float(np.clip(state["GHI"]        / self._solar_max, 0.0, 1.0))
        wind_norm  = float(np.clip(state["wind_speed"] / self._wind_max,  0.0, 1.0))

        return np.array(
            [temp_error,  outdoor_norm, sin_hour, cos_hour,
             sin_year,    cos_year,     price_norm, solar_norm,
             wind_norm,   sim.C_ratio,  sim.U_ratio, self._prev_action,
             float(np.clip(sim.current_cop / 10.0, 0.0, 1.0)),
             float(np.clip(sim.current_U_eff / 150.0, 0.0, 1.0))],
            dtype=np.float32,
        )

    def _get_info(self) -> dict:
        info = self.simulator.get_state()
        info["current_price_cents_kwh"] = self._current_price
        return info


# ── Factory functions ────────────────────────────────────────────────────────

def _load_simulator(weather_files: list[Path], **sim_kwargs) -> ThermalZoneSimulator:
    df = load_hnn_multi(weather_files)
    return ThermalZoneSimulator(weather_df=df, **sim_kwargs)


def make_train_env(**env_kwargs) -> HVACControlEnv:
    """HVACControlEnv using Syracuse NY training data (1989–1990)."""
    sim    = _load_simulator(_TRAIN_FILES)
    prices = load_monthly_prices(_PRICE_FILE, year=2025)
    return HVACControlEnv(simulator=sim, monthly_prices=prices, **env_kwargs)


def make_test_env(**env_kwargs) -> HVACControlEnv:
    """HVACControlEnv using Albany NY generalisation-test data (1989–1990)."""
    sim    = _load_simulator(_TEST_FILES)
    prices = load_monthly_prices(_PRICE_FILE, year=2025)
    return HVACControlEnv(simulator=sim, monthly_prices=prices, **env_kwargs)


# ── Gymnasium registration (guarded) ────────────────────────────────────────
try:
    gym.register(
        id="HVACControl-v0",
        entry_point="envs.environment:make_train_env",
        max_episode_steps=288,      # 24 h at 5-min intervals
    )
except Exception:
    pass  # already registered by a previous import
