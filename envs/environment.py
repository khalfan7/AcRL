from __future__ import annotations
from pathlib import Path
from typing import Optional
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from envs.simulator import ThermalZoneSimulator
from envs.weather import load_hnn_multi
from envs.pricing import load_monthly_prices
_ROOT = Path(__file__).resolve().parent.parent
_TRAIN_DIR = _ROOT / 'Data' / 'Data_Syracuse_train'
_TEST_DIR = _ROOT / 'Data' / 'Data_Albany_test'
_PRICE_FILE = _ROOT / 'Data' / 'Pricing' / 'newyork_monthly.txt'
_TRAIN_FILES = sorted(_TRAIN_DIR.glob('*.h[0-9][0-9]'))
_TEST_FILES = sorted(_TEST_DIR.glob('*.h[0-9][0-9]'))

class HVACControlEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, simulator: ThermalZoneSimulator, monthly_prices: np.ndarray, comfort_weight: float=1.0, energy_weight: float=0.1, slew_weight: float=0.05, render_mode: Optional[str]=None):
        super().__init__()
        self.simulator = simulator
        self._prices = monthly_prices
        self._max_price = float(monthly_prices.max())
        self.comfort_weight = comfort_weight
        self.energy_weight = energy_weight
        self.slew_weight = slew_weight
        self.render_mode = render_mode
        self._prev_action: float = 0.0
        self._current_price: float = float(monthly_prices.mean())
        self._solar_max = 1200.0
        self._wind_max = 20.0
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        low = np.array([-50, -2, -1, -1, -1, -1, 0, 0, 0, 0.2, 0.2, -1, 0.0, 0.0], dtype=np.float32)
        high = np.array([50, 2, 1, 1, 1, 1, 1, 1, 1, 1.8, 1.8, 1, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, seed: Optional[int]=None, options=None):
        super().reset(seed=seed)
        self.simulator.reset(seed=seed)
        self._prev_action = 0.0
        state = self.simulator.get_state()
        self._current_price = float(self._prices[state['month'] - 1])
        return (self._get_observation(), self._get_info())

    def step(self, action):
        a = float(np.clip(action[0], -1.0, 1.0))
        indoor_temp, weather, done = self.simulator.step(a)
        self._current_price = float(self._prices[weather['month'] - 1])
        reward = self._compute_reward(indoor_temp, a)
        obs = self._get_observation()
        self._prev_action = a
        return (obs, reward, done, False, self._get_info())

    def render(self):
        if self.render_mode == 'human':
            s = self.simulator.get_state()
            print(f't={s['elapsed_hours']:.2f}h | T_in={s['indoor_temp']:.2f}°C | err={s['temp_error']:+.2f}°C | HVAC={s['hvac_power']:.0f}W | GHI={s['GHI']:.0f}Wh/m² | wind={s['wind_speed']:.1f}m/s | price={self._current_price:.1f}¢/kWh | C×{s['C_ratio']:.2f}  U×{s['U_ratio']:.2f}')

    def close(self):
        pass

    def _compute_reward(self, indoor_temp: float, action: float) -> float:
        comfort = abs(indoor_temp - self.simulator.setpoint_temp)
        energy = abs(action) / self.simulator.current_cop * (self._current_price / self._max_price)
        slew = abs(action - self._prev_action)
        return -(self.comfort_weight * comfort + self.energy_weight * energy + self.slew_weight * slew)

    def _get_observation(self) -> np.ndarray:
        sim = self.simulator
        state = sim.get_state()
        temp_error = sim.indoor_temp - sim.setpoint_temp
        outdoor_norm = float(np.clip((state['outdoor_temp'] - 20.0) / 20.0, -2.0, 2.0))
        h = state['elapsed_hours'] % 24.0
        sin_hour = float(np.sin(2.0 * np.pi * h / 24.0))
        cos_hour = float(np.cos(2.0 * np.pi * h / 24.0))
        doy = state['doy']
        sin_year = float(np.sin(2.0 * np.pi * doy / 365.0))
        cos_year = float(np.cos(2.0 * np.pi * doy / 365.0))
        price_norm = float(self._current_price / self._max_price)
        solar_norm = float(np.clip(state['GHI'] / self._solar_max, 0.0, 1.0))
        wind_norm = float(np.clip(state['wind_speed'] / self._wind_max, 0.0, 1.0))
        return np.array([temp_error, outdoor_norm, sin_hour, cos_hour, sin_year, cos_year, price_norm, solar_norm, wind_norm, sim.C_ratio, sim.U_ratio, self._prev_action, float(np.clip(sim.current_cop / 10.0, 0.0, 1.0)), float(np.clip(sim.current_U_eff / 150.0, 0.0, 1.0))], dtype=np.float32)

    def _get_info(self) -> dict:
        info = self.simulator.get_state()
        info['current_price_cents_kwh'] = self._current_price
        return info

def _load_simulator(weather_files: list[Path], **sim_kwargs) -> ThermalZoneSimulator:
    df = load_hnn_multi(weather_files)
    return ThermalZoneSimulator(weather_df=df, **sim_kwargs)

def make_train_env(**env_kwargs) -> HVACControlEnv:
    sim = _load_simulator(_TRAIN_FILES)
    prices = load_monthly_prices(_PRICE_FILE, year=2025)
    return HVACControlEnv(simulator=sim, monthly_prices=prices, **env_kwargs)

def make_test_env(**env_kwargs) -> HVACControlEnv:
    sim = _load_simulator(_TEST_FILES)
    prices = load_monthly_prices(_PRICE_FILE, year=2025)
    return HVACControlEnv(simulator=sim, monthly_prices=prices, **env_kwargs)
try:
    gym.register(id='HVACControl-v0', entry_point='envs.environment:make_train_env', max_episode_steps=288)
except Exception:
    pass
