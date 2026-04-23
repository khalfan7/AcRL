from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from numpy.random import Generator
_MONTH_TO_SEASON = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
_SEASON_NAMES = ['Winter', 'Spring', 'Summer', 'Fall']

class ThermalZoneSimulator:

    def __init__(self, weather_df: pd.DataFrame, max_hvac_power: float=3000.0, control_timestep: int=300, episode_hours: int=24, setpoint_temp: float=22.0, C_min: float=200000.0, C_max: float=500000.0, U_min: float=30.0, U_max: float=80.0, solar_gain_coeff: float=0.5, n_wind: float=0.65, k_wind: float=4.45, k_stack: float=2.6, eta_cop: float=0.4, cop_min: float=0.8, cop_max: float=10.0):
        self._weather = weather_df.reset_index(drop=True)
        self._n_hours = len(self._weather)
        self.max_hvac_power = max_hvac_power
        self.control_timestep = control_timestep
        self.episode_hours = episode_hours
        self.max_steps = int(episode_hours * 3600 / control_timestep)
        self.setpoint_temp = setpoint_temp
        self.C_min = C_min
        self.C_max = C_max
        self.U_min = U_min
        self.U_max = U_max
        self.C_nominal = (C_min + C_max) / 2.0
        self.U_nominal = (U_min + U_max) / 2.0
        self.solar_gain_coeff = solar_gain_coeff
        self.n_wind = n_wind
        self.k_wind = k_wind
        self.k_stack = k_stack
        self.eta_cop = eta_cop
        self.cop_min = cop_min
        self.cop_max = cop_max
        self.thermal_capacitance: float = self.C_nominal
        self.heat_transfer_coeff: float = self.U_nominal
        self.C_ratio: float = 1.0
        self.U_ratio: float = 1.0
        self.indoor_temp: float = setpoint_temp
        self.hvac_power: float = 0.0
        self.current_step: int = 0
        self._ep_weather: pd.DataFrame = self._weather.iloc[:episode_hours + 1].copy()
        self._last_weather: dict = self._row_to_dict(self._weather.iloc[0])
        self._rng: Generator = np.random.default_rng()
        self.current_U_eff: float = float(self.U_nominal)
        self.current_cop: float = 5.0
        self._season_starts = self._build_season_indices()
        self._season_cycle: int = 0

    def reset(self, seed: Optional[int]=None) -> float:
        self._rng = np.random.default_rng(seed)
        self.thermal_capacitance = float(self._rng.uniform(self.C_min, self.C_max))
        self.heat_transfer_coeff = float(self._rng.uniform(self.U_min, self.U_max))
        self.C_ratio = self.thermal_capacitance / self.C_nominal
        self.U_ratio = self.heat_transfer_coeff / self.U_nominal
        season_id = self._season_cycle % 4
        self._season_cycle += 1
        starts = self._season_starts[season_id]
        start_idx = int(self._rng.choice(starts))
        self._ep_weather = self._weather.iloc[start_idx:start_idx + self.episode_hours + 1].reset_index(drop=True)
        self.indoor_temp = float(self._rng.uniform(self.setpoint_temp - 4.0, self.setpoint_temp + 4.0))
        self.hvac_power = 0.0
        self.current_step = 0
        self._last_weather = self._row_to_dict(self._ep_weather.iloc[0])
        return self.indoor_temp

    def step(self, action: float) -> Tuple[float, dict, bool]:
        action = float(np.clip(action, -1.0, 1.0))
        self.hvac_power = action * self.max_hvac_power
        w = self._current_weather()
        self._last_weather = w
        v = max(w['wind_speed'], 0.0)
        delta_T = abs(self.indoor_temp - w['T_out'])
        U_eff = self.heat_transfer_coeff + self.k_wind * v ** self.n_wind + self.k_stack * delta_T ** 0.5
        self.current_U_eff = U_eff
        eps = 1.0
        T_in_K = self.indoor_temp + 273.15
        T_out_K = w['T_out'] + 273.15
        if action >= 0:
            cop = self.eta_cop * T_in_K / max(T_out_K - T_in_K, eps)
        else:
            cop = self.eta_cop * T_in_K / max(T_in_K - T_out_K, eps)
        self.current_cop = float(np.clip(cop, self.cop_min, self.cop_max))
        Q_solar = self.solar_gain_coeff * w['GHI']
        dT = self.control_timestep / self.thermal_capacitance * (U_eff * (w['T_out'] - self.indoor_temp) + Q_solar - self.hvac_power)
        self.indoor_temp += dT
        self.current_step += 1
        done = self.current_step >= self.max_steps
        return (self.indoor_temp, w, done)

    def get_state(self) -> dict:
        w = self._last_weather
        return {'indoor_temp': self.indoor_temp, 'outdoor_temp': w['T_out'], 'GHI': w['GHI'], 'wind_speed': w['wind_speed'], 'setpoint_temp': self.setpoint_temp, 'temp_error': self.indoor_temp - self.setpoint_temp, 'hvac_power': self.hvac_power, 'current_step': self.current_step, 'elapsed_hours': self.elapsed_hours, 'month': int(w['month']), 'doy': float(w['doy']), 'C_ratio': self.C_ratio, 'U_ratio': self.U_ratio, 'current_cop': self.current_cop, 'current_U_eff': self.current_U_eff}

    @property
    def elapsed_hours(self) -> float:
        return self.current_step * self.control_timestep / 3600.0

    def _current_weather(self) -> dict:
        steps_per_hour = int(3600 / self.control_timestep)
        hour_idx = min(self.current_step // steps_per_hour, len(self._ep_weather) - 1)
        return self._row_to_dict(self._ep_weather.iloc[hour_idx])

    def _build_season_indices(self) -> dict[int, np.ndarray]:
        months = self._weather['month'].values
        max_start = max(0, self._n_hours - self.episode_hours - 1)
        buckets: dict[int, list[int]] = {0: [], 1: [], 2: [], 3: []}
        for i in range(max_start + 1):
            buckets[_MONTH_TO_SEASON[int(months[i])]].append(i)
        return {s: np.array(idxs, dtype=np.int64) for s, idxs in buckets.items()}

    @staticmethod
    def _row_to_dict(row: pd.Series) -> dict:
        return {'T_out': float(row['T_out']), 'GHI': float(row['GHI']), 'wind_speed': float(row['wind_speed']), 'month': int(row['month']), 'doy': float(row['doy']), 'hour_of_day': float(row['hour_of_day'])}
