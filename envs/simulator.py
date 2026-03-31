"""
Lumped-capacitance (RC) thermal simulator
==========================================
Extended physics with real weather data, solar gain, wind-modified U,
and per-episode domain randomisation of C and U.

Governing ODE (Euler forward, step Δt = control_timestep seconds):

    dT_in/dt = (1/C) · [ U_eff·(T_out − T_in) + α·GHI − Q_hvac ]

where:
    C       = thermal_capacitance  [J/°C]   — randomised ∈ [C_min, C_max]
    U_eff   = U · (1 + k_w · wind_speed)   — wind-modified envelope conductance
    U       = heat_transfer_coeff  [W/°C]   — randomised ∈ [U_min, U_max]
    k_w     = wind_coeff           [s/m]
    α       = solar_gain_coeff     [m²]     — effective aperture × absorptance
    GHI     = Global Horizontal Irradiance  [Wh/m²] (from weather data)
    Q_hvac  = action × max_hvac_power [W]  — +ve = cooling, -ve = heating

Domain randomisation
--------------------
    C ~ Uniform(C_min, C_max)
    U ~ Uniform(U_min, U_max)

This creates model variation across episodes so the agent learns a policy
that is robust to building-parameter uncertainty.

Weather data
------------
Accepts a pandas DataFrame produced by envs.weather.load_hnn_multi().
Each episode draws a random 24-hour window from the full archive.
Sub-hourly steps within the same hour share the same hourly weather value
(zero-order hold).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from numpy.random import Generator

# Month → season quartile (0=Winter, 1=Spring, 2=Summer, 3=Fall)
_MONTH_TO_SEASON = {
    12: 0,  1: 0,  2: 0,   # DJF — Winter
     3: 1,  4: 1,  5: 1,   # MAM — Spring
     6: 2,  7: 2,  8: 2,   # JJA — Summer
     9: 3, 10: 3, 11: 3,   # SON — Fall
}
_SEASON_NAMES = ["Winter", "Spring", "Summer", "Fall"]


class ThermalZoneSimulator:

    def __init__(
        self,
        weather_df: pd.DataFrame,
        max_hvac_power:   float = 3_000.0,
        control_timestep: int   = 300,          # seconds (5 min)
        episode_hours:    int   = 24,
        setpoint_temp:    float = 22.0,
        # Domain-randomisation bounds
        C_min: float = 200_000.0,               # J/°C
        C_max: float = 500_000.0,
        U_min: float =     30.0,                # W/°C
        U_max: float =     80.0,
        # Extended physics
        solar_gain_coeff: float = 0.5,          # m² (aperture × absorptance)
        wind_coeff:       float = 0.05,          # s/m
    ):
        self._weather          = weather_df.reset_index(drop=True)
        self._n_hours          = len(self._weather)
        self.max_hvac_power    = max_hvac_power
        self.control_timestep  = control_timestep
        self.episode_hours     = episode_hours
        self.max_steps         = int(episode_hours * 3600 / control_timestep)
        self.setpoint_temp     = setpoint_temp

        # DR bounds and midpoint (nominal) values
        self.C_min     = C_min
        self.C_max     = C_max
        self.U_min     = U_min
        self.U_max     = U_max
        self.C_nominal = (C_min + C_max) / 2.0   # 350 000 J/°C
        self.U_nominal = (U_min + U_max) / 2.0   #      55 W/°C

        self.solar_gain_coeff = solar_gain_coeff
        self.wind_coeff       = wind_coeff

        # Episode state — properly initialised by reset()
        self.thermal_capacitance: float = self.C_nominal
        self.heat_transfer_coeff: float  = self.U_nominal
        self.C_ratio: float = 1.0
        self.U_ratio: float = 1.0
        self.indoor_temp:  float = setpoint_temp
        self.hvac_power:   float = 0.0
        self.current_step: int   = 0

        self._ep_weather: pd.DataFrame = self._weather.iloc[:episode_hours + 1].copy()
        self._last_weather: dict = self._row_to_dict(self._weather.iloc[0])
        self._rng: Generator = np.random.default_rng()

        # Stratified seasonal sampling — pre-compute valid start indices per quartile
        self._season_starts = self._build_season_indices()
        self._season_cycle: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> float:
        """Start a new episode. Returns the initial indoor temperature."""
        self._rng = np.random.default_rng(seed)

        # Domain randomisation — new C and U per episode
        self.thermal_capacitance = float(self._rng.uniform(self.C_min, self.C_max))
        self.heat_transfer_coeff  = float(self._rng.uniform(self.U_min, self.U_max))
        self.C_ratio = self.thermal_capacitance / self.C_nominal
        self.U_ratio = self.heat_transfer_coeff  / self.U_nominal

        # Stratified seasonal sampling — cycle through quartiles,
        # randomise the starting hour within the current season.
        season_id = self._season_cycle % 4
        self._season_cycle += 1
        starts = self._season_starts[season_id]
        start_idx = int(self._rng.choice(starts))
        self._ep_weather = (
            self._weather
            .iloc[start_idx : start_idx + self.episode_hours + 1]
            .reset_index(drop=True)
        )

        # Initial indoor temperature: ±4 °C around setpoint
        self.indoor_temp  = float(
            self._rng.uniform(self.setpoint_temp - 4.0, self.setpoint_temp + 4.0)
        )
        self.hvac_power   = 0.0
        self.current_step = 0

        self._last_weather = self._row_to_dict(self._ep_weather.iloc[0])
        return self.indoor_temp

    def step(self, action: float) -> Tuple[float, dict, bool]:
        """
        Advance the simulation by one control step.

        Parameters
        ----------
        action : float ∈ [-1, 1]

        Returns
        -------
        indoor_temp : float
        weather     : dict  — weather variables used for this step
        done        : bool
        """
        action = float(np.clip(action, -1.0, 1.0))
        self.hvac_power = action * self.max_hvac_power

        w = self._current_weather()
        self._last_weather = w

        U_eff   = self.heat_transfer_coeff * (1.0 + self.wind_coeff * w["wind_speed"])
        Q_solar = self.solar_gain_coeff * w["GHI"]

        dT = (self.control_timestep / self.thermal_capacitance) * (
            U_eff * (w["T_out"] - self.indoor_temp)
            + Q_solar
            - self.hvac_power
        )
        self.indoor_temp  += dT
        self.current_step += 1

        done = self.current_step >= self.max_steps
        return self.indoor_temp, w, done

    def get_state(self) -> dict:
        w = self._last_weather
        return {
            "indoor_temp":   self.indoor_temp,
            "outdoor_temp":  w["T_out"],
            "GHI":           w["GHI"],
            "wind_speed":    w["wind_speed"],
            "setpoint_temp": self.setpoint_temp,
            "temp_error":    self.indoor_temp - self.setpoint_temp,
            "hvac_power":    self.hvac_power,
            "current_step":  self.current_step,
            "elapsed_hours": self.elapsed_hours,
            "month":         int(w["month"]),
            "doy":           float(w["doy"]),
            "C_ratio":       self.C_ratio,
            "U_ratio":       self.U_ratio,
        }

    @property
    def elapsed_hours(self) -> float:
        return self.current_step * self.control_timestep / 3600.0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _current_weather(self) -> dict:
        """Return the hourly weather row for the current step (zero-order hold)."""
        steps_per_hour = int(3600 / self.control_timestep)
        hour_idx = min(
            self.current_step // steps_per_hour,
            len(self._ep_weather) - 1,
        )
        return self._row_to_dict(self._ep_weather.iloc[hour_idx])

    def _build_season_indices(self) -> dict[int, np.ndarray]:
        """Pre-compute valid episode-start indices for each season quartile."""
        months = self._weather["month"].values
        max_start = max(0, self._n_hours - self.episode_hours - 1)
        buckets: dict[int, list[int]] = {0: [], 1: [], 2: [], 3: []}
        for i in range(max_start + 1):
            buckets[_MONTH_TO_SEASON[int(months[i])]].append(i)
        return {s: np.array(idxs, dtype=np.int64) for s, idxs in buckets.items()}

    @staticmethod
    def _row_to_dict(row: pd.Series) -> dict:
        return {
            "T_out":       float(row["T_out"]),
            "GHI":         float(row["GHI"]),
            "wind_speed":  float(row["wind_speed"]),
            "month":       int(row["month"]),
            "doy":         float(row["doy"]),
            "hour_of_day": float(row["hour_of_day"]),
        }
