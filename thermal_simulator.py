import numpy as np
from typing import Optional, Tuple


class ThermalZoneSimulator:
    def __init__(
        self,
        thermal_capacitance: float = 300000,
        heat_transfer_coeff: float = 50,
        max_hvac_power: float = 3000,
        simulation_duration: int = 24 * 3600,
        control_timestep: int = 300,
        setpoint_temp: float = 22.0,
        initial_temp_range: Tuple[float, float] = (18.0, 28.0),
        outdoor_temp_profile: str = 'sinusoidal',
        outdoor_temp_base: float = 30.0,
        outdoor_temp_amplitude: float = 5.0,
    ):
        self.control_timestep = control_timestep
        self.simulation_duration = simulation_duration
        self.max_steps = int(simulation_duration / control_timestep)
        self.thermal_capacitance = thermal_capacitance
        self.heat_transfer_coeff = heat_transfer_coeff
        self.max_hvac_power = max_hvac_power
        self.setpoint_temp = setpoint_temp
        self.initial_temp_range = initial_temp_range
        self.outdoor_temp_profile = outdoor_temp_profile
        self.outdoor_temp_base = outdoor_temp_base
        self.outdoor_temp_amplitude = outdoor_temp_amplitude
        self.indoor_temp: float = 0.0
        self.outdoor_temp_array: np.ndarray = np.array([])
        self.current_step: int = 0
        self.hvac_power: float = 0.0

    def reset(self, initial_indoor_temp: Optional[float] = None, seed: Optional[int] = None) -> float:
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        self.hvac_power = 0.0
        self._generate_outdoor_temp_profile()
        if initial_indoor_temp is None:
            self.indoor_temp = np.random.uniform(self.initial_temp_range[0], self.initial_temp_range[1])
        else:
            self.indoor_temp = initial_indoor_temp
        return self.indoor_temp

    def _generate_outdoor_temp_profile(self) -> None:
        time_array = np.arange(self.max_steps) * self.control_timestep
        if self.outdoor_temp_profile == 'constant':
            self.outdoor_temp_array = np.full(self.max_steps, self.outdoor_temp_base)
        elif self.outdoor_temp_profile == 'step':
            self.outdoor_temp_array = np.empty(self.max_steps)
            midpoint = self.max_steps // 2
            self.outdoor_temp_array[:midpoint] = self.outdoor_temp_base
            self.outdoor_temp_array[midpoint:] = self.outdoor_temp_base + self.outdoor_temp_amplitude
        elif self.outdoor_temp_profile == 'sinusoidal':
            period = 24 * 3600

            # Peak at 15:00 (3 PM) — standard for many thermal models.
            # Math: we want (2π·t/24 + phase) = π/2 at t = peak_hour
            peak_hour = 15.0
            phase = (np.pi / 2) - (2 * np.pi * peak_hour / 24)

            self.outdoor_temp_array = (
                self.outdoor_temp_base
                + self.outdoor_temp_amplitude
                * np.sin(2 * np.pi * time_array / period + phase)
            )
        else:
            raise ValueError(f"Unknown outdoor_temp_profile: {self.outdoor_temp_profile}")

    def step(self, action: float) -> Tuple[float, float, bool]:
        action = np.clip(action, -1.0, 1.0)
        self.hvac_power = action * self.max_hvac_power
        outdoor_temp_current = self.outdoor_temp_array[self.current_step]
        delta_temp = (self.control_timestep / self.thermal_capacitance) * (
            self.heat_transfer_coeff * (outdoor_temp_current - self.indoor_temp) - self.hvac_power
        )
        self.indoor_temp += delta_temp
        self.current_step += 1
        done = self.current_step >= self.max_steps
        return self.indoor_temp, outdoor_temp_current, done

    def get_state(self) -> dict:
        return {
            'indoor_temp': self.indoor_temp,
            'outdoor_temp': self.outdoor_temp_array[min(self.current_step, self.max_steps - 1)],
            'setpoint_temp': self.setpoint_temp,
            'temp_error': self.indoor_temp - self.setpoint_temp,
            'hvac_power': self.hvac_power,
            'current_step': self.current_step,
            'elapsed_hours': self.current_step * self.control_timestep / 3600,
        }

    @property
    def elapsed_hours(self) -> float:
        return self.current_step * self.control_timestep / 3600
