import gymnasium as gym
from gymnasium import spaces
import numpy as np
from thermal_simulator import ThermalZoneSimulator


class HVACControlEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        thermal_capacitance: float = 300000,
        heat_transfer_coeff: float = 50,
        max_hvac_power: float = 3000,
        simulation_duration: int = 24 * 3600,
        control_timestep: int = 300,
        setpoint_temp: float = 22.0,
        outdoor_temp_base: float = 30.0,
        outdoor_temp_amplitude: float = 5.0,
        lambda_weight: float = 0.1,
        render_mode=None,
    ):
        super().__init__()
        
        self.render_mode = render_mode
        # Energy-penalty weight: trades off comfort vs. power usage
        # R = -|temp_error| - lambda_weight * |action|
        self.lambda_weight = lambda_weight

        # Create thermal zone simulator
        self.simulator = ThermalZoneSimulator(
            thermal_capacitance=thermal_capacitance,
            heat_transfer_coeff=heat_transfer_coeff,
            max_hvac_power=max_hvac_power,
            simulation_duration=simulation_duration,
            control_timestep=control_timestep,
            setpoint_temp=setpoint_temp,
            outdoor_temp_base=outdoor_temp_base,
            outdoor_temp_amplitude=outdoor_temp_amplitude,
        )
        
        # Define action space: continuous control signal [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(1,), 
            dtype=np.float32
        )
        
        # Define observation space:
        #   [temp_error, outdoor_temp_normalized, sin_time, cos_time]
        #
        # sin/cos encode the hour-of-day cyclically:
        #   midnight  -> sin=0,  cos=+1
        #   06:00     -> sin=+1, cos=0   
        #   noon      -> sin=0,  cos=-1  
        #   18:00     -> sin=-1, cos=0   
        #
        # Night windows assumed: 00-06 and 18-24
        self.observation_space = spaces.Box(
            low=np.array([-50.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([ 50.0,  1.0,  1.0,  1.0], dtype=np.float32),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if seed is not None:
            self.simulator.reset(seed=seed)
        else:
            self.simulator.reset()
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info

    def step(self, action):
        # Extract scalar action from array
        action_value = float(action[0])
        
        # Step the thermal simulator
        indoor_temp, outdoor_temp, done = self.simulator.step(action_value)
        
        # Get observation
        observation = self._get_observation()
        
        # Reward: penalise both temperature deviation and energy use
        #   R = -|temp_error| - λ·|action|
        # lambda_weight << 1 keeps comfort as the primary objective while
        # discouraging unnecessary full-power operation.
        temperature_error = abs(indoor_temp - self.simulator.setpoint_temp)
        energy_penalty    = self.lambda_weight * abs(action_value)
        reward = -temperature_error - energy_penalty
        
        # Episode termination
        terminated = done
        truncated = False
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        # Temperature error (indoor - setpoint)
        temperature_error = self.simulator.indoor_temp - self.simulator.setpoint_temp
        
        # Normalize outdoor temp to [-1, 1] range
        current_step = min(self.simulator.current_step, self.simulator.max_steps - 1)
        outdoor_temp_current = self.simulator.outdoor_temp_array[current_step]
        outdoor_temp_normalized = (
            (outdoor_temp_current - self.simulator.outdoor_temp_base) / 
            max(self.simulator.outdoor_temp_amplitude, 1.0)
        )
        outdoor_temp_normalized = np.clip(outdoor_temp_normalized, -1.0, 1.0)

   
        hour = self.simulator.elapsed_hours % 24.0
        angle = 2.0 * np.pi * hour / 24.0
        sin_time = float(np.sin(angle))   # +1 at 06:00, -1 at 18:00
        cos_time = float(np.cos(angle))   # +1 at 00:00, -1 at 12:00

        return np.array(
            [temperature_error, outdoor_temp_normalized, sin_time, cos_time],
            dtype=np.float32,
        )

    def _get_info(self) -> dict:
        return self.simulator.get_state()

    def render(self):
        if self.render_mode == 'human':
            state = self.simulator.get_state()
            print(
                f"Time: {state['elapsed_hours']:.2f}h | "
                f"Indoor: {state['indoor_temp']:.2f}C | "
                f"Setpoint: {state['setpoint_temp']:.2f}C | "
                f"Error: {state['temp_error']:.2f}C | "
                f"HVAC Power: {state['hvac_power']:.0f}W"
            )

    def close(self):
        pass


# Register the environment with Gymnasium — guarded so repeated imports
# (or imports from different scripts) don't raise a duplicate-id error.
# Using try/except is more robust across Gymnasium versions than checking
# the registry dict directly.
try:
    gym.register(
        id="HVACControl-v0",
        entry_point="hvac_environment:HVACControlEnv",
        max_episode_steps=288,  # 24 hours at 5-min control intervals
    )
except Exception:
    pass  # already registered by a previous import
