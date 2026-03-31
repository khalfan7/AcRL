"""
Generalisation Test -- PPO Agent on Albany (unseen city)
========================================================
Runs 100 episodes (25 per season via stratified round-robin sampling) on
Albany NY test weather data.  The agent was trained only on Syracuse NY.

Outputs
-------
  Console   : overall + per-season statistics (MAE, RMSE, violations, energy, reward)
  CSV       : results/PPO/generalization_stats_ppo.csv    (per-episode metrics)
  Figure 1  : results/PPO/generalization_boxplots_ppo.png (metric distributions by season)
  Figure 2  : results/PPO/seasonal_profiles_ppo.png       (representative 24 h trace per season)
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from stable_baselines3 import PPO
from evaluation.generalization import run_generalization_test

run_generalization_test(PPO, "PPO")
