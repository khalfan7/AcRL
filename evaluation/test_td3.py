"""
Generalisation Test -- TD3 Agent on Albany (unseen city)
========================================================
Runs 100 episodes (25 per season via stratified round-robin sampling) on
Albany NY test weather data.  The agent was trained only on Syracuse NY.

Outputs
-------
  Console   : overall + per-season statistics (MAE, RMSE, violations, energy, reward)
  CSV       : results/TD3/generalization_stats_td3.csv    (per-episode metrics)
  Figure 1  : results/TD3/generalization_boxplots_td3.png (metric distributions by season)
  Figure 2  : results/TD3/seasonal_profiles_td3.png       (representative 24 h trace per season)
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from stable_baselines3 import TD3
from evaluation.generalization import run_generalization_test

run_generalization_test(TD3, "TD3")
