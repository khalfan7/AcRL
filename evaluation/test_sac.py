"""
Generalisation Test -- SAC Agent on Albany (unseen city)
========================================================
Runs 100 episodes (25 per season via stratified round-robin sampling) on
Albany NY test weather data.  The agent was trained only on Syracuse NY.

Outputs
-------
  Console   : overall + per-season statistics (MAE, RMSE, violations, energy, reward)
  CSV       : results/SAC/generalization_stats_sac.csv    (per-episode metrics)
  Figure 1  : results/SAC/generalization_boxplots_sac.png (metric distributions by season)
  Figure 2  : results/SAC/seasonal_profiles_sac.png       (representative 24 h trace per season)
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from stable_baselines3 import SAC
from evaluation.generalization import run_generalization_test

run_generalization_test(SAC, "SAC")
