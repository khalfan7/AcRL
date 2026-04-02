"""
Generalisation Test -- PPO Agent
================================
Runs 100 episodes (25 per season) on both cities and compares performance.

Outputs (per city)
------------------
  Console   : overall + per-season statistics (MAE, RMSE, violations, energy, reward)
  CSV       : results/PPO/generalization_stats_ppo_<city>.csv
  Figure 1  : results/PPO/generalization_boxplots_ppo_<city>.png
  Figure 2  : results/PPO/seasonal_profiles_ppo_<city>.png

Comparison output
-----------------
  Figure 3  : results/PPO/train_vs_test_comparison_ppo.png
  Console   : generalisation gap (%) per metric
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from stable_baselines3 import PPO
from evaluation.generalization import run_train_vs_test

run_train_vs_test(PPO, "PPO_nl")
