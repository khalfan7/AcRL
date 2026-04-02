"""
Generalisation Test -- TD3 Agent
================================
Runs 100 episodes (25 per season) on both cities and compares performance.

Outputs (per city)
------------------
  Console   : overall + per-season statistics (MAE, RMSE, violations, energy, reward)
  CSV       : results/TD3/generalization_stats_td3_<city>.csv
  Figure 1  : results/TD3/generalization_boxplots_td3_<city>.png
  Figure 2  : results/TD3/seasonal_profiles_td3_<city>.png

Comparison output
-----------------
  Figure 3  : results/TD3/train_vs_test_comparison_td3.png
  Console   : generalisation gap (%) per metric
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from stable_baselines3 import TD3
from evaluation.generalization import run_train_vs_test

run_train_vs_test(TD3, "TD3_nl")
