"""
Generalisation Test -- A2C Agent
================================
Runs 100 episodes (25 per season) on both cities and compares performance.

Outputs (per city)
------------------
  Console   : overall + per-season statistics (MAE, RMSE, violations, energy, reward)
  CSV       : results/A2C/generalization_stats_a2c_<city>.csv
  Figure 1  : results/A2C/generalization_boxplots_a2c_<city>.png
  Figure 2  : results/A2C/seasonal_profiles_a2c_<city>.png

Comparison output
-----------------
  Figure 3  : results/A2C/train_vs_test_comparison_a2c.png
  Console   : generalisation gap (%) per metric
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from stable_baselines3 import A2C
from evaluation.generalization import run_train_vs_test

run_train_vs_test(A2C, "A2C_nl")
