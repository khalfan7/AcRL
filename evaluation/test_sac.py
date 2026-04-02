"""
Generalisation Test -- SAC Agent
================================
Runs 100 episodes (25 per season) on both cities and compares performance.

Outputs (per city)
------------------
  Console   : overall + per-season statistics (MAE, RMSE, violations, energy, reward)
  CSV       : results/SAC/generalization_stats_sac_<city>.csv
  Figure 1  : results/SAC/generalization_boxplots_sac_<city>.png
  Figure 2  : results/SAC/seasonal_profiles_sac_<city>.png

Comparison output
-----------------
  Figure 3  : results/SAC/train_vs_test_comparison_sac.png
  Console   : generalisation gap (%) per metric
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from stable_baselines3 import SAC
from evaluation.generalization import run_train_vs_test

run_train_vs_test(SAC, "SAC_nl")
