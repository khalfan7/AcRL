import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from stable_baselines3 import PPO
from evaluation.generalization import run_train_vs_test
run_train_vs_test(PPO, 'PPO_nl')
