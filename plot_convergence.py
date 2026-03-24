"""
Training Convergence Plot — All Algorithms
===========================================
Loads evaluations.npz produced by EvalCallback (from each train_*.py run)
for PPO, A2C, SAC, TD3 and produces a single comparison figure:

    X-axis : Total environment timesteps
    Y-axis : Mean episode reward  ±1 std  (shaded band)

Each algorithm needs results/<ALGO>/evaluations.npz to be present;
missing algorithms are skipped with a warning.

Output: results/training_convergence.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE   = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(_HERE, "results")

ALGOS  = ["PPO", "A2C", "SAC", "TD3"]
COLORS = {"PPO": "#1f77b4", "A2C": "#ff7f0e", "SAC": "#2ca02c", "TD3": "#d62728"}
STYLES = {"PPO": "-",       "A2C": "--",       "SAC": "-.",       "TD3": ":"}

fig, ax = plt.subplots(figsize=(12, 6))
any_plotted = False

for algo in ALGOS:
    path = os.path.join(RES_DIR, algo, "evaluations.npz")
    if not os.path.exists(path):
        print(f"  [skip] {algo}: {path} not found — run train_{algo.lower()}.py first")
        continue

    data = np.load(path)
    timesteps   = data["timesteps"]   # shape (n_evals,)
    results     = data["results"]     # shape (n_evals, n_eval_episodes)

    mean_reward = results.mean(axis=1)
    std_reward  = results.std(axis=1)

    ax.plot(timesteps, mean_reward,
            color=COLORS[algo], linestyle=STYLES[algo],
            linewidth=2.2, label=algo, zorder=3)
    ax.fill_between(
        timesteps,
        mean_reward - std_reward,
        mean_reward + std_reward,
        color=COLORS[algo], alpha=0.15, zorder=2,
    )
    any_plotted = True

    print(
        f"{algo:4s}  final mean reward = {mean_reward[-1]:8.2f}  "
        f"(±{std_reward[-1]:.2f})  @ {timesteps[-1]:,} steps"
    )

if not any_plotted:
    print("No evaluations.npz files found. Train the agents first.")

ax.set_xlabel("Total Timesteps", fontsize=12)
ax.set_ylabel("Mean Episode Reward", fontsize=12)
ax.set_title(
    "Training Convergence — Mean Episode Reward vs Timesteps",
    fontsize=14, fontweight="bold",
)
ax.legend(fontsize=11, loc="lower right")
ax.grid(alpha=0.3)
ax.ticklabel_format(axis="x", style="scientific", scilimits=(0, 0))

out_path = os.path.join(RES_DIR, "training_convergence.png")
os.makedirs(RES_DIR, exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nFigure saved → {out_path}")
