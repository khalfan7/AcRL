"""
Train A2C agent for HVAC control using StableBaselines3.

A2C (Advantage Actor-Critic) is a synchronous on-policy algorithm:
  - Multiple parallel envs produce diverse rollouts per update step.
  - Synchronous updates (unlike async A3C) ensure deterministic training.
  - SubprocVecEnv + CPU is the typical optimal setup for A2C.
  - Rewards are normalised to stabilise the critic's value target.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from envs import make_train_env
from training.callbacks import SyncNormCallback


if __name__ == '__main__':
    # ── Config ────────────────────────────────────────────────────────────
    n_envs          = 8  # A2C benefits from many parallel workers
    total_timesteps = 500_000
    log_dir = str(_ROOT / "results" / "A2C_nl")
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # ── Training envs (SubprocVecEnv → true parallelism on CPU) ──────────
    env = make_vec_env(make_train_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv, seed=42)
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # ── Eval env — must NOT update normalisation stats during evaluation ──
    eval_env = make_vec_env(make_train_env, n_envs=1, seed=123)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=10_000 // n_envs,   # per-env steps → global steps
        deterministic=True,
        render=False,
    )
    sync_callback = SyncNormCallback(env, eval_env)

    # ── Model ─────────────────────────────────────────────────────────────
    model = A2C(
        policy='MlpPolicy',
        env=env,
        learning_rate=7e-4,
        n_steps=5,
        gamma=0.99,
        gae_lambda=1.0,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_rms_prop=True,
        normalize_advantage=True,
        device='cpu',  # A2C typically runs better on CPU
        seed=42,
        verbose=1,
        tensorboard_log=log_dir,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[sync_callback, eval_callback],
        log_interval=1,
        progress_bar=True,
    )
    model.save(f'{log_dir}/final_model')
    env.save(f'{log_dir}/vecnormalize.pkl')
    env.close()
    eval_env.close()

    print(f"Training complete. Model saved to {log_dir}/final_model")
