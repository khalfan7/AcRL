"""
Train PPO agent for HVAC control using StableBaselines3.

PPO (Proximal Policy Optimisation) is an on-policy algorithm suited to
continuous control.  It is run with 8 parallel sub-process environments
to collect diverse rollouts and VecNormalize for observation/reward scaling.
"""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from hvac_environment import HVACControlEnv


def make_env():
    return HVACControlEnv()


class SyncNormCallback(BaseCallback):
    """Keep eval-env normalization stats in sync with training env."""

    def __init__(self, train_env: VecNormalize, eval_env: VecNormalize):
        super().__init__(verbose=0)
        self._train_env = train_env
        self._eval_env  = eval_env

    def _on_step(self) -> bool:
        self._eval_env.obs_rms    = self._train_env.obs_rms
        self._eval_env.ret_rms    = self._train_env.ret_rms
        return True


if __name__ == '__main__':
    # ── Config ────────────────────────────────────────────────────────────
    n_envs          = 8
    total_timesteps = 500_000
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'PPO')
    os.makedirs(log_dir, exist_ok=True)

    # ── Training envs (on-policy: multiple parallel envs help) ────────────
    env = make_vec_env(make_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv, seed=42)
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # ── Eval env — must NOT update normalisation stats during evaluation ──
    eval_env = make_vec_env(make_env, n_envs=1, seed=123)
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
    model = PPO(
        policy='MlpPolicy',
        env=env,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.98,
        gae_lambda=0.95,
        ent_coef=0.01,
        device='auto',
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
