"""
Train A2C agent for HVAC control using StableBaselines3

A2C (Advantage Actor-Critic) is an on-policy algorithm that:
- Uses multiple parallel environments for stable training
- Is synchronous (deterministic) unlike A3C
- Best run on CPU with SubprocVecEnv
"""

import os
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from hvac_environment import HVACControlEnv


def make_env():
    return HVACControlEnv()


if __name__ == '__main__':
    # Config
    n_envs = 8  # A2C benefits from multiple parallel envs
    total_timesteps = 500_000

    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'A2C')
    os.makedirs(log_dir, exist_ok=True)

    # Vectorized envs (A2C runs better on CPU with SubprocVecEnv)
    env = make_vec_env(make_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv, seed=42)
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # Eval env
    eval_env = make_vec_env(make_env, n_envs=1, seed=123)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=10000,
        deterministic=True,
        render=False,
    )

    # A2C
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

    model.learn(total_timesteps=total_timesteps, callback=eval_callback, log_interval=1, progress_bar=True)
    model.save(f'{log_dir}/final_model')
    env.save(f'{log_dir}/vecnormalize.pkl')
    env.close()
    eval_env.close()
    
    print(f"Training complete. Model saved to {log_dir}/final_model")
