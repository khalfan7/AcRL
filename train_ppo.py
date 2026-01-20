"""
Train PPO agent for HVAC control using StableBaselines3
"""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from hvac_environment import HVACControlEnv


def make_env():
    return HVACControlEnv()


if __name__ == '__main__':
    # Config
    n_envs = 8
    total_timesteps = 500_000
    log_dir = 'results/PPO'
    os.makedirs(log_dir, exist_ok=True)

    # Vectorized envs
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

    # PPO
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

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save(f'{log_dir}/final_model')
    env.save(f'{log_dir}/vecnormalize.pkl')
    env.close()
    eval_env.close()
    
    print(f"Training complete. Model saved to {log_dir}/final_model")
