"""
Train SAC agent for HVAC control using StableBaselines3
"""

import os
from stable_baselines3 import SAC
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
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'SAC')
    os.makedirs(log_dir, exist_ok=True)

    # Vectorized envs
    env = make_vec_env(make_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv, seed=42)
    env = VecNormalize(env, norm_obs=True, norm_reward=False)

    # Eval env
    eval_env = make_vec_env(make_env, n_envs=1, seed=123)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=10000,
        deterministic=True,
        render=False,
    )

    # SAC
    model = SAC(
        policy='MlpPolicy',
        env=env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        tau=0.02,
        gamma=0.98,
        learning_starts=1000,
        device='auto',
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
