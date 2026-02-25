"""
Train TD3 agent for HVAC control using StableBaselines3

TD3 (Twin Delayed DDPG) is an off-policy algorithm that improves DDPG with:
- Clipped double Q-learning
- Delayed policy updates  
- Target policy smoothing
"""

import os
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.noise import NormalActionNoise
from hvac_environment import HVACControlEnv


def make_env():
    return HVACControlEnv()


if __name__ == '__main__':
    # Config

    n_envs = 1  # TD3 is off-policy, typically uses single env
    total_timesteps = 500_000
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'TD3')
    os.makedirs(log_dir, exist_ok=True)

    # Environment (TD3 works best with single env)
    env = make_vec_env(make_env, n_envs=n_envs, seed=42)
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

    # Action noise for exploration
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions)
    )

    # TD3
    model = TD3(
        policy='MlpPolicy',
        env=env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        tau=0.005,
        gamma=0.98,
        learning_starts=1000,
        action_noise=action_noise,
        policy_delay=2,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
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
