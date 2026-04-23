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
    n_envs = 8
    total_timesteps = 500000
    log_dir = str(_ROOT / 'results' / 'A2C_nl')
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    env = make_vec_env(make_train_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv, seed=42)
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    eval_env = make_vec_env(make_train_env, n_envs=1, seed=123)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir, log_path=log_dir, eval_freq=10000 // n_envs, deterministic=True, render=False)
    sync_callback = SyncNormCallback(env, eval_env)
    model = A2C(policy='MlpPolicy', env=env, learning_rate=0.0007, n_steps=5, gamma=0.99, gae_lambda=1.0, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, use_rms_prop=True, normalize_advantage=True, device='cpu', seed=42, verbose=1, tensorboard_log=log_dir)
    model.learn(total_timesteps=total_timesteps, callback=[sync_callback, eval_callback], log_interval=1, progress_bar=True)
    model.save(f'{log_dir}/final_model')
    env.save(f'{log_dir}/vecnormalize.pkl')
    env.close()
    eval_env.close()
    print(f'Training complete. Model saved to {log_dir}/final_model')
