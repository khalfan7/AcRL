from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize

class SyncNormCallback(BaseCallback):

    def __init__(self, train_env: VecNormalize, eval_env: VecNormalize):
        super().__init__(verbose=0)
        self._train_env = train_env
        self._eval_env = eval_env

    def _on_step(self) -> bool:
        self._eval_env.obs_rms = self._train_env.obs_rms
        if self._train_env.norm_reward:
            self._eval_env.ret_rms = self._train_env.ret_rms
        return True
