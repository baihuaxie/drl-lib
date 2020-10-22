"""
    subprocess vectorized environment class
"""

import multiprocessing
from .vec_env import VecEnv


class SubprocVecEnv(VecEnv):
    """
    A vectorized environment that runs in parallel in subprocesses
    """
    def __init__(self, env_fns, context='spawn'):
        """
        Constructor

        Args:
            env_fns: (iterable of callable functions) each fn() in env_fns return a copy of
                     environment object
            context: (str) supported by multiprocessing; 'spawn', 'fork' or 'forkserver'
                     windows only supports 'spawn'
        """
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(env_fns)
        env = self.envs[0]
        obs_space = env.observation_space

        super().__init__(self.num_envs, obs_space, env.action_space)


    def reset(self):
        """
        """
        
    
    def step_async(self, actions):
        """
        """

    def step_wait(self):
        """
        """

    def get_images(self):
        """
        """