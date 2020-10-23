"""
    Classes for episode runners
"""

from abc import ABC, abstractmethod

import torch
import numpy as np
from baselines.common.misc_util import dtype_to_torch

class AbstractRunner(ABC):
    """
    Abstract base class for runners
    """
    def __init__(self, *, env, model, nsteps):
        """
        Constructor

        Args:
            env: (gym.Env) environment
            model: (object) algorithm + network
            nsteps: (int) number of timesteps
        """
        super().__init__()
        self.env = env
        self.nenv = nenv = 1
        self.model = model
        self.nsteps = nsteps

        self.obs = torch.zeros(env.observation_space.shape, \
            dtype=dtype_to_torch(env.observation_space.dtype))
        init_obs = env.reset()
        if len(init_obs.shape) == 0:
            # some Discrete environments might reset() to 0D tensor
            # add one dimension
            init_obs = [init_obs]
        self.obs = torch.tensor(init_obs, dtype=dtype_to_torch( \
            env.observation_space.dtype))

        self.dones = torch.BoolTensor([False for _ in range(nenv)])

    @abstractmethod
    def run(self):
        """
        runner method
        """
        raise NotImplementedError