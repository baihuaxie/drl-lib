"""
    Abstract base class for vectorized environment
"""

from abc import ABC, abstractmethod


class VecEnv(ABC):
    """
    Abstract asynchronous, vectorized environment
    - batch observation data from multiple copies of an environment running in parallel
    - an action from an agent becomes a batch of actions to be applied per-environment
    """
    def __init__(self, num_envs, observation_space, action_space):
        """
        Constructor

        Args:
            num_envs: (int) number of environment copies
            observation_space: (gym.space) Discrete, Box or MultiDiscrete
            actioN_space: (gym.space) Discrete, Box or MultiDiscrete
        """
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        """
        Reset all environments and return an iterable of initial observations
        """
        raise NotImplementedError

    @abstractmethod
    def step_async(self, actions):
        """
        Step all environment copies asynchronously

        Args:
            actions: (tensor) each environment copy receives its own action
        """
        raise NotImplementedError

    @abstractmethod
    def step_wait(self):
        """
        Returns stepped results from step() or step_async() calls

        Returns:
            obs: (tensor or dict of tensors)
            rewards: (tensor)
            dones: (BoolTensor)
            infos: (a sequence of info objects)
        """
        raise NotImplementedError

    @abstractmethod
    def get_images(self):
        """
        Returns an RGB array image for each environment copy
        """
        pass

    def step_sync(self, action):
        """
        Step all environment copies synchronously

        Args:
            action: (tensor) all environment copies receive the same action
        """
        pass