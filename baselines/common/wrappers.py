"""
    Common wrappers
"""

import numpy as np
import gym


class TimeLimit(gym.Wrapper):
    """
    
    """
    def __init__(self, env, max_episode_steps=None):
        """
        Constructor

        Args:
            env: (Env) environment
            max_episode_steps: (int) maximum number of steps in one episode
        """
        super(TimeLimit, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapse_steps = 0

    def step(self, action):
        """
        step() function

        Args:
            action: (gym.action_space.dtype) action
        """
        # make one step
        observation, reward, done, info = self.env.step(action)
        # increase elapsed steps
        self._elapse_steps += 1
        # if espisode terminates, add flags
        if self._elapse_steps >= self._max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True

        return observation, reward, done, info


class ClipActionsWrapper(gym.ActionWrapper):
    """
    Clips the values of given actions into bounded by [action_space.low, action_space_high]

    note:
    - need not overload __init__() as there is no additional initialization required
    - need not overload step() for it is an action wrapper, better to overload action()
    """

    def action(self, action):
        """
        overload action() function

        Args:
            action: (gym.action_space.dtype) action for the step
        """
        action = np.nan_to_num(action)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return action

    def reset(self, **kwargs):
        """
        reset function
        """
        return self.env.reset(**kwargs)

class RewardScalerWrapper(gym.RewardWrapper):
    """
    Scales the reward properly

    Note:
    - a very important feature for PPO
    - affects performance drastically
    """
    def __init__(self, env, scale=1.0):
        """
        Constructor

        Args:
            env: (Env) environment
            scale: (float) reward scaling factor
        """
        super(RewardScalerWrapper, self).__init__(env)
        self.scale = scale

    def reward(self, reward):
        """
        overload reward() function by multiplying reward with the scale factor
        """
        return reward * self.scale
