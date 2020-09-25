"""
    Common wrappers
"""

import numpy as np
import gym



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

