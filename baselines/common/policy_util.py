"""
Encapsulation of policy and value function estimations with shared parameters
"""

import torch

class PolicyWithValue(object):
    """
    Common interface for policy and value networks
    """

    def __init__(self, env, observations, latent, estimate_q=False, vf_latent=None):
        """
        Constructor

        Args:
            env:            (gym.Env) environment
            observation:   (tensor) observations from environment
        """

        self.x = observations

    def step(self, observation, **extra_feed):
        """
        Compute the next action(s) given the observation(s)

        Args:
            observation: (tensor) observation(s) data

        Returns:
            
        """


def build_policy(env, policy_network, value_network=None, **policy_kwargs):
    """
    Policy / Value network builder
    Returns a PolicyWithValue class object
    """