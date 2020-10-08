"""
Encapsulation of policy and value function estimations with shared parameters
"""

import torch

from networks.networks_util import get_network_builder

class PolicyWithValue(object):
    """
    Common interface for policy and value networks
    """

    def __init__(self, env, observation, latent, estimate_q=False, vf_latent=None):
        """
        Constructor

        Args:
            env:            (gym.Env) environment
            observation:   (tensor) observations from environment
        """

        raise NotImplementedError

    def normalize_observations(self, observations):
        """
        Normalizes the observation space
        """
        raise NotImplementedError

    def step(self, observation, **extra_feed):
        """
        Compute the next action(s) given the observation(s)

        Args:
            observation: (tensor) observation(s) data

        Returns:
 
        """

        raise NotImplementedError

    def value(self, observations, **extra_feed):
        """
        Compute the value estimate given the observation(s)
        """
        raise NotImplementedError


def build_policy(env, policy_network, value_network=None, normalize_observations=False, **policy_kwargs):
    """
    Policy / Value network builder

    Args:
        env: (gym.Env class) environment
        policy_network: (str) name for the policy network type
        normalize_observations: (bool) if true normalizes and clips the observation space
        **policy_kwargs: pointer to arguments to instantiate the policy network

    Returns:
        (PolicyWithValue class) a class object that contains the instantiated policy network
                                along with methods to return actions / value estimates given
                                current observations
    """
    if isinstance(policy_network, str):
        # if 'policy_network' is a network type name
        # call the get_network_builder function
        # Q: is this redundant? because get_network_builder already has a check for
        # if 'policy_network' name is callable
        network_type = policy_network
        # policy_network is a customized (by network_type & **policy_kwargs) network object
        policy_network = get_network_builder(network_type)(**policy_kwargs)

        def policy_fn(nbatch=None, nsteps=None):
            """
            A function to return a policy network + methods
            """
            # get observation space
            ob_space = env.observation_space

