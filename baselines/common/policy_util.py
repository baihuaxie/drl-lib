"""
Encapsulation of policy and value function estimations with shared parameters
"""

import torch

from common.networks.get_networks import get_network_builder

class PolicyWithValue(object):
    """
    Common interface for policy and value networks
    """

    def __init__(self, policy_net, env=None, observations=None, value_net=None,
                 estimate_q=False, normalize_observations=False):
        """
        Constructor

        Args:
            env:            (gym.Env) environment
            observations:   (torch.Tensor) observations from environment
            policy_net:     (str) type of policy network
            value_net:      (str) type of value network; if None or 'shared', default
                            value_net = policy_net
            estimate_q:     (bool) if True also returns an estimate for q-value 
        """
        super().__init__()

        self._policy_net = get_network_builder(policy_net)()

        # consider cases where policy net and value net share all but last few output layers?
        if value_net is None or value_net == 'shared':
            self._value_net = self._policy_net
        else:
            self._value_net = get_network_builder(value_net)()

        self._X = observations
        self._env = env
        self._estimate_q = estimate_q

        if normalize_observations:
            self._normalize_observations()

        self._encode_observations()

    def _normalize_observations(self):
        """
        Normalizes the observations
        """
        pass

    def _encode_observations(self):
        """
        Encode the observations to be suitable for env.observation_space
        """
        pass
        
    def step(self, observation, **extra_feed):
        """
        Compute the next action(s) given the observation(s)

        Args:
            observation: (tensor) observation(s) data

        Returns:
 
        """
        pass

    def value(self, observations, **extra_feed):
        """
        Compute the value estimate given the observation(s)
        """
        pass


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

