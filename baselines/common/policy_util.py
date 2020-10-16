"""
Encapsulation of policy and value function estimations with shared parameters
"""

import sys
sys.path.append('../')

import torch
from gym.spaces import Discrete, Box, MultiDiscrete

from common.networks.get_networks import get_network_builder
from common.preprocessor import RunningMeanStd, OneHotPreprocessor

class PolicyWithValue(object):
    """
    Common interface for policy and value networks
    """

    def __init__(self, policy_net, env=None, value_net=None,
                 estimate_q=False, normalize_observations=False):
        """
        Constructor

        Args:
            env:            (gym.Env) environment
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

        self._env = env
        self._estimate_q_flag = estimate_q
        self._normalize_obs_flag = normalize_observations

        if isinstance(env.observation_space, Box):
            self._rms = RunningMeanStd(shape=torch.Size((1,) + env.observation_space.shape))
            # returns obs directly without encoding for Box environments
            self._encoder = lambda *args: args
        elif isinstance(env.observation_space, Discrete):
            self._rms = RunningMeanStd(shape=torch.Size((1,) + (env.observation_space.n,)))
            self._encoder = OneHotPreprocessor(env.observation_space)
            

    def _normalize_observations(self, obs, clip_range=[-50.0, 50.0]):
        """
        Normalizes the observations to approximately N(0,1)
        by running mean & std with clipping

        Args:
            obs: (tensor) observations
        """
        self._rms.update(obs)
        norm_x = torch.clamp((obs - self._rms.mean) / self._rms.std, \
                              min(clip_range), max(clip_range))
        return norm_x


    def _encode_observations(self, obs):
        """
        Encode the observations to be suitable for env.observation_space

        Args:
            obs_space: (gym.Space) type of obsrervation space; Box, Discrete, MultiDiscrete
            obs: (tensor) observation
        """
        return self._encoder(obs)


    def _preprocess(self, obs):
        """
        Preprocess observations
        - normalize
        - encode
        """
        if self._normalize_obs_flag:
            obs = self._normalize_observations(obs)
        obs = self._encode_observations(obs)
        return obs
        
        
    def step(self, obs, **extra_feed):
        """
        Compute the next action(s) given the observation(s)

        Args:
            obs: (tensor) observation(s)

        Returns:
            action: (tensor)
            value: (tensor)
            next_state: (tensor)
            neglogp: (tensor)
        """
        # 1) preprocess
        obs = self._preprocess(obs)
        obs = obs.float()
        # 2) compute action
        policy_logits = self._policy_net(obs[:,0:1,:])
        return policy_logits






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



if __name__ == '__main__':
    net = PolicyWithValue(
        policy_net = 'simplecnn',
        normalize_observations=True
    )
    import numpy as np
    obs = np.random.randn(1, 2)
    obs_norm = net._normalize_observations(torch.from_numpy(obs))
    print(obs_norm.numpy())