"""
Encapsulation of policy and value function estimations with shared parameters
"""

import sys
sys.path.append('../')

import torch
import torch.distributions as distr
import torch.nn as nn
from gym.spaces import Discrete, Box, MultiDiscrete

from baselines.common.networks.get_networks import get_network_builder
from baselines.common.preprocessor import RunningMeanStd, OneHotPreprocessor

class PolicyWithValue(object):
    """
    Common interface for policy and value networks

    Example:

    net = PolicyWithValue(policy_network, env, **extra_kwargs)

    # make one agent step by taking in a single observation tensor
    action, value_estimate, neglogp = net.step(observation)
    """
    def __init__(self, env, policy_net, value_net=None,
                 estimate_q=False, normalize_observations=False):
        """
        Constructor

        Args:
            env:            (gym.Env) environment, required
            policy_net:     (str) type of policy network, required
            value_net:      (str) type of value network; if None or 'shared', default
                            value_net = policy_net (plus output layers)
            estimate_q:     (bool) if True returns an estimate for q-value instead
                            only valid if env.action_space is Discrete
            normalize_observations: (bool) if True normalize observations by running
                            mean & std; does not support Discrete space
        """
        super().__init__()

        # get environment
        self._env = env
        # get env type flags
        self._obs_is_Discrete = True if isinstance(self._env.observation_space, Discrete) \
            else False
        self._obs_is_Box = True if isinstance(self._env.observation_space, Box) else False
        self._act_is_Discrete = True if isinstance(self._env.action_space, Discrete) \
            else False
        self._act_is_Box = True if isinstance(self._env.action_space, Box) else False

        # get policy / value network input dimension
        if self._obs_is_Discrete:
            self._obs_dim = self._env.observation_space.n
        elif self._obs_is_Box:
            self._obs_dim = self._env.observation_space.shape[0]
        else:
            raise TypeError("Observation space type {} not supported!".format( \
                            self._env.observation_space))
        # get policy network output dimension
        if self._act_is_Discrete:
            self._act_dim = self._env.action_space.n
        elif self._act_is_Box:
            self._act_dim = self._env.action_space.shape
        else:
            raise TypeError("Action space type {} not supported!".format( \
                            self._env.action_space))
        # get value network output dimension
        if estimate_q:
            assert self._act_is_Discrete, "Can not estimate Q value for non-Discrete env"
            self._value_latent_dim = self._env.action_space.n
        else:
            self._value_latent_dim = 1 

        # build policy network
        self._policy_latent_dim = self._act_dim
        self._policy_net = get_network_builder(policy_net)(self._obs_dim, \
            self._policy_latent_dim)
        # build probability distribution layer from action space
        self.pdtype = make_pdtype(self._env.action_space)
        # build value network
        if value_net is None or value_net == 'shared':
            # value & policy nets shared all but last output layer(s)
            self._value_net = nn.Sequential(
                self._policy_net,
                nn.Linear(self._policy_latent_dim, self._value_latent_dim)
            )       
        else:
            self._value_net = get_network_builder(value_net)(self._obs_dim, \
                self._value_latent_dim)

        # build pre-processors
        if self._obs_is_Box:
            self._rms = RunningMeanStd(shape=torch.Size(self._env.observation_space.shape))
            self._encoder = None
        elif self._obs_is_Discrete:
            self._rms = None
            self._encoder = OneHotPreprocessor(self._env.observation_space)

        # additional flags
        if normalize_observations:
            assert not self._obs_is_Discrete
        self._normalize_obs_flag = normalize_observations


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
        return torch.tensor(norm_x)

    def _preprocess(self, obs):
        """
        Preprocess observations
        - normalize
        - encode
        """
        if self._normalize_obs_flag:
            obs = self._normalize_observations(obs)
        if self._encoder:
            obs = self._encoder(obs)
        return obs


    def step(self, observation):
        """
        Compute the next action(s) given the observation(s)

        Args:
            observation: (tensor) a tensor of observation data sample
                         supports single sample only (not batched)

        Returns:
            action: (tensor) sampled action
            value: (tensor) value / q-value estimation
            neglogp: (tensor) negative log-likelihood of given sampled action
        """
        # preprocess
        obs = self._preprocess(observation)
        # get action probabilities
        policy_latent = self._policy_net(obs)
        # build distribution
        pd = self.pdtype(policy_latent)
        # sample action
        # Q: how to ensure sampled action dtype == env.action_space.dtype?
        action = pd.sample()
        # get returned values
        neglogp = -pd.log_prob(action)
        value = self._value_net(obs)

        return map(postprocess, (action, value, neglogp))


def postprocess(x):
    """
    Post-process
    - turn all returns into tensors of proper shape
    - for single-value return, convert to 1D (not 0D) tensor
    """
    x = torch.tensor(x)
    if x.dim() == 0:
        x = x.reshape((1,))
    return x

def build_policy(env, policy_network, **policy_kwargs):
    """
    Policy / Value network builder

    Args:
        env: (gym.Env class) environment
        policy_network: (str) name for the policy network type
        **policy_kwargs: pointer to arguments to instantiate the policy network

    Returns:
        (PolicyWithValue class) a class object that contains the instantiated policy network
                                along with methods to return actions / value estimates given
                                current observations
    """
    return PolicyWithValue(env, policy_network, **policy_kwargs)


def make_pdtype(act_space):
    """
    Return a parameterized probability distribution suitable for act_space

    Args:
        act_space: (env.space) action_space
    """
    if isinstance(act_space, Box):
        assert len(act_space.shape) == 1
        return distr.Normal
    if isinstance(act_space, Discrete):
        return distr.Categorical
    

if __name__ == '__main__':
    import gym
    env = gym.make('CartPole-v0')
    net = PolicyWithValue(
        policy_net = 'mlp',
        env = env,
        normalize_observations=True
    )
    obs = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32)
    action, neglogp, value = net.step(obs)
