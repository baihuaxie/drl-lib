"""
    test if the policy and value networks are properly trained
"""

import pytest

import sys
sys.path.append('../../')

import torch
import gym
import numpy as np

from common.policy_util import PolicyWithValue

@pytest.fixture
def env_fn_discrete():
    """
    Return a gym.env object
    """
    return gym.make('FrozenLake-v0')

@pytest.fixture
def init_obj_discrete(env_fn_discrete):
    """
    Instantiate a PolicyWithValue class object
    """
    MyNet = PolicyWithValue(
        policy_net = 'simplecnn',
        env = env_fn_discrete,
        normalize_observations=False
    )
    return MyNet

@pytest.fixture
def env_fn_box():
    """
    Return a gym.env object
    """
    return gym.make('CartPole-v0')

@pytest.fixture
def init_obj_box(env_fn_box):
    """
    Instantiate a PolicyWithValue class object
    """
    MyNet = PolicyWithValue(
        policy_net = 'simplecnn',
        env = env_fn_box,
        normalize_observations=True
    )
    return MyNet


@pytest.mark.skip('this test is wrong, delete later')
def test_normalize_obs(init_obj_discrete):
    """
    normalize observations by running mean & std
    """
    net = init_obj
    x = np.empty((0, 2))
    for idx in range(1, 10):
        # new random observations
        obs = np.random.randn(idx, 2)

        x = np.concatenate([x, obs], axis=0)
        ms1 = [x.mean(axis=0), x.std(axis=0)]

        obs_norm = net._normalize_observations(torch.from_numpy(obs))
        obs_norm = obs_norm.numpy()
        ms2 = [obs_norm.mean(axis=0), obs_norm.std(axis=0)]
    print(ms1)
    print(ms2)
    assert np.allclose(ms1, ms2)


def test_step_discrete(init_obj_discrete, env_fn_discrete):
    """
    Make one agent step in a Discrete environment
    """
    net = init_obj_discrete
    env = env_fn_discrete
    # 20 random observations from Discrete space 
    obs = torch.randint(0, env.observation_space.n, (1, 20,), dtype=torch.int64)

    print(obs)
    print(net.step(obs).shape)

