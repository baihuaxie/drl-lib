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
def env_fn():
    """
    Return a gym.env object
    """
    return gym.make('CartPole-v0')

@pytest.fixture
def init_obj(env_fn):
    """
    Instantiate a PolicyWithValue class object
    """
    obs = np.random.randn(1, 2)
    MyNet = PolicyWithValue(
        policy_net = 'simplecnn',
        env = env_fn,
        obs_shape = obs.shape,
        normalize_observations=True
    )
    return MyNet


@pytest.mark.skip('this test is wrong, delete later')
def test_normalize_obs(init_obj):
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


@pytest.mark.skip('not yet ready')
def test_step(init_obj):
    """
    Make one step
    """
    net = init_obj
    obs = torch.rand(1, 8, 8, 3, dtype=torch.float32)
    net.step(obs)

