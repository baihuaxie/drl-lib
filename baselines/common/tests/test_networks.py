"""
    test if the policy and value networks are properly trained
"""

import pytest

import sys
sys.path.append('../../')

import torch
import torch.nn as nn
import gym

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
    obs = torch.rand(1, 8, 8, 3, dtype=torch.float32)
    MyNet = PolicyWithValue(
        policy_net = 'simplecnn',
        env = env_fn,
        observations = obs,
    )

    assert isinstance(MyNet._policy_net, nn.Module)
    assert isinstance(MyNet._value_net, nn.Module)
    assert isinstance(MyNet._env, gym.Env)
    assert isinstance(MyNet._X, torch.FloatTensor)
 
    return MyNet



if __name__ == '__main__':
    init_object(env)