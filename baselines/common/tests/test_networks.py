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
        policy_net = 'mlp',
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
        policy_net = 'mlp',
        env = env_fn_box,
        normalize_observations=True
    )
    return MyNet


def test_step_discrete(init_obj_discrete, env_fn_discrete):
    """
    Step agent in a Discrete environment
    """
    net = init_obj_discrete
    env = env_fn_discrete

    obs = env.reset()

    for _ in range(10):
        obs = torch.tensor([obs], dtype=torch.int64)
        action, neglogp, value = net.step(obs)
        obs, reward, _, _ = env.step(action.item())
        # env.render()
        print("action: {}".format(action.item()))
        print("neglogp: {}".format(neglogp.item()))
        print("value: {}".format(value.item()))


def test_step_box(init_obj_box, env_fn_box):
    """
    Step agent in a Box environment
    """
    net = init_obj_box
    env = env_fn_box

    obs = env.reset()

    for _ in range(10):
        obs = torch.tensor(obs, dtype=torch.float32)
        print(obs)
        action, neglogp, value = net.step(obs)
        obs, reward, _, _ = env.step(action.item())
        # env.render()
        print("action: {}".format(action))
        print("neglogp: {}".format(neglogp))
        print("value: {}".format(value))

