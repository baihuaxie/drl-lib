"""
    test if the policy and value networks are properly trained
"""

import pytest

import sys
sys.path.append('../../')

import torch
import gym
import numpy as np

from common.policy_util import PolicyWithValue, build_policy, \
    postprocess


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
        action, value, neglogp = net.step(obs)
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
        action, value, neglogp = net.step(obs)
        obs, reward, _, _ = env.step(action.item())
        # env.render()
        print("action: {}".format(action))
        print("neglogp: {}".format(neglogp))
        print("value: {}".format(value))


def test_build_policy():
    """
    Test policy network builder function
    """
    env = gym.make('FrozenLake-v0')
    network = 'mlp'
    network_kwargs = {}
    build_policy(env, network,**network_kwargs)


def test_postprocess():
    """
    test postprocessing network outputs
    """
    # x1 is single value int, convert to 1D tensor
    x1 = 10
    assert postprocess(x1).shape == torch.Size([1])
    # x2 is a 0D tensor, convert to 1D tensor
    x2 = torch.tensor(3)
    assert postprocess(x2).shape == torch.Size([1])
    # x3 is a 1D tensor, no conversion
    x3 = torch.tensor([1])
    assert postprocess(x3).shape == torch.Size([1])
    # x4 is a list of floats, convert to 1D tensor
    x4 = [.2, .4, .7]
    assert postprocess(x4).shape == torch.Size([3])
    # x5 is a 1D tensor of multiple elements, no conversion
    x5 = torch.rand(3,)
    assert (postprocess(x5) == x5).all()

