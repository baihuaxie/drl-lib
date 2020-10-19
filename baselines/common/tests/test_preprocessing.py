"""
    Test observation pre-processing
    - running mean & std calculation
    - one-hot encoding of Discrete observations
"""

import pytest

import sys
sys.path.append('../../')

import torch
import gym
import numpy as np

import gym

from common.preprocessor import RunningMeanStd, OneHotPreprocessor


def test_runningmeanstd():
    """
    check running mean & std computations
    """
    # compute running average on axis=0 for 3 tensors x1, x2, x3
    # e.g., for shape batch x H x W x C, axis=0 is summary over batch
    for (x1, x2, x3) in [
        (np.random.randn(3), np.random.randn(4), np.random.randn(5)),
        (np.random.randn(3,2), np.random.randn(4,2), np.random.randn(5,2))
    ]:
        # x1.shape[0] is the axis=0 for summary
        rms = RunningMeanStd(shape=x1.shape[1:])
        print(x1.shape)

        x = np.concatenate([x1, x2, x3], axis=0)
        ms1 = [
            np.asarray(x.mean(axis=0)),
            np.asarray(x.std(axis=0))
        ]
        print(ms1)

        rms.update(x1)
        rms.update(x2)
        rms.update(x3)
        ms2 = [rms.mean.numpy(), rms.std.numpy()]
        print(ms2)

        assert np.allclose(ms1, ms2)


def test_runningmeanstd_discrete_obs():
    """
    check call in Discrete environment
    """
    env = gym.make('FrozenLake-v0')
    rms = RunningMeanStd(shape=torch.Size((1,) + \
        (env.observation_space.n,)))
    obs = env.reset()
    # Discrete env requires one-hot encoding
    encoder = OneHotPreprocessor(env.observation_space)
    for _ in range(10):
        obs = torch.tensor([obs], dtype=torch.int64)
        obs = encoder(obs)
        rms.update(obs)
        obs, _, _, _ = env.step(env.action_space.sample())


def test_runningmeanstd_box_obs():
    """
    check call in Box environment
    """
    env = gym.make('CartPole-v0')
    rms = RunningMeanStd(shape= \
        torch.Size(env.observation_space.shape))
    obs = env.reset()
    for _ in range(10):
        obs = torch.tensor(obs, dtype=torch.float32)
        rms.update(obs)
        obs, _, _, _ = env.step(env.action_space.sample())

    
def test_onehotpreprocessor():
    """
    test one-hot encoding
    """
    # discrete observations as 2-d tensor of batch x timestep
    obs = torch.randint(0, 10, (20,))

    encoder = OneHotPreprocessor(gym.spaces.Discrete(10))
    # take non-zero elements' indices of encoded observation
    # along one-hot vector axis (axis=2)
    indices = encoder(obs).nonzero(as_tuple=True)[1]

    assert (obs == indices).all() 