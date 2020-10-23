"""
    Test PPO runner
"""

import pytest

import sys

import gym
import torch

from baselines.ppo.runner import Runner
from baselines.common.policy_util import build_policy


def test_build_runner():
    """ instantiate runner """
    env = gym.make('FrozenLake-v0')
    policy = build_policy(env, 'mlp')
    runner = Runner(env=env, model=policy, nsteps=100, \
        gamma=1.0, lamb=1.0)
    return env, runner

def test_initial_obs():
    """
    Check if a runner's obs property can accept new observation
    - Discrete & Box environments
    """
    # Discrete env
    env = gym.make('FrozenLake-v0')
    policy = build_policy(env, 'mlp')
    runner = Runner(env=env, model=policy)
    env.reset()
    obs, _, _, _ = env.step(env.action_space.sample())
    runner.obs = torch.tensor(obs)
    assert (runner.obs.numpy() == obs).all()
    # Box env
    env = gym.make('CartPole-v0')
    policy = build_policy(env, 'mlp')
    runner = Runner(env=env, model=policy)
    env.reset()
    obs, _, _, _ = env.step(env.action_space.sample())
    runner.obs = torch.tensor(obs)
    assert (runner.obs.numpy() == obs).all()


def test_run_episode_Discrete():
    """
    Take steps in Discrete environment
    """
    env = gym.make('FrozenLake-v0')
    policy = build_policy(env, 'mlp')
    runner = Runner(env=env, model=policy)
    env.reset()
    runner.run()


def test_run_episode_Box():
    """
    Take steps in Box environment 
    """
    env = gym.make('CartPole-v0')
    policy = build_policy(env, 'mlp')
    runner = Runner(env=env, model=policy)
    env.reset()
    mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, \
        mb_neglogps = runner.run()
    print(mb_obs)


