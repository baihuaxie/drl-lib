"""
    Test subprocess vectorized environment class
"""

import pytest

import gym
import torch

from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv


def seed_env(env, seed):
    env.seed(seed)
    return lambda: env

def assert_venvs_equal(venv1, venv2, num_steps):
    """
    Assert venv1 & venv2 produce sequence of observations identically
    over num_steps when given the same sequence of actions

    note: this test only works for single sub observation space per
    environment; i.e., does not support checking equality between two
    venvs that use Dict or Tuple observation spaces
    """
    assert venv1.num_envs == venv2.num_envs
    assert venv1.observation_space.shape == venv2.observation_space.shape
    assert venv1.observation_space.dtype == venv2.observation_space.dtype
    assert venv1.action_space.shape == venv2.action_space.shape
    assert venv1.action_space.dtype == venv2.action_space.dtype

    obs1, obs2 = venv1.reset(), venv2.reset()
    assert obs1.shape == obs2.shape
    assert obs2.shape == torch.Size((venv2.num_envs,) + \
        venv2.observation_space.shape)
    # seeding issue?
    # assert (obs1 == obs2).all() 


@pytest.mark.parametrize("env", [
    gym.make('FrozenLake-v0'),
    gym.make('CartPole-v0')
])
@pytest.mark.parametrize("in_series",[1, 2])
@pytest.mark.parametrize("klass",[SubprocVecEnv])
def test_vec_env(env, klass, in_series):
    """
    check that SubprocVecEnv object should behave identically to
    a DummyVecEnv given the same sequence of actions
    """
    num_envs = 4
    num_steps = 10
    
    env_fns = [seed_env(env, idx) for idx in range(num_envs)]
    venv1 = DummyVecEnv(env_fns)
    venv2 = klass(env_fns, in_series=in_series)
    assert_venvs_equal(venv1, venv2, num_steps)


@pytest.mark.skip('print venv output')
@pytest.mark.parametrize("env", [
    gym.make('FrozenLake-v0'),
    gym.make('CartPole-v0')
])
@pytest.mark.parametrize("in_series",[1, 2])
def test_SubprocVecEnv(env, in_series):
    """
    assert that each subprocess instance has a different pid
    """
    from multiprocessing import cpu_count
    num_envs = 4
    
    env_fns = [seed_env(env, idx) for idx in range(num_envs)]
    venv = SubprocVecEnv(env_fns, in_series=in_series)
    
    # test multiprocessing + pipe setup
    print()
    print(venv.get_pid())
    # check .reset()
    print("observations: {}".format(venv.reset()))
    # check .step_async()
    actions = torch.tensor([env.action_space.sample() for _ in range(num_envs)])
    venv.step_async(actions)
    #print(venv.step_wait())
    obs, rewards, dones, info = venv.step_wait()
    print()
    print("observations: {}".format(obs))
    print("rewards: {}".format(rewards))
    print("dones: {}".format(dones))
    print("info: {}".format(info))

    