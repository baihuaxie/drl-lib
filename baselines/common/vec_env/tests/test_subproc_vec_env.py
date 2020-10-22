"""
    Test subprocess vectorized environment class
"""

import pytest

import gym

from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv


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
    print(obs1)
    print(obs2)


@pytest.mark.parametrize("env", [
    gym.make('FrozenLake-v0'),
    gym.make('CartPole-v0')
])
@pytest.mark.parametrize("klass",[SubprocVecEnv])
def test_vec_env(env, klass):
    """
    check that SubprocVecEnv object should behave identically to
    a DummyVecEnv given the same sequence of actions
    """
    def seed_env(env, seed):
        env.seed(seed)
        return lambda: env
    num_envs = 3
    num_steps = 10
    
    env_fns = [seed_env(env, idx) for idx in range(num_envs)]
    venv1 = DummyVecEnv(env_fns)
    venv2 = klass(env_fns)
    assert_venvs_equal(venv1, venv2, num_steps)

    