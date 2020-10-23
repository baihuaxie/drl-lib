"""
    test dummy vectorized environment class
"""

import pytest

import gym

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv


def seed_env(env, seed):
    env.seed(seed)
    return lambda: env

@pytest.mark.skip
@pytest.mark.parametrize("env", [
    gym.make('FrozenLake-v0'),
    gym.make('CartPole-v0')
])
def test_init_dummy_vec_env(env):
    """
    test instantiate a DummyVecEnv object
    """
    num_envs = 5
    env_fns = [seed_env(env, i) for i in range(num_envs)]
    env = DummyVecEnv(env_fns)
    # 1) check environment properties
    print(env.num_envs)
    print(env.observation_space.shape)
    print(env.observation_space.dtype)
    print(env.action_space.shape)
    print(env.action_space.dtype)
    # 2) reset all environments
    # should return a dict of env_id: initial observation
    print(env.reset())
    # 3) step the environments synchronously
    # 4) step the environemnents asynchronously
    # 5) wait for environments to finish step asynchronously
    # 6) close all environments
    # 7) render all environments
    # 8) capture image from environments

@pytest.mark.skip
@pytest.mark.parametrize("env, num_envs", [
    (gym.make('FrozenLake-v0'), 5),
    (gym.make('FrozenLake-v0'), 1),
    (gym.make('CartPole-v0'), 5),
    (gym.make('CartPole-v0'), 1),
])
def test_step_async(env, num_envs):
    """
    test asynchronous stepping environment
    """
    env_fns = [seed_env(env, i) for i in range(num_envs)]
    env = DummyVecEnv(env_fns)
    actions = []
    for _ in range(num_envs):
        action = env.action_space.sample()
        actions.append(action)
    # pass to step_async as a single np.ndarray
    if num_envs == 1:
        print(env.step_async(action))
    print(env.step_async(actions))


@pytest.mark.parametrize("env, num_envs", [
    (gym.make('FrozenLake-v0'), 5),
    #(gym.make('FrozenLake-v0'), 1),
    (gym.make('CartPole-v0'), 5),
    #(gym.make('CartPole-v0'), 1),
])
def test_step_wait(env, num_envs):
    """
    test step_wait() returns
    """
    env_fns = [seed_env(env, i) for i in range(num_envs)]
    env = DummyVecEnv(env_fns)
    actions = []
    for _ in range(num_envs):
        action = env.action_space.sample()
        actions.append(action)
    env.reset()
    env.step_async(actions)
    obs, rewards, dones, info = env.step_wait()
    print()
    print("observations: {}".format(obs))
    print("rewards: {}".format(rewards))
    print("dones: {}".format(dones))
    print("info: {}".format(info))



