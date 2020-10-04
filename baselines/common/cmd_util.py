"""
Utilities for command line

- arg_parser(): creates an empty argument parser object to hold command line arguments
- common_arg_parser(): add command line arguments with default values, help strings, etc
- make_env(): return an environment as set up specified by command line arguments
"""

import argparse

import gym
from gym.wrappers import FlattenObservation, FilterObservation
from common import wrappers_retro, wrappers
from common.wrappers_atari import make_atari, wrap_deepmind

def arg_parser():
    """
    Returns an empty argparse.ArgumentParser object
    """
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


def common_arg_parser():
    """
    Returns an argparse.ArgumentParser object with common arguments for atari and mujoco environments
    """

    parser = arg_parser()
    parser.add_argument("--env", help="environment ID", type=str, default="CartPole")
    parser.add_argument("--env_type", help="type of environment", type=str, default=None)
    parser.add_argument("--seed", help="random number generator seed", type=int, default=None)
    parser.add_argument("--alg", help="algorithm", type=str, default="ppo")
    parser.add_argument("--num_timesteps", help="maximum number of time steps per iteration", type=float, default=1e6)
    parser.add_argument("--network", help="network type for function approximation, (mlp, cnn, lstm, cnn_lstm, conv_only)", type=str, default=None)
    parser.add_argument("--gamestate", help="game state to load", type=str, default=None)
    parser.add_argument("--num_env", help="number of environment copies run in parallel", type=int, default=None)
    parser.add_argument("--reward_scale", help="reward scale factor", type=float, default=1.0)
    parser.add_argument("--save_path", help="path to save trained model", type=str, default=None)
    parser.add_argument("--save_video_interval", help="save video every x steps", type=int, default=0)
    parser.add_argument("--save_video_length", help="length of recorded video in seconds", type=int, default=200)
    parser.add_argument("--log_path", help="directory to save learning curve data", type=str, default=None)
    parser.add_argument("--play", help="", default=False, action="store_true")

    return parser

def make_env(env_id, env_type, mpi_rank=0, subrank=0, seed=None, reward_scale=1.0, gamestate=None, flatten_dict_observations=True,
             wrapper_kwargs=None, env_kwargs=None, logger_dir=None, initializer=None):
    """
    Make environment

    Args:
        env_id: (str) environment id e.g. 'Reacher-v2'
        env_type: (str) environment type e.g. 'atari'
        mpi_rank: (int) rank for mpi; default=0 (disabled on windows for lack of MPI support from pytorch)
        subrank: (int) subrank; default=0 (disabled on windows for lack of MPI support from pytorch)
        seed: (int) random seed
        reward_scale: (float) scale factor for reward (== discount factor??); default=1.0
        gamestate: (??) game state to load (for retro games only)
        flatten_dict_observations: (??) ??
        wrapper_kwargs: (dict) dictionary of parameter settings for wrapper
        env_kwargs: (dict) dictionary of parameter settings for environment
        logger_dir: (str) logger path
        initializer: (??) ??

    Returns:
        env: (Env) the set-up environment
    """
    if initializer is not None:
        initializer(mpi_rank=mpi_rank, subrank=subrank)

    wrapper_kwargs = wrapper_kwargs or {}
    env_kwargs = env_kwargs or {}

    if ':' in env_id:
        raise ValueError("env_id {} does not conform to accepted format!".format(env_id))

    if env_type == 'atari':
        # make atari environments with a wrapper function
        env = make_atari(env_id)
    elif env_type == 'retro':
        raise ValueError("retro environments not supported yet!")
    else:
        # make a gym environment with parameter settings
        env = gym.make(env_id, **env_kwargs)

    # flatten the observation space
    if flatten_dict_observations and isinstance(env.observation_spaces, gym.spaces.Dict):
        env = FlattenObservation(env)

    # add seed to env
    env.seed(seed + subrank if seed is not None else None)

    # set up Monitor (TBD)

    if env_type == 'atari':
        env = wrap_deepmind(env, **wrapper_kwargs)
    elif env_type == 'retro':
        if 'frame_stack' not in wrapper_kwargs:
            wrapper_kwargs['frame_stack'] = 1
        # wrap retro games
        env = wrappers_retro.wrap_deepmind_retro(env, **wrapper_kwargs)

    if isinstance(env.action_space, gym.spaces.Box):
        # if action_space is Box type, clip the action values to be within the box's boundaries
        env = wrappers.ClipActionsWrapper(env)

    if reward_scale != 1:
        # if reward scaling factor is used, scale the rewards accordingly
        # very important feature for PPO
        env = wrappers.RewardScalerWrapper(env, reward_scale)

    return env
