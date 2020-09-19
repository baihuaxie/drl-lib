"""
Utilities for logging, etc.
"""
# standard imports
import sys
import re
import multiprocessing
from importlib import import_module
from collections import defaultdict
# custom libs
import logger
import gym
# user libs
from common.cmd_util import make_env



# gym environment type
_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    env_type = env.entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)


def configure_logger(log_path=None, **kwargs):
    """
    Configure logger

    Args:
        log_path: (str) path to logger
        **kwargs: pointer to additional arguments if log_path is not given
    """
    if log_path is not None:
        logger.configure(log_path)
    else:
        logger.configure(**kwargs)



def get_env_type(args):
    """
    Get environment type from command line arguments

    Args:
        args: (argparse.ArgumentParser) parsed commandline arguments
        _game_envs: (defaultdict) dictionary containing {env_type: env_id} pairs

    Return:
        env_type: (str) environment type
        env_id: (str) environment id / name
    """
    # get env_id from arguments
    env_id = args.env

    if args.env_type is not None:
        # if env_type is available in arguments
        return args.env_type, env_id

    # re-parse the gym registry for possible addition of environments
    for env in gym.envs.registry.all():
        # env.entry_point e.g. gym.envs.algorithmic:CopyEnv
        # env_type would be 'algorithmic'
        env_type = env.entry_point.split(':')[0].split('.')[-1]
        _game_envs[env_type].add(env.id)

    if env_id in _game_envs.keys():
        env_type = env_id
        env_id = [g for g in _game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break
        if ':' in env_id:
            env_type = re.sub(r':.*', '', env_id)
        assert env_type is not None, 'env_id {} is not recognized in env types {}'.format(env_id, _game_envs.keys())

    return env_type, env_id


def get_default_network(env_type):
    """
    Return the default network type

    Args:
        env_type: (str) environment type

    Returns:
        (str) name of network type
    """
    if env_type in ['atari', 'retro']:
        return 'cnn'
    else:
        return 'mlp'


def get_alg_module(alg, submodule=None):
    """
    Return the main submodule from the agent directory
    - main submodule contains the learn function for agent

    Args:
        alg: (str) name of agent / algorithm
        submodule: (str) name of main submodule

    Returns:
        alg_module: (module??) agent module
    """
    submodule = submodule or alg

    try:
        alg_module = import_module('.'.join(['baselines', alg, submodule]))
    except ImportError:
        print('Unable to import agent {}!'.format(alg))

    return alg_module


def get_learn_function(alg):
    """
    Return the learn method of agent
    - every agent's module need to contain a function named 'learn'

    Args:
        alg: (str) name of agent / algorithm
    """
    return get_alg_module(alg).learn


def get_learn_function_defaults(alg, env_type):
    """
    Return parameter defaults of given environment type (if available) for the learn function of the algorithm
    - an agent may have an agent.defaults submodule, which contains functions like atari() that returns a dictionary of default settings

    Args:
        alg: (str) name of agent / algorithm
        env_type: (str) environment type

    Returns:
        kwargs: (dict) a dictionary that contains default parameters
    """
    try:
        # try to get defaults if the agent has one
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)() 
    except (ImportError, AttributeError):
        # else just set defaults to empty dictionary
        kwargs = {}
    return kwargs


def build_env(args):
    """
    Build gym environment

    Args:
        args: (argparse.ArgumentParser) parsed command line arguments
    """

    # number of cpu cores
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin':
        # for 'darwin' system halve the processor cores
        ncpu //= 2
    # set nenv = cpu if args.num_env == 0 or None
    nenv = args.num_env or ncpu
    # agent
    alg = args.alg
    # seed
    seed = args.seed
    # environment type & id
    env_type, env_id = get_env_type(args)

    if env_type in ['atari', 'retro']:
        if alg == 'deepq':
            env = make_env(env_id, env_type, seed=seed, wrapper_kwarg={'frame_stack': True})
