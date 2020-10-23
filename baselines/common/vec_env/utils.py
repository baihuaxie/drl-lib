"""
    utilities for vectorized environments
"""
from collections import OrderedDict
import gym


def obs_space_info(obs_space):
    """
    Obtain observation space info as a dictionary

    Supports gym.spaces objects:
    - Discrete
    - Box
    - MultiDiscrete
    - MultiBinary
    - Dict
    - Tuple

    Args:
        obs_space: (gym.spaces)
    
    Returns:
        keys: (list) a list of observation space keys (str) for each subspace
        shapes: (dict) a dict of key: shape (tuple)
        dtypes: (dict) a dict of key: dtype (np.dtype)
    """
    # 'spaces' is a dict object of {key: gym.spaces object}
    # the value is of simpler space types (Dicrete, Box, ...)
    if isinstance(obs_space, gym.spaces.Dict):
        assert isinstance(obs_space.spaces, OrderedDict)
        spaces = obs_space.spaces
    elif isinstance(obs_space, gym.spaces.Tuple):
        # Tuple object can have a tuple or list of spaces
        assert isinstance(obs_space.spaces, (tuple, list))
        spaces = {i: obs_space.spaces[i] for i in range(len(obs_space.spaces))}
    else:
        # for simpler space object use a null key
        spaces = {None: obs_space}

    keys = []
    shapes = {}
    dtypes = {}
    # Q: how to add space value range?
    # e.g., for Discrete add space.n; for Box add space.high & space.low?

    for key, space in spaces.items():
        keys.append(key)
        shapes[key] = space.shape
        dtypes[key] = space.dtype

    return keys, shapes, dtypes


def obs_dict_to_tensor(obs_dict):
    """
    Convert {None: obs (tensor)} to tensor when original observation space is not
    Dict or Tuple; else do nothing
    """
    if set(obs_dict.keys()) == {None}:
        return obs_dict[None]
    return obs_dict