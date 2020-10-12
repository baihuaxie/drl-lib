"""
Miscellaneous utilities
"""

try:
    import MPI
except ImportError:
    MPI = None

import torch
from gym.spaces import Discrete, Box, MultiDiscrete



def set_global_seed(seed):
    """
    set seed globally in distributed setting

    Args:
        seed: (float) random seed
    """
    if MPI is None:
        rank = 0
    else:
        rank = MPI.COMM_WORLD.Get_rank()
    # calculate seed(s) for each rank
    myseed = seed + 1000 * rank if seed is not None else None
    # set seed(s) by torch.manual_seed()
    if torch.cuda.is_available():
        torch.cuda.manual_seed(myseed)
    else:
        torch.manual_seed(myseed)


def encode_observation(ob_space):
    """
    Encode observation in the way that is appropriate to the environment's observation space

    Args:
        ob_space: (gym.space) observation space type; Box, Discrete or MultiDiscrete
    """
    if isinstance(ob_space, Box):
        pass




    