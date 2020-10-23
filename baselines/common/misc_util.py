"""
Miscellaneous utilities
"""

try:
    import MPI
except ImportError:
    MPI = None

import torch
import numpy as np
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


def dtype_to_torch(dtype):
    """
    Convert an np.dtype to torch.dtype

    Args:
        dtype: (np.dtype) a np datatype
    """
    import numpy as np
    import torch

    dict_dtype = {
        np.dtype('int64') : torch.int64,
        np.dtype('float32') : torch.float32
    }
    return dict_dtype[dtype]


def numpy_to_torch(x):
    """
    Convert a scalar or np.ndarray to torch.tensor
    """
    if np.isscalar(x):
        return torch.tensor(x)
    else:
        return torch.from_numpy(x)




    