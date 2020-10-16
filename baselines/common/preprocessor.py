"""
    Pre-process observations
    - normalization
    - encoding
"""

import torch
import numpy as np
from gym.spaces import Box, Discrete, MultiDiscrete

try:
    from mpi4py import MPI
except ImportError:
    MPI = None
    print("Can not import MPI!")


class RunningMeanStd(object):
    """
    Running Mean & Std class
    """
    def __init__(self, shape=(), epsilon=1e-5):
        """
        Constructor

        Args:
            shape: (tuple) a tuple of integers for the shape of input tensor
                   axis=0 is the summary axis
        """
        self._shape = shape[1:]
        # use a shape of (1, shape) to initialize sum & sumsq
        # b.c. size of axis=0 is insignificant for initialization
        # this enables instantiation by env.observation_space.shape
        self._sum = torch.zeros(((1,) + self._shape), dtype=torch.float32)
        self._sumsq = torch.zeros(((1,) + self._shape), dtype=torch.float32)
        self._count = epsilon

        # mean & std tensors omits the summary axis
        self._mean = torch.zeros(self._shape, dtype=torch.float32)
        self._std = torch.ones(self._shape, dtype=torch.float32)

    def update(self, x):
        """
        Update running sum and sumsq by new tensor x
        - axis=0 in x is the summary axis
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        self._count += x.shape[0]
        self._sum = torch.sum(torch.cat((self._sum, x), dim=0), dim=0, keepdim=True)
        self._sumsq = torch.sum(torch.cat((self._sumsq, torch.square(x)), dim=0), dim=0, keepdim=True)

    @property
    def mean(self):
        """
        Return running mean
        """
        # use torch.squeeze to remove redundant axis=0
        # & match self._mean's initial shape
        self._mean = torch.squeeze(self._sum, dim=0) / self._count
        return self._mean

    @property
    def std(self):
        """
        Return running standard deviation
        """
        self._std = torch.sqrt(torch.squeeze(self._sumsq, dim=0) / self._count \
                    - torch.square(self._mean))
        return self._std


class OneHotPreprocessor(object):
    """
    Class for one-hot encoding of observations (Discrete / Multi-Discrete spaces)
    """
    def __init__(self, obs_space):
        """
        Constructor

        Args:
            obs_space: (gym.Env.observation_space) Discrete, Box or Multi-Discrete
            episode_mode: (bool) if true treat input obs as 1-d tensor containing
                          multiple observations over an episode
        """
        if isinstance(obs_space, Box):
            raise ValueError("one-hot encoding not supported on Box space!")
        self._obs_space_n = obs_space.n


    def _check_type(self, obs):
        """
        Check if input observation tensor is suitable for Discrete spaces

        Args:
            obs: (tensor) a 2D tensor of observations (int64) in form = batch x timesteps

        Returns:
            (bool) if true suggests obs tensor can be one-hot encoded
        """
        if obs.dtype == torch.int64 and len(obs.shape) == 2:
            pass
        else:
            raise TypeError("Observation type error! Expected 2D tensor of torch.int64\
                but got {}D tensor of {}".format(len(obs.shape), obs.dtype))


    def encode(self, obs):
        """
        Encode the observations by one-hot encoding

        Args:
            obs: (tensor) a 2D tensor of observations (int64) in form = batch x timesteps
        """
        # returns a 3-d tensor of dimension = batch x timestep x obs_space_n
        arr = torch.zeros(tuple(obs.shape) + (self._obs_space_n,), dtype=int)
        # create a conditional boolean mask for one-hot encoding
        # if obs[i] == j set mask[i][j] = True else False
        mask = torch.BoolTensor(
            [
                [
                    [
                    True if i == obs[j][k] else False for i in range(arr.shape[2])
                    ]
                    for k in range(arr.shape[1])
                ]
                for j in range(arr.shape[0])
            ]
        )
        # one-hot encoding
        arr[mask] = 1
        return arr


    def __call__(self, obs):
        """
        Call method

        Args:
            obs: (tensor) observations
        """
        self._check_type(obs)
        return self.encode(obs)
            

if __name__ == '__main__':
    obs = torch.randint(0, 10, (20,))
    encoder = OneHotPreprocessor(Discrete(10))
    print(encoder(obs))