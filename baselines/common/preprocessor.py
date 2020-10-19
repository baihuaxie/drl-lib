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
                   except axis=0 (the summary axis)
        Example:
        - if each new tensor is a 1D tensor of dim=10, then shape=10
        - if each new tensor is a 2D tensor of dim=10x20, then shape=(10,20)
        """
        self._shape = shape
        # use a shape of (1, shape) to initialize sum & sumsq
        # b.c. size of axis=0 is insignificant for initialization
        # this enables instantiation by env.observation_space.shape (Box type)
        self._sum = torch.zeros(((1,) + self._shape), dtype=torch.float32)
        self._sumsq = torch.zeros(((1,) + self._shape), dtype=torch.float32)
        self._count = epsilon

        # mean & std tensors omits the summary axis
        self._mean = torch.zeros(self._shape, dtype=torch.float32)
        self._std = torch.ones(self._shape, dtype=torch.float32)

    def _reshape_obs(self, x):
        """
        Reshape input tensor x into acceptable dimension
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if x.dim() == len(self._shape):
            # input tensor is a single data sample
            # add axis=0 as summary axis
            x_shaped = x.reshape((1,)+tuple(x.shape))
            return x_shaped
        elif x.dim() == len(self._shape) + 1:
            # input tensor is a batch of data samples
            # treat axis=0 as summary axis
            return x
        else:
            raise TypeError("Expected input tensor of shape {} or {} but got \
                tensor of shape {}".formate(self._shape, self._shape+1, x.dim()))

    def update(self, x):
        """
        Update running sum and sumsq by new tensor x

        Args:
            x: (tensor) input tensor
        """
        x = self._reshape_obs(x)
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
            obs: (tensor) a 1D tensor of observations (int64) in form = batch x 1

        Returns:
            (bool) if true suggests obs tensor can be one-hot encoded
        """
        if obs.dtype == torch.int64 and len(obs.shape) == 1:
            pass
        else:
            raise TypeError("Observation type error! Expected 1D tensor of torch.int64\
                but got {}D tensor of {}".format(len(obs.shape), obs.dtype))


    def encode(self, obs):
        """
        Encode the observations by one-hot encoding

        Args:
            obs: (tensor) a 1D tensor of observations (int64) in form = batch x 1
        """
        # returns a 2-d tensor of dimension = batch x obs_space_n
        arr = torch.zeros((obs.shape[0], self._obs_space_n), dtype=torch.float32)
        # create a conditional boolean mask for one-hot encoding
        # if obs[i] == j set mask[i][j] = True else False
        mask = torch.BoolTensor(
            [
                [
                    True if i == obs[j] else False for i in range(arr.shape[1])
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
    for (x1, x2, x3) in [
        (np.random.randn(3), np.random.randn(4), np.random.randn(5)),
        (np.random.randn(3,2), np.random.randn(4,2), np.random.randn(5,2))
    ]:
        # x1.shape[0] is the axis=0 for summary
        rms = RunningMeanStd(shape=x1.shape[1:])
        print(x1.shape)

        x = np.concatenate([x1, x2, x3], axis=0)
        ms1 = [
            np.asarray(x.mean(axis=0)),
            np.asarray(x.std(axis=0))
        ]
        print(ms1)

        rms.update(x1)
        rms.update(x2)
