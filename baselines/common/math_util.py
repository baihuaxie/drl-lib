"""
Utilities for common math operations
"""

import torch
import numpy as np

try:
    from mpi4py import MPI
except ImportError:
    MPI = None
    print("Can not import MPI!")


class RunningMeanStd(object):
    """
    Running Mean & Std (works with MPI)
    """
    def __init__(self, shape=(), epsilon=1e-1):
        """
        Constructor

        Args:
            shape: (tuple) a tuple of integers for the shape of input tensor
                   axis=0 is the summary axis
        """
        self._sum = torch.zeros(shape, dtype=torch.float32)
        self._sumsq = torch.zeros(shape, dtype=torch.float32)
        self._count = epsilon
        self._shape = shape[1:]

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



def explained_variance(ypred, y):
    """
    Computes the fraction of variance that ypred explains about y
    """


if __name__ == '__main__':
    (x1, x2, x3) = (np.random.randn(3,2), np.random.randn(4,2), np.random.randn(5,2))
    rms = RunningMeanStd(shape=x1.shape, epsilon=0.0)

    rms.update(x1)
    rms.update(x2)
    print(rms.mean)