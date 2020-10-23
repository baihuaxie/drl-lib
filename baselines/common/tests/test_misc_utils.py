"""
    test some utility functions
"""

import pytest

import numpy as np
import torch

from baselines.common.misc_util import numpy_to_torch

@pytest.mark.parametrize("x, _type, shape, dtype",[
    (np.ndarray((3, 4), dtype=np.float32), torch.Tensor,\
        torch.Size([3, 4]), torch.float32),
    (np.random.randn(3), torch.Tensor, torch.Size([3]), torch.float64),
    # array of booleans to BoolTensor
    (np.ones((2, 3), dtype=bool), torch.Tensor, \
        torch.Size((2, 3)), torch.bool),
    # single scalar to 0D tensor
    (3, torch.Tensor, torch.Size([]), torch.int64),
    # ndarray of a single scalar to 1D tensor of a single scalar
    (np.ndarray((1,), dtype=np.float64), torch.Tensor, \
        torch.Size([1]), torch.float64)
])
def test_numpy_to_torch(x, _type, shape, dtype):
    """
    """
    y = numpy_to_torch(x)
    # check result type
    assert (type(y) == _type)
    # check shape
    assert (y.shape == shape)
    # check dtype
    assert (y.dtype == dtype)

   