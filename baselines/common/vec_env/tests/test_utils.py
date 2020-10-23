"""
    tests for vectorized environment utilities
"""

import pytest

from gym.spaces import Discrete, Dict, MultiDiscrete, MultiBinary, \
    Box, Tuple
import numpy as np

from baselines.common.vec_env.utils import obs_space_info


@pytest.mark.parametrize("spaces, shapes, dtypes", [
    (Discrete(5), [()], [np.dtype('int64')]),
    (MultiDiscrete([2, 3, 5]), [(3,)], [np.dtype('int64')]),
    (Box(low=0, high=1e10, shape=(5,)), [(5,)], [np.dtype('float32')]),
    (Box(low=0, high=1e10, shape=(5, 6)), [(5, 6)], [np.dtype('float32')]),
    # MultiBinary(size,); can only take one integer
    (MultiBinary(10,), [(10,)], [np.dtype('int8')]),
    (Dict(
        {
            'key1': Discrete(10),
            'key2': Box(low=0, high=1e5, shape=(4,)),
            'key3': MultiDiscrete([5, 4])
        }
    ), [(), (4,), (2,)], [np.dtype('int64'), np.dtype('float32'), np.dtype('int64')]),
    (Tuple(
        [Discrete(5), Discrete(2), Box(low=0, high=1e5, shape=(6,))]
    ), [(), (), (6,)], [np.dtype('int64'), np.dtype('int64'), np.dtype('float32')])
    ])
def test_spaces(spaces, shapes, dtypes):
    """
    check retrieving shape & dtype of each gym.spaces classes
    """
    _keys, _shapes, _dtypes = obs_space_info(spaces)
    assert len(_keys) != 0
    for i, s in enumerate(_shapes.values()):
        assert s == shapes[i]
    for i, d in enumerate(_dtypes.values()):
        assert d == dtypes[i]
