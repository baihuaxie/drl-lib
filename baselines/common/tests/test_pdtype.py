"""
    test probability distributions
"""

import pytest

import sys
sys.path.append('../../')

import torch

from common.distributions import CategoricalPdType


def test_categorical():
    """
    validate categorical probability distribution
    """
    pdparams_categorical = torch.rand()
    categorical = CategoricalPdType(ncat=10)
