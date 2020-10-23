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




def explained_variance(ypred, y):
    """
    Computes the fraction of variance that ypred explains about y
    """
