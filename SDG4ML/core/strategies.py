#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    This module implements several synthetic strategies for the generation of synthetic regression problems.

    - NULL: regression problem with no signal, random labels uncorrelated with data.
    - SPARSE: sparse linear regression problem y = univ_random_X*sparse_beta + noise.
    - CORRELATED: linearly correlated regression problem y = multiv_random_X*sparse_beta + noise.
    - BLOCK CORRELATED: linearly block correlated regression problem y = multiv_random_X*block_sparse_beta + noise.
    - ZOU HASTIE 2005d: regression problem as in Zou Hastie 2005 (d) scenario.


    REF:
    Zou and Hastie (2005), Regularization and variable selection via the elastic net.
"""
import numpy as np

def null(n=1000, d=5000, normalized=False, seed=None, **kwargs):
    """
    Generate a signal-less (X,Y) dataset with data and labels completely uncorrelated.

    Parameters
    ----------
    n : float, optional (default is `1000`)
        number of samples
    d : float, optional (default is `5000`)
        number of dimensions
    seed : float, optional (default is `None`)
        random seed initialization
    normalized : bool, optional (default is `False`)
        if normalized is true than the data matrix is normalized as data/sqrt(n)

    Returns
    -------
    X : (n, d) ndarray
        data matrix
    Y : (n, 1) ndarray
        label vector
    """
    if not seed is None:
        np.random.seed(seed)

    if normalized:
        factor = np.sqrt(n)
    else:
        factor = 1

    X = np.random.randn(n,d)/factor
    Y = -1 + 2*np.random.randn(n,1)

    return X, Y

def sparse():
    pass

def correlated():
    pass

def block_correlated():
    pass

def zou_hastie_2005d():
    pass
