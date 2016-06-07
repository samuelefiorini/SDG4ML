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

def null(n=100, d=150, normalized=False, seed=None, **kwargs):
    """
    Generate a signal-less {X,Y} dataset with data and labels completely uncorrelated.

    Parameters
    ----------
    n : int, optional (default is `100`)
        number of samples
    d : int, optional (default is `150`)
        number of dimensions
    normalized : bool, optional (default is `False`)
        if normalized is true than the data matrix is normalized as data/sqrt(n)
    seed : float, optional (default is `None`)
        random seed initialization

    Returns
    -------
    X : (n, d) ndarray
        data matrix
    Y : (n, 1) ndarray
        label vector
    """
    if seed is not None:
        state0 = np.random.get_state()
        np.random.seed(seed)

    if normalized:
        factor = np.sqrt(n)
    else:
        factor = 1

    X = np.random.randn(n,d)/factor # Generate random data
    Y = -1 + 2*np.random.randn(n,1) # Generate random labels

    if seed is not None: # restore random seed
        np.random.set_state(state0)

    return X, Y

def sparse(n=100, d=150, k=15, amplitude=3.5, sigma=0.5, normalized=False, seed=None, **kwargs):
    """
    Generate a sparse linear regression (X,Y) problem. The relationship between input and output is given by:

                                            Y = X*beta + noise

    where X ~ N(0,1), beta is sparse and the nonzero values are in {+-amplitude}, noise ~ N(0,sigma).

    Parameters
    ----------
    n : int, optional (default is `100`)
        number of samples
    d : int, optional (default is `150`)
        total number of dimensions
    k : int, optional (default is `15`)
        number of relevant dimensions
    amplitude : float,  optional (default is `3.5`)
        amplitude of the generative linear model
    sigma : float, optional (default is `0.5`)
        Gaussian noise std
    normalized : bool, optional (default is `False`)
        if normalized is true than the data matrix is normalized as data/sqrt(n)
    seed : float, optional (default is `None`)
        random seed initialization

    Returns
    -------
    X : (n, d) ndarray
        data matrix
    Y : (n, 1) ndarray
        label vector
    beta : (d, 1) ndarray
        real beta vector
    """
    if seed is not None:
        state0 = np.random.get_state()
        np.random.seed(seed)

    if normalized:
        factor = np.sqrt(n)
    else:
        factor = 1

    X = np.random.randn(n,d)/factor            # generate random data

    beta = np.zeros(d)                         # init beta vector
    S0 = np.random.choice(d, k, replace=False) # extract k indexes from d
    beta[S0] = amplitude                       # set them to amplitude
    beta *= np.sign(np.random.randn(d))        # with random sign

    Y = X.dot(beta) + sigma * np.random.randn(n) # evaluate labels

    if seed is not None: # restore random seed
        np.random.set_state(state0)

    return X, Y, beta


def correlated(n=100, d=150, k=15, rho=0.5, amplitude=3.5, sigma=0.5, normalized=False, seed=None, **kwargs):
    """
    Generate a linear regression (X,Y) problem. The relationship between input and output is given by:

                                            Y = X*beta + noise

    where X ~ multi variate Gaussian (0, THETA), THETA is a matrix that has THETA_jk = rhp^abs(j-k), beta is sparse and the nonzero values are in {+-amplitude}, noise ~ N(0,sigma).

    Parameters
    ----------
    n : int, optional (default is `100`)
        number of samples
    d : int, optional (default is `150`)
        total number of dimensions
    k : int, optional (default is `15`)
        number of relevant dimensions
    rho : float, optional (default is `0.5`)
        correlation level
    amplitude : float,  optional (default is `3.5`)
        amplitude of the generative linear model
    sigma : float, optional (default is `0.5`)
        Gaussian noise std
    normalized : bool, optional (default is `False`)
        if normalized is true than the data matrix is normalized as data/sqrt(n)
    seed : float, optional (default is `None`)
        random seed initialization

    Returns
    -------
    X : (n, d) ndarray
        data matrix
    Y : (n, 1) ndarray
        label vector
    beta : (d, 1) ndarray
        real beta vector
    """
    if seed is not None:
        state0 = np.random.get_state()
        np.random.seed(seed)

    if normalized:
        factor = np.sqrt(n)
    else:
        factor = 1

    # Create covariance matrix
    THETA = np.zeros((d,d))
    for i in range(d):
        for j in range(d):
            THETA[i,j] = rho**np.abs(i-j)

    X = np.random.multivariate_normal(mean=np.zeros(d), cov=THETA, size=(n))/factor

    beta = np.zeros(d)                         # init beta vector
    S0 = np.random.choice(d, k, replace=False) # extract k indexes from d
    beta[S0] = amplitude                       # set them to amplitude
    beta *= np.sign(np.random.randn(d))        # with random sign

    Y = X.dot(beta) + sigma * np.random.randn(n) # evaluate labels

    if seed is not None: # restore random seed
        np.random.set_state(state0)

    return X, Y, beta


def block_correlated():
    pass

def zou_hastie_2005d():
    pass
