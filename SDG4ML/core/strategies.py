#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Strategies for synthetic regression problems generation.

    IMPLEMTENTED STRATEGIES:

    - NULL: regression problem with no signal, random labels uncorrelated with
            data.
    - SPARSE: sparse linear regression problem
              y = univ_random_X*sparse_beta + noise.
    - CORRELATED: linearly correlated regression problem
              y = multiv_random_X*sparse_beta + noise.
    - BLOCK CORRELATED: linearly block correlated regression problem
              y = multiv_random_X*block_sparse_beta + noise.
    - ZOU HASTIE 2005d: regression problem as in Zou Hastie 2005 (d) scenario.
    - MULTIVARIATE GROUPS: classification problem, each point is sampled from
                           a multivariate normal distribution.

    REF:

    Zou and Hastie (2005), Regularization and variable selection via the
    elastic net.
"""
from __future__ import division
import numpy as np


def null(n=100, d=150, normalized=False, seed=None, **kwargs):
    """Generate a signal-less {X,Y} dataset with data and labels uncorrelated.

    Parameters
    ----------
    n : int, optional (default is `100`)
        number of samples
    d : int, optional (default is `150`)
        number of dimensions
    normalized : bool, optional (default is `False`)
        if normalized is true than the data matrix is normalized as
        data/sqrt(n)
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

    X = np.random.randn(n, d)/factor  # Generate random data
    Y = -1 + 2*np.random.randn(n, 1)  # Generate random labels

    if seed is not None:  # restore random seed
        np.random.set_state(state0)

    return X, Y, np.zeros(d)


def sparse(n=100, d=150, k=15, amplitude=3.5, sigma=0.5, normalized=False,
           seed=None, **kwargs):
    """Generate a sparse linear regression (X,Y) problem.

    The relationship between input and output is given by:

                                            Y = X*beta + noise

    where X ~ N(0,1), beta is sparse and the nonzero values are in
    {+-amplitude}, noise ~ N(0,sigma).

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
        if normalized is true than the data matrix is normalized as
        data/sqrt(n)
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

    X = np.random.randn(n, d)/factor            # generate random data

    beta = np.zeros(d)                           # init beta vector
    S0 = np.random.choice(d, k, replace=False)   # extract k indexes from d
    beta[S0] = amplitude                         # set them to amplitude
    beta *= np.sign(np.random.randn(d))          # with random sign

    Y = X.dot(beta) + sigma * np.random.randn(n)  # evaluate labels

    if seed is not None:  # restore random seed
        np.random.set_state(state0)

    return X, Y, beta


def correlated(n=100, d=150, k=15, rho=0.5, amplitude=3.5, sigma=0.5,
               normalized=False, seed=None, **kwargs):
    """Generate a linear regression (X,Y) problem.

    The relationship between input and output is given by:

                                            Y = X*beta + noise

    where X ~ multi variate random Gaussian (0, THETA), THETA is a (d,d)
    matrix that has THETA[j,k] = rho^abs(j-k), beta is sparse and the nonzero
    values are in {+-amplitude}, noise ~ N(0,sigma).

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
        if normalized is true than the data matrix is normalized as
        data/sqrt(n)
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
    THETA = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            THETA[i, j] = rho**np.abs(i-j)

    X = np.random.multivariate_normal(mean=np.zeros(d), cov=THETA,
                                      size=(n))/factor

    beta = np.zeros(d)                          # init beta vector
    S0 = np.random.choice(d, k, replace=False)  # extract k indexes from d
    beta[S0] = amplitude                        # set them to amplitude
    beta *= np.sign(np.random.randn(d))         # with random sign

    Y = X.dot(beta) + sigma * np.random.randn(n)  # evaluate labels

    if seed is not None:  # restore random seed
        np.random.set_state(state0)

    return X, Y, beta


def block_correlated(n=100, d=150, k=15, rho=0.5, amplitude=3.5, sigma=0.5,
                     normalized=False, seed=None, **kwargs):
    """Generate a linear regression (X,Y) problem.

    The relationship between input and output is given by:

                                            Y = X*beta + noise

    where X is made by stacking on the rows a multi variate random Gaussian
    (0, THETA) of shape (n,k) with a single variate random Gaussian of shape
    (n,d-k), THETA is a (d,d) matrix that has THETA[j,k] = rho^abs(j-k), beta
    is sparse and the nonzero values are in {+-amplitude}, noise ~ N(0,sigma).

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
        if normalized is true than the data matrix is normalized as
        data/sqrt(n)
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
    THETA = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            THETA[i, j] = rho**np.abs(i-j)

    Xleft = np.random.multivariate_normal(mean=np.zeros(k), cov=THETA,
                                          size=(n))
    Xright = np.random.randn(n, d-k)
    X = np.hstack((Xleft, Xright))/factor

    beta = np.zeros(d)                           # init beta vector
    S0 = np.random.choice(d, k, replace=False)   # extract k indexes from d
    beta[S0] = amplitude                         # set them to amplitude
    beta *= np.sign(np.random.randn(d))          # with random sign

    Y = X.dot(beta) + sigma * np.random.randn(n)   # evaluate labels

    if seed is not None:   # restore random seed
        np.random.set_state(state0)

    return X, Y, beta


def zou_hastie_2005d(n=100, d=150, k=15, amplitude=3.5, sigma=0.5,
                     normalized=False, seed=None, **kwargs):
    """Generate a linear regression problem (X,Y) as in Zou Hastie 2005 (d).

    The relationship between input and output is given by:

                                            Y = X*beta + noise

     where beta = (a,...,a, 0,...,0)
                  |------| |-------|
                     k       p - k

     X is a (n,d) matrix made like: X = [G1, G2, G3, G4] where Gj are
     groups of features built as follows:
       - Z1, Z2, Z3 ~ N(0,1)
       - group 1 (G1): (signal) Z1 + (noise) ~ N(0,sigma) -> (n, k/3)
       - group 2 (G2): (signal) Z2 + (noise) ~ N(0,sigma) -> (n, k/3)
       - group 3 (G3): (signal) Z3 + (noise) ~ N(0,sigma) -> (n, k/3)
       - group 4 (G4): ~ N(0,1) -> (n, d-k)

     Parameters
     ----------
     n : int, optional (default is `100`)
         number of samples
     d : int, optional (default is `150`)
         total number of dimensions
     k : int, optional (default is `15`)
         total number of features with coefficient different from zero (k must
         be integer multiple of 3. If it is not, then k is forced to
         k - mod(k,3)).
     amplitude : float,  optional (default is `3.5`)
         amplitude of the generative linear model
     sigma : float, optional (default is `0.5`)
         Gaussian noise std
     normalized : bool, optional (default is `False`)
         if normalized is true than the data matrix is normalized as
         data/sqrt(n)
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

    # check if k is integer multiple of 3
    if k % 3 != 0:
        k -= k % 3

    # Generate groups
    Z = np.random.randn(n, 3)
    G1 = np.array([Z[:, 0], ]*(k//3)).T + 0.001 * np.random.randn(n, k//3)
    G2 = np.array([Z[:, 1], ]*(k//3)).T + 0.001 * np.random.randn(n, k//3)
    G3 = np.array([Z[:, 2], ]*(k//3)).T + 0.001 * np.random.randn(n, k//3)
    G4 = np.random.randn(n, d-k)
    X = np.hstack((G1, G2, G3, G4))/factor

    beta = np.zeros(d)   # init beta vector
    beta[:k] = amplitude
    beta *= np.sign(np.random.randn(d))   # with random sign

    Y = X.dot(beta) + sigma * np.random.randn(n)   # evaluate labels

    if seed is not None:  # restore random seed
        np.random.set_state(state0)

    return X, Y, beta


def wrapper(strategy='sparse', **kwargs):
    """Wrapper that let you generate a regression problem according to several
    strategies (see docstrings for further details).

    Parameters
    ----------
    strategy : str, optional (default is `sparse`)
        data generation strategy, this can be either 'null', 'sparse',
        'correlated', 'block_correlated' or 'zou_hastie_2005d'.
    """
    if strategy.lower() == 'null':
        return null(**kwargs)
    elif strategy.lower() == 'sparse':
        return sparse(**kwargs)
    elif strategy.lower() == 'correlated':
        return correlated(**kwargs)
    elif strategy.lower() == 'block_correlated':
        return block_correlated(**kwargs)
    elif strategy.lower() == 'zou_hastie_2005d':
        return zou_hastie_2005d(**kwargs)
    else:
        print("{} is not a valid strategy. Accepted strategies are: 'null', \
              'sparse', 'correlated', 'block_correlated' or \
              'zou_hastie_2005d'.")
