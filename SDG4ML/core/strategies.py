#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Strategies for synthetic regression problems generation.

IMPLEMETENTED STRATEGIES:

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
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer



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


def sparse(n=100, d=150, k=15, degree = 1, func = 'l', amplitude=3.5, SNR=0.5, normalized=False, seed=None, **kwargs):
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
    SNR : float, optional (default is `0.5`)
        sigma_signal/sigma_noise nb: mean(signal)=0. sigma_n or s gaussiano
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

    X = np.random.randn(n,d)/factor            # generate random data
    
    if (degree == 1 and func == 'l'):
        X_trans = X
    
    elif degree != 1:
        poly = PolynomialFeatures(degree = 2)
        X_trans = poly.fit_transform(X)  

    if func == 'log':
        log = FunctionTrasformer(np.log1p) #log(1+x)
        X_trans = log.fit_transform(log)
    
    if func == 'gaus':
        gaus = FunctionTransformer(np.exp)   #verificare se exp è element wise
        X_trans = gaus.fit_transform(X)

    if func == 'tanh':
        tanh = FunctionTransformer(np.tanh)
        X_trans = log.fit_transform(X)

    beta = np.zeros(d*degree)                         # init beta vector
    S0 = np.random.choice(d*degree, k, replace=False) # extract k indexes from d
    beta[S0] = amplitude                       # set them to amplitude
    beta *= np.sign(np.random.randn(d*degree))        # with random sign

   # variance_signal = (1/n)*np.dot(X.dot(beta).T,X.dot(beta)) #questa dipenderà dalla dipendenza di Y da X
    variance_signal = (1/n)*np.dot(X_trans.dot(beta).T, X_trans.dot(beta))
    sd_noise = np.sqrt(variance_signal/SNR**2)

    Y = X_trans.dot(beta) + sd_noise*np.random.randn(n) # evaluate labels #Y a questo punto dipende dalla funzione

    if seed is not None:  # restore random seed
        np.random.set_state(state0)

    return X, Y, beta


def correlated(n=100, d=150, k=15, rho=0.5, amplitude=3.5, SNR = 10, normalized=False, seed=None, **kwargs):
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
    SNR : float, optional (default is `10`, STILL NOT IMPLEMENTED)
        Signal to noise ratio - signal variance on noise variance - from this we can determine sigma noise
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


def multivariate_groups(n=100, d=150, k=15, rho=0.5, n_classes=2, means=None,
                        cov=None, normalized=False, seed=None, **kwargs):
    """Sample random points from multivariate gaussians distribution.

    Parameters
    ----------
    n : int, optional (default is `100`)
        total number of samples
    d : int, optional (default is `150`)
        total number of dimensions
    k : int, optional (default is `15`)
        number of relevant dimensions
    rho : float, optional (default is `0.5`)
        correlation level
    n_classes : int, optional (default is `2`)
        number of classes in which n are grouped
    means : array (n_classes, k), optional (default is fixed)
        the means of the multivariate gaussian distribution
    cov : array (n_classes, k, k), optional (default is fixed)
        the covariances of the multivariate gaussian distribution
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

    # Define the number of samples for each group
    samples_per_class = [int(n / n_classes) for j in range(n_classes)]
    samples_per_class[-1] += n - sum(samples_per_class)

    # Define default values for means and cov
    if means is None:
        means = list()
        for j in range(n_classes):
            means.append(np.ones(k) * j)

    if cov is None:
        theta_list = list()
        for j in range(n_classes):
            theta = np.zeros((k, k))
            for r in range(k):
                for c in range(k):
                    theta[r, c] = rho**np.abs(r-c)
            theta_list.append(theta)

    # Create the relevant samples (first class, then the others)
    X = np.random.multivariate_normal(mean=means[0],
                                      cov=theta_list[0],
                                      size=(samples_per_class[0],))
    for j in range(1, n_classes):
        xx = np.random.multivariate_normal(mean=means[j],
                                           cov=theta_list[j],
                                           size=(samples_per_class[j]))
        X = np.vstack((X, xx))

    # Add noise
    noise = np.random.randn(n, d - k)
    X = np.hstack((X, noise)) / factor

    # Generate labels
    y = list()
    for i, j in enumerate(samples_per_class):
        y.extend(j * [i])
    y = np.array(y)

    # check if binary classification
    if len(np.unique(y)) == 2:
        y = 2 * np.array(y) - 1

    if seed is not None:  # restore random seed
        np.random.set_state(state0)

    return X, y, None
