#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from SDG4ML.core import strategies

def main():
    # x,y = strategies.null(n=5, d=3, normalized=False, seed=42)
    # x, y, beta = strategies.sparse(n=5, d=10, k=3, amplitude=3.5, normalized=True, seed=42)
    # x, y, beta = strategies.correlated(n=300, d=100, k=5, rho=0.85, amplitude=3.5, sigma= 0.5, normalized=True)#, seed=42)
    x, y, beta = strategies.block_correlated(n=300, d=100, k=30, rho=0.95, amplitude=3.5, sigma= 0.5, normalized=True)#, seed=42)

    print("data:\n{}".format(x.shape))
    print("labels:\n{}".format(y.shape))
    print("beta:\n{}".format(beta.shape))

    plt.imshow(x.T.dot(x))
    plt.colorbar()
    plt.show()





if __name__ == '__main__':
    main()
