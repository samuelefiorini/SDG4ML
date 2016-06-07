#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from SDG4ML.core import strategies as st

def main():

    # problems
    kwargs = {'n': 100, 'd': 300, 'k': 15, 'rho': 0.5,
              'amplitude': 3.5, 'normalized': False, 'seed': 42}
    strategies = ['null', 'sparse', 'correlated', 'block_correlated', 'zou_hastie_2005d']

    # Test all strategies
    for s in strategies:
        x, y, beta = st.wrapper(strategy=s, **kwargs)

        print("\n***************************************************************")
        print("Strategy: {}".format(s))
        print("data:\n{}".format(x.shape))
        print("labels:\n{}".format(y.shape))
        print("beta:\n{}".format(beta.shape))

        plt.imshow(x.T.dot(x))
        plt.colorbar()
        plt.title('strategy: '+str(s))
        plt.show()

if __name__ == '__main__':
    main()
