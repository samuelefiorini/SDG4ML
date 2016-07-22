#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test SDG4ML strategies."""

import matplotlib.pyplot as plt
from SDG4ML.core.wrappers import generate_data


def main():
    """Run strategies."""
    # problems
    kwargs = {'n': 100, 'd': 300, 'k': 15, 'rho': 0.5,
              'amplitude': 3.5, 'normalized': False, 'seed': 42}
    strategies = ['null', 'sparse', 'correlated',
                  'block_correlated', 'zou_hastie_2005d',
                  'multivariate_groups']

    # Test all strategies
    for s in strategies:
        x, y, beta = generate_data(strategy=s, **kwargs)

        print("\n************************************************************")
        print("Strategy: {}".format(s))
        print("data:\n{}".format(x.shape))
        print("labels:\n{}".format(y.shape))
        print("beta:\n{}".format(beta.shape))

        plt.figure()
        plt.imshow(x.T.dot(x))
        plt.colorbar()
        plt.title('strategy: '+str(s))
        plt.savefig("{}.png".format(s))

if __name__ == '__main__':
    main()
