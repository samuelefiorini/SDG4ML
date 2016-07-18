#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate synthetic dataset for machine learning using SDG4ML."""

import os
import argparse
import cPickle as pkl
import numpy as np
import pandas as pd

from SDG4ML import __version__
from SDG4ML.core.wrappers import generate_data


def main(args):
    """The main routine."""
    # Parse arguments
    kwargs = {'n': args.n, 'd': args.d, 'k': args.k, 'rho': args.rho,
              'n_classes': args.n_classes, 'amplitude': args.amplitude,
              'normalized': args.normalized, 'seed': args.seed}

    print("* Strategy of choice: {}".format(args.strategy))
    print("\t- samples {} - dimensions {}".format(args.n, args.d))
    print("\t- relevant dimensions {}".format(args.k))

    # Generate data
    x, y, beta = generate_data(strategy=args.strategy, **kwargs)
    indcol = {'columns': ['feat_'+str(i) for i in range(args.d)],
              'index': ['sample_'+str(i) for i in range(args.n)]}

    # Save results
    base_folder = args.strategy + "_" + str(args.n) + "_x_" + str(args.d)
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)

    if "csv" not in args.output:
        # save 2 npy + 1 pkl for columns and index
        np.save(os.path.join(base_folder, "data.npy"), x)  # save data
        np.save(os.path.join(base_folder, "labels.npy"), y)  # save labels
        with open(os.path.join(base_folder, "indcols.pkl"), "w") as f:
            pkl.dump(indcol, f)
    else:
        # save 2 csv
        data = pd.DataFrame(data=x, index=indcol['index'],
                            columns=indcol['columns'])
        labels = pd.DataFrame(data=y, index=indcol['index'],
                              columns=['Class'])
        data.to_csv(os.path.join(base_folder, "data.csv"))
        labels.to_csv(os.path.join(base_folder, "labels.csv"))
    print("\t- {} output files generated in {}".format(args.output,
                                                       base_folder))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Synthetic Data Generator'
                                                 'For Machine Lerning.')
    parser.add_argument('--version', action='version',
                        version='%(prog)s v' + __version__)
    parser.add_argument('--strategy', dest="strategy",
                        action='store', default='sparse',
                        help="synthetic data generation strategy of choice. "
                             "Accepted strategies are: null, "
                             "sparse [DEFAULT], correlated, "
                             "block_correlated, zou_hastie_2005d"
                             " or multivariate_groups.")

    parser.add_argument("-n", dest="n", action="store",
                        help="number of samples (default = 100)",
                        default=100)
    parser.add_argument("-d", dest="d", action="store",
                        help="total number of dimensions (default = 150)",
                        default=150)
    parser.add_argument("-k", dest="k", action="store",
                        help="number of relevant dimensions (default = 15)",
                        default=15)
    parser.add_argument("-rho", dest="rho", action="store",
                        help="correlation level (default = 0.5)",
                        default=0.5)
    parser.add_argument("-n_classes", dest="n_classes", action="store",
                        help="number of classes (default = 2), valid only for"
                        "multivariate_groups strategy",
                        default=2)
    parser.add_argument("-amplitude", dest="amplitude", action="store",
                        help="amplitude of the generative linear model"
                        "(default = 3.5)",
                        default=3.5)
    parser.add_argument("-normalized", dest="normalized", action="store_false",
                        help="normalized data as data/sqrt(n)")
    parser.add_argument("-seed", dest="seed", action="store",
                        help="random number generation (default = None)",
                        default=None)
    parser.add_argument("-o", dest="output", action="store",
                        help="output file, either csv (default), or pkl+npy",
                        default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = str(args.strategy + "csv")

    main(args)
