#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Wrapper strategies module for SDG4ML."""

from SDG4ML.core import strategies as st


def generate_data(strategy='sparse', **kwargs):
    """Wrapper that let you generate a regression problem according to several
    strategies (see docstrings for further details).

    Parameters
    ----------
    strategy : str, optional (default is `sparse`)
        data generation strategy, this can be either 'null', 'sparse',
        'correlated', 'block_correlated', 'zou_hastie_2005d' or 'multitask'.
    """
    if strategy.lower() == 'null':
        return st.null(**kwargs)
    elif strategy.lower() == 'sparse':
        return st.sparse(**kwargs)
    elif strategy.lower() == 'correlated':
        return st.correlated(**kwargs)
    elif strategy.lower() == 'block_correlated':
        return st.block_correlated(**kwargs)
    elif strategy.lower() == 'zou_hastie_2005d':
        return st.zou_hastie_2005d(**kwargs)
    elif strategy.lower() == 'multivariate_groups':
        return st.multivariate_groups(**kwargs)
    elif strategy.lower() == 'multitask':
        return st.multitask(**kwargs)
    else:
        print("{} is not a valid strategy. Accepted strategies are: 'null', \
              'sparse', 'correlated', 'block_correlated', \
              'zou_hastie_2005d', 'multivariate_groups' or 'multitask'.")
