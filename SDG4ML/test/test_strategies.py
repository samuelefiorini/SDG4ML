#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from SDG4ML.core import strategies

def main():
    x,y = strategies.null(n=5, d=3, normalized=False, seed=42)

    print("data : {}".format(x))
    print("labels : {}".format(y))





if __name__ == '__main__':
    main()
