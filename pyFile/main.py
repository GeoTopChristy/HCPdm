import numpy as np
import pandas as pd
import time
import itertools
import functools
import chcp48
import chcp59
import hcp47
import os

if __name__ == '__main__':
    # chcp48 case
    Vert = [[1, 2, 4, 6], [1, 3, 4, 6], [2, 3, 4, 6], [1, 2, 5, 6], [1, 3, 5, 6], [2, 3, 5, 6],
            [1, 2, 5, 7], [1, 3, 5, 7], [1, 2, 4, 7], [1, 3, 4, 7], [2, 3, 5, 8], [2, 5, 7, 8],
            [3, 5, 7, 8], [2, 3, 4, 8], [2, 4, 7, 8], [3, 4, 7, 8]]
    # input vertex flag directly, and the above is the one of P8_17
    chcp48.run48(Vert)

    # chcp59 case
    # input the number of polytope in the script chcp59.py, the default value is num = 322.
    # input the number of flag,  flag=2 means basis yes, flag=1 means basis no, the default value is flag =2.
    chcp59.py

    # hcp47 case
    # revise the value of "num" in hcp47.py and run this script, "num" is the label of the polytope and the default value 1.
    hcp47.py



