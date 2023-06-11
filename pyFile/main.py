import numpy as np
import pandas as pd
import time
import itertools
import functools
import chcp48

if __name__ == '__main__':
    # chcp48 case
    Vert = [[1, 2, 4, 6], [1, 3, 4, 6], [2, 3, 4, 6], [1, 2, 5, 6], [1, 3, 5, 6], [2, 3, 5, 6],
            [1, 2, 5, 7], [1, 3, 5, 7], [1, 2, 4, 7], [1, 3, 4, 7], [2, 3, 5, 8], [2, 5, 7, 8],
            [3, 5, 7, 8], [2, 3, 4, 8], [2, 4, 7, 8], [3, 4, 7, 8]]
    # input vertex flag directly, and the above is the one of P8_17
    chcp48.run48(Vert)

    # chcp59 case
    num = 322  # input the number of polytope
    flag = 2  # flag=2 means basis yes, flag=1 means basis no
    chcp59.run59(num, flag)



