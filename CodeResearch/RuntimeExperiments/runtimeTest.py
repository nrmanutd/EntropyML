import time
import numpy as np

from CodeResearch.DiviserCalculation.getDiviserFastNumba import getMaximumDiviserFastNumba
from CodeResearch.permutationHelpers import extractDataSet


def runtimeTest(x, y, n, k, attempts):

    hotNumber = 5

    times = []

    for i in range(attempts + hotNumber):
        print(f'Attempt # {i}')
        xx, yy = extractDataSet(x, y, n, k)

        t1 = time.time()
        getMaximumDiviserFastNumba(xx, yy)
        t2 = time.time()

        if i < hotNumber:
            continue

        print(f'Attempt #{i} of {attempts}')
        times.append(t2 - t1)

    return np.array(times)