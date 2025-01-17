import numpy as np

from sortedcontainers import SortedList, SortedSet, SortedDict
from CodeResearch.calculateDistributionDelta import getMaximumDiviser, GetSortedData
from CodeResearch.diviserCalcuation import getMaximumDiviserProd, getMaximumDiviserFast

s = np.array([[1,0,0], [0,1,0], [0,0,1], [3, 2, -1], [2, 3, -1], [2, 2, 0]])
c = np.array([1, 1, 1, -1, -1, -1])

#s = np.array([[1,4], [2,3], [3,2], [4,1]])
#c = np.array([1, 1, -1, -1])

def showResults(diviser, values, name):
    print('{:} diviser...'.format(name))
    print(diviser)
    print(values)
    pass


diviser, values = getMaximumDiviserProd(s, c)
showResults(diviser, values, 'Prod')

diviser, values = getMaximumDiviser(s, c)
showResults(diviser, values, 'Stable')

diviser, values = getMaximumDiviserFast(s, c)
showResults(diviser, values, 'Fast')