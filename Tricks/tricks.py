import numpy as np

from CodeResearch.DiviserCalculation.diviserCalcuation import getMaximumDiviser
from CodeResearch.DiviserCalculation.getCorrectDiviser import getMaximumDiviserCorrect
from CodeResearch.DiviserCalculation.getDiviserFast import getMaximumDiviserFast
from CodeResearch.DiviserCalculation.getDiviserProd import getMaximumDiviserProd
from CodeResearch.DiviserCalculation.getDiviserRTree import getMaximumDiviserRTree
from CodeResearch.DiviserCalculation.statisticsCalculation import getMaximumPossibleByAnalysis

s = np.array([[1,0,0], [0,1,0], [0,0,1], [3, 2, -1], [2, 3, -1], [2, 2, 0]])
c = np.array([1, 1, 1, -1, -1, -1])

s1 = np.array([[1,4], [2,3], [3,2], [4,1]])
c1 = np.array([1, 1, -1, -1])

s2 = np.array([[5.1, 3.4, 1.5, 0.2],
 [4.8, 3. , 1.4, 0.3],
 [4.6, 3.1, 1.5, 0.2],
 [4.9, 3.1, 1.5, 0.1]])
c2 = np.array([-0.5, -0.5,  0.5,  0.5])

s3 = np.array([[5.,  3.,  1.6, 0.2],
 [5.2, 3.4, 1.4, 0.2],
 [5.1, 3.4, 1.5, 0.2],
 [5. , 3.5, 1.3, 0.3],
 [4.4, 3.2, 1.3, 0.2],
 [5. , 3.5, 1.6, 0.6],
 [4.8, 3.,  1.4, 0.3],
 [5. , 3.3, 1.4, 0.2],
 [5.1, 3.5, 1.4, 0.2],
 [4.9, 3.,  1.4, 0.2],
 [4.6, 3.1, 1.5, 0.2],
 [4.4, 2.9, 1.4, 0.2],
 [4.3, 3.,  1.1, 0.1],
 [5.1, 3.5, 1.4, 0.3],
 [5.1, 3.8, 1.5, 0.3],
 [5.1, 3.7, 1.5, 0.4]])
c3 =[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]

def showConcrete(diviser, values, name):
    print('{:} diviser...'.format(name))
    print(diviser)
    print(values)
    pass

def showResults(s, c):

    diviser, values = getMaximumDiviserProd(s, c)
    showConcrete(diviser, values, 'Prod')

    diviser, values = getMaximumDiviser(s, c)
    showConcrete(diviser, values, 'Stable')

    diviser, values = getMaximumDiviserFast(s, c)
    showConcrete(diviser, values, 'Fast')

    #diviser, values = getMaximumDiviserCorrect(s, c)
    #showConcrete(diviser, values, 'Correct')

    diviser, values = getMaximumDiviserRTree(s, c)
    showConcrete(diviser, values, 'RT')
    pass

#showResults(s, c)
#showResults(s1, c1)
showResults(s2, c2)
#showResults(s3, c3)