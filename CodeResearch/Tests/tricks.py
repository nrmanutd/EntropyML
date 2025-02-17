import numpy as np

from CodeResearch.DiviserCalculation.getDiviserFast import getMaximumDiviserFast
from CodeResearch.DiviserCalculation.getDiviserRTree import getMaximumDiviserRTree
from CodeResearch.DiviserCalculation.getDiviserRTreeStochastic import getMaximumDiviserRTreeStochastic

temp = np.array([1, 3, 2, 8, 5])
print(np.argsort(temp))
print(np.argsort(-temp))
pass

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

s4 = np.array([
[5.,3.4,1.6,0.4]
,[5.2,3.5,1.5,0.2]
,[4.7,3.2,1.6,0.2]
,[5.4,3.4,1.5,0.4]
,[5.5,4.2,1.4,0.2]
,[4.9,3.1,1.5,0.1]
,[5.5,3.5,1.3,0.2]
,[4.9,3.1,1.5,0.1]
,[5.1,3.4,1.5,0.2]
,[5.,3.5,1.3,0.3]
,[4.5,2.3,1.3,0.3]
,[5.,3.5,1.6,0.6]
,[5.1,3.8,1.9,0.4]
,[5.1,3.8,1.6,0.2]
,[4.6,3.2,1.4,0.2]
,[5.3,3.7,1.5,0.2]
,[5.1,3.5,1.4,0.2]
,[4.9,3.,1.4,0.2]
,[5.,3.6,1.4,0.2]
,[5.4,3.9,1.7,0.4]
,[4.6,3.4,1.4,0.3]
,[5.,3.4,1.5,0.2]
,[4.4,2.9,1.4,0.2]
,[4.9,3.1,1.5,0.1]
,[4.8,3.4,1.6,0.2]
,[4.8,3.,1.4,0.1]
,[4.3,3.,1.1,0.1]
,[5.7,3.8,1.7,0.3]
,[5.1,3.7,1.5,0.4]
,[4.6,3.6,1.,0.2]
,[5.1,3.3,1.7,0.5]
,[4.8,3.4,1.9,0.2]])
c4 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

s5 = np.array([[5. ,3.4,1.6,0.4]
,[5.2,3.5,1.5,0.2]
,[5.2,3.4,1.4,0.2]
,[4.7,3.2,1.6,0.2]
,[5.5,4.2,1.4,0.2]
,[5.,3.2,1.2,0.2]
,[5.5,3.5,1.3,0.2]
,[4.9,3.1,1.5,0.1]
,[5.,3.5,1.6,0.6]
,[5.1,3.8,1.6,0.2]
,[4.6,3.2,1.4,0.2]
,[5.3,3.7,1.5,0.2]
,[4.9,3.,1.4,0.2]
,[4.7,3.2,1.3,0.2]
,[5.,3.6,1.4,0.2]
,[4.6,3.4,1.4,0.3]
,[5.,3.4,1.5,0.2]
,[4.9,3.1,1.5,0.1]
,[4.8,3.4,1.6,0.2]
,[4.8,3.,1.4,0.1]
,[4.3,3.,1.1,0.1]
,[5.7,4.4,1.5,0.4]
,[5.1,3.5,1.4,0.3]
,[5.1,3.7,1.5,0.4]])
c5 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

def showConcrete(diviser, values, name):
    print('{:} diviser...'.format(name))
    print(diviser)
    print(values)
    pass

def showResults(s, c):

    #diviser, values = getMaximumDiviserProd(s, c)
    #showConcrete(diviser, values, 'Prod')

    #diviser, values = getMaximumDiviser(s, c)
    #showConcrete(diviser, values, 'Stable')

    diviser, values = getMaximumDiviserFast(s, c)
    showConcrete(diviser, values, 'Fast')

    #diviser, values = getMaximumDiviserCorrect(s, c)
    #showConcrete(diviser, values, 'Correct')

    diviser, values = getMaximumDiviserRTree(s, c)
    showConcrete(diviser, values, 'RT')

    diviser, values = getMaximumDiviserRTreeStochastic(s, c)
    showConcrete(diviser, values, 'Stochastic')
    pass

showResults(s, c)
showResults(s1, c1)
showResults(s2, c2)
showResults(s3, c3)
showResults(s4, c4)
showResults(s5, c5)