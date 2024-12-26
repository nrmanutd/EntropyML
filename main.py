import math
import pandas as pd
from calcEntropy import calculateAndVisualizeSeveralEntropies
from ucimlrepo import fetch_ucirepo
import numpy as np

from showRademacherDistribution import showRademacherDistirbution


def showCircles(elements, nClasses):
    dataSet = np.zeros((elements, 2))
    target = np.zeros(elements)
    bucket = math.floor(elements / nClasses)
    currentClass = 0;

    transformedDataSet = np.zeros((elements, 2))

    for i in np.arange(elements):
        if math.floor(i / bucket) > currentClass:
            currentClass += 1

        radious = (10 - currentClass * (10 - 5) / nClasses)

        value = np.random.uniform(0, 1)
        dataSet[i, 0] = radious * math.cos(value * 2*math.pi)
        dataSet[i, 1] = radious * math.sin(value * 2*math.pi)

        transformedDataSet[i, 0] = radious
        transformedDataSet[i, 1] = value * 2 * math.pi

        target[i] = currentClass%2

    calculateAndVisualizeSeveralEntropies(dataSet, target, 'circles')
    calculateAndVisualizeSeveralEntropies(transformedDataSet, target, 'circles_radian')

    #df = pd.DataFrame(dataSet, columns=['X1', 'X2', 'X3', 'Y'])
    #parallel_coordinates(df, class_column='Y', colormap=plt.get_cmap("Set2"))

    # Hide the color scale that is useless in this case
    #fig.update_layout(coloraxis_showscale=False)

    #plt.show()

    #calcAndVisualize(dataSet, cl, 10)


def showRandom(elements, features):
    dataSet = np.zeros((elements, features))
    target = np.zeros(elements, dtype=int)

    for i in np.arange(elements):
        for f in np.arange(features):
            dataSet[i, f] = np.random.uniform(-10, 10)

        target[i] = 1 if np.random.uniform(-1, 1) > 0 else -1

    calculateAndVisualizeSeveralEntropies(dataSet, target, 'random')


def showTaskFromUciById(id):
    # fetch dataset
    set = fetch_ucirepo(id=id)

    dataSet = np.array(set.data.features)
    target = np.array(set.data.targets)

    calculateAndVisualizeSeveralEntropies(dataSet, target, set.metadata.name)

#showRandom(150, 2)
#showCircles(150, 2)
#showCircles(1500, 2)
#showCircles(154, 4)
#showCircles(1504, 4)
#showTaskFromUciById(53) #iris
#showTaskFromUciById(186) #wine quality
#showTaskFromUciById(17) #breast cancer wisconsin
#showTaskFromUciById(602) #dry bean
#showTaskFromUciById(54) #isolet


def rademacherTest():

    nDimension = 10
    nVectors = 10
    nTimes = 100

    showRademacherDistirbution(nDimension, nVectors, nTimes)

    return

rademacherTest()