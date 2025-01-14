import time

from sklearn.preprocessing import LabelEncoder
import numpy as np

from CodeResearch.calcModelEstimations import calcModel
from CodeResearch.calculateDistributionDelta import calcRademacherForSets, calcRademacherDeltasForSets


def calculateRademacherComplexity(dataSet, nObjects, nAttempts, modelAttempts, nRadSets, target):
    enc = LabelEncoder()
    target = enc.fit_transform(np.ravel(target))

    print('Calculating Rademacher & Model scores...')
    start = time.time()
    radResult = calcRademacherForSets(dataSet, nObjects, nAttempts, nRadSets, target)
    end = time.time()
    print('Calculated Rademacher: {:.2f}s'.format(end - start))

    start = time.time()
    modelResult = calcModel(dataSet, nObjects, modelAttempts, target)
    end = time.time()
    print('Calculated Model: {:.2f}s'.format(end - start))

    return {'radResult': radResult, 'modelResult': modelResult}

def calculateModelAndDistributionDelta(dataSet, nObjects, nAttempts, modelAttempts, nRadSets, target):
    #enc = LabelEncoder()
    #target = enc.fit_transform(np.ravel(target))

    print('Calculating Rademacher & Model scores...')
    start = time.time()
    radResult = calcRademacherDeltasForSets(dataSet, nObjects, nAttempts, nRadSets, target)
    end = time.time()
    print('Calculated Rademacher: {:.2f}s'.format(end - start))

    start = time.time()
    modelResult = calcModel(dataSet, nObjects, modelAttempts, target)
    end = time.time()
    print('Calculated Model: {:.2f}s'.format(end - start))

    return {'radResult': radResult, 'modelResult': modelResult}