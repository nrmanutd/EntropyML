import math
import random

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from CodeResearch.SaveRademacherResults import SaveRademacherResults
from CodeResearch.calculateDistributionDelta import calculateRademacherComplexity


def calcModel(dataSet, ratio, target):

    enc = LabelEncoder()
    target = enc.fit_transform(np.ravel(target))

    test_size = 0.8 * (1 - ratio) + 0.1 * ratio
    X_train, X_test, y_train, y_test = train_test_split(dataSet, target, test_size=test_size,
                                                        random_state=random.randint(1, 1000))
    model = XGBClassifier().fit(X_train, np.ravel(y_train))
    return accuracy_score(y_test, model.predict(X_test))



def calculateAndVisualizeEmpiricalDistribution(dataSet, target, taskName):

    nAttempts = 20
    modelAttempts = 20
    probability = 0.95
    delta = 1 - probability
    step = 5
    nObjects = dataSet.shape[0]
    nFeatures = dataSet.shape[1]

    print('Starting task: {0}. nAttempts: {1}, modelAttempts: {2}, objects: {3}'.format(taskName, nAttempts, modelAttempts, nObjects))

    maxObjects = min(60, math.floor(nObjects/2/step))
    minObjects = min(1, maxObjects)

    avgDistributions = np.zeros(maxObjects - minObjects, dtype=float)
    avgDistributions2 = np.zeros(maxObjects - minObjects, dtype=float)
    avgDistributions3 = np.zeros(maxObjects - minObjects, dtype=float)
    accuracy = np.zeros(maxObjects - minObjects, dtype=float)
    modelSigma = np.zeros(maxObjects - minObjects, dtype=float)

    sigmas = np.zeros(maxObjects - minObjects, dtype=float)
    mcDiarmids = np.zeros(maxObjects - minObjects, dtype=float)

    objectsIterator = np.arange(minObjects, maxObjects, dtype=int)

    for iDistribution in np.arange(len(objectsIterator), dtype=int):

        currentObjects = objectsIterator[iDistribution] * step
        print('Calculating for {0}'.format(currentObjects))

        mcDiarmids[iDistribution] = math.sqrt(math.log(1/delta) / currentObjects)
        result = calculateRademacherComplexity(dataSet, currentObjects, nAttempts, modelAttempts, target)

        radResult = result['radResult']
        modelResult = result['modelResult']

        avgDistributions[iDistribution] = radResult['rad']
        sigmas[iDistribution] = radResult['sigma']
        avgDistributions2[iDistribution] = radResult['upperRad']
        #avgDistributions3[iDistribution] = radResult['upperRad2']

        accuracy[iDistribution] = modelResult['accuracy']
        modelSigma[iDistribution] = modelResult['modelSigma']
        #accuracy_full[iDistribution] = modelResult['accuracy_full']


    xLabels = np.arange(minObjects, maxObjects) * step

    data = {'xLabels': xLabels, 'rad': avgDistributions,
            'upperRad': avgDistributions2, 'accuracy': accuracy, 'modelSigma': modelSigma,
            'mcDiarmid': mcDiarmids}

    task = {'name': taskName, 'nObjects': nObjects, 'nFeatures': nFeatures, 'nAttempts': nAttempts,
            'modelAttempts': modelAttempts, 'step': step, 'prob': probability}

    SaveRademacherResults(data, task)

    return