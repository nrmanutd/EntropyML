import math
import numpy as np

def extractData(baseArray, epochs, iterations, repeats):

    baseRepeatSigmas = np.zeros(epochs)
    baseIterationSigmas = np.zeros(epochs)
    baseTotalSigmas = np.zeros(epochs)
    baseMean = np.zeros(epochs)
    baseTargetSigmas = np.zeros(epochs)

    for epoch in range(epochs):
        curEpochArray = np.array(baseArray[0][epoch])

        totalStd = np.std(curEpochArray)
        valuesInsideRepeat = np.zeros(len(curEpochArray))

        for curIteration in range(iterations):
            idx = range(curIteration * repeats, (curIteration + 1) * repeats)
            curValues = curEpochArray[idx]
            curValues = curValues - np.mean(curValues)

            valuesInsideRepeat[idx] = curValues

        repeatStd = np.std(valuesInsideRepeat)
        iterationsStd = iterations / (iterations - 1) * math.sqrt(totalStd ** 2 - (iterations * repeats - 1) / (iterations * repeats) * repeatStd ** 2)

        baseRepeatSigmas[epoch] = repeatStd
        baseTotalSigmas[epoch] = totalStd
        baseIterationSigmas[epoch] = iterationsStd

        baseMean[epoch] = np.mean(curEpochArray)
        baseTargetSigmas[epoch] = math.sqrt(
            iterationsStd ** 2 / iterations + repeatStd ** 2 / (iterations * repeats))

    return baseMean, baseTargetSigmas, baseRepeatSigmas, baseIterationSigmas, baseTotalSigmas