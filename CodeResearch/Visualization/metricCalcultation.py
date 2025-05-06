import numpy as np

def calculateMetric(ksData, pData):
    ksMedians = np.array([np.average(k) for k in ksData])
    permutationMedians = np.array([np.average(k) for k in pData])

    ksMedians = ksMedians - permutationMedians
    maxDeltas = np.ones(len(ksMedians)) - permutationMedians

    ksMedians = ksMedians / maxDeltas

    #print(1 - ksMedians)

    return 1 - ksMedians