import numpy as np
from rtree import index

def uploadRTree(dataSet, target):
    nObjects = dataSet.shape[0]
    nFeatures = dataSet.shape[1]

    p = index.Property()
    p.dimension = nFeatures
    # Создание индекса
    idx = index.Index(properties=p)
    # idx = index.Rtree('rtree')

    idx.properties.dimension = nFeatures

    query = np.zeros(2*nFeatures)
    query[0:nFeatures] = dataSet[0, :]
    query[nFeatures:2*nFeatures] = dataSet[0, :]

    for iObject in range(0, nObjects):
        point = np.zeros(2*nFeatures)
        point[0:nFeatures] = dataSet[iObject, :]
        point[nFeatures:2*nFeatures] = dataSet[iObject, :]
        idx.insert(iObject, point)

        for iFeature in range(0, nFeatures):
            query[iFeature] = min(query[iFeature], dataSet[iObject, iFeature])
            query[iFeature + nFeatures] = max(query[iFeature + nFeatures], dataSet[iObject, iFeature])

    res = list(idx.intersection(query))
    print(res)