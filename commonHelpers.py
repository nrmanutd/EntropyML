import numpy as np

def getPermutation(target):
    target = np.random.permutation(target)
    return getClassesIndex(target)

def getClassesIndex(target):
    entries = np.unique(target)
    classes = {}

    cl = []
    for iClass in np.arange(0, len(entries)):
        cl.append(np.where(target == entries[iClass])[0])
        classes[entries[iClass]] = iClass

    return {'classes':classes, 'classesIndexes':cl}