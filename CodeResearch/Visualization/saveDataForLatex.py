import numpy as np


def getBestM(mValues):

    m = max(mValues)
    i = mValues.tolist().index(m)

    ms = ['M1', 'M2', 'M3', 'M4']

    return '{:} ({:})'.format(m, ms[i])


def getKSD(ksValue):
    if ksValue > 0.95:
        return '$\\longrightarrow 1$'

    if ksValue < 0.05:
        return '$\\longrightarrow 0$'

    return '{:.2f}'.format(ksValue)


def convert(data):
    objc = data['total']
    iClass = data['iClass']
    jClass = data['jClass']

    description = getBestM(data['nPoints'])
    KSdescription = getKSD(data['KSl'])

    return '{:}&{:}({:.0f}/{:.0f})&{:}&{:}\\\\\n'.format(data['Classes'], objc, iClass / objc * 100, jClass / objc * 100, description, KSdescription)


def getSortedIdx(data):

    vals = np.zeros(len(data))
    for i in range(len(data)):
        vals[i] = data[i]['KSl']

    idx = np.argsort(vals)
    return idx

def saveDataForTable(data, task, classes):
    fileName = 'PValuesFigures\\latex_{0}_{1}.txt'.format(task, classes)

    idx = getSortedIdx(data)

    with open(fileName, "w") as write:
        for i in range(len(idx)):
            s = convert(data[idx[i]])
            write.write(s)