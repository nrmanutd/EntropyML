import os
import re
from CodeResearch.Visualization.saveDataForVisualization import deserialize_labeles_list_of_arrays


def getFileType(file):

    if "KS_permutation" in file:
        return 'permutation'

    if "KS" in file:
        return 'KS'

    if "NN" in file or "ML" in file:
        return 'ML'

    raise ValueError(f'Error: unrecognized file: {file}')


def trimSuffix(taskName):

    if taskName.endswith('_KS') or taskName.endswith('_ML'):
        return taskName[:-3]

    if taskName.endswith('_KS_permutation'):
        return taskName[:-len('_KS_permutation')]

    raise ValueError(f'Unknown suffix: {taskName}')

def loadData(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.txt') and os.path.isfile(os.path.join(directory, f))]

    result = dict()

    for file in files:
        file = os.path.join(directory, file)
        data = deserialize_labeles_list_of_arrays(file)
        type = getFileType(file)

        taskName = data[2]
        taskName = trimSuffix(taskName)

        if taskName not in result:
            result[taskName] = {'ksData': [], 'pData': [], 'mlData': [], 'taskName': taskName}

        curData = result[taskName]

        if type == 'KS':
            curData['ksData'] = data[0]

        if type == 'permutation':
            curData['pData'] = data[0]

        if type == 'ML':
            curData['mlData'] = data[0]

    return result

def getFileName(taskName, type):

    amount = 1000 if type == 'KS' else 100

    if taskName == 'mnist':
        return '..\\Data\\mnist\\27.03_logs\\{:}_mnist_{:}_9_8.txt'.format(type, amount)

    if taskName == 'cifar':
        return '..\\Data\\cifar\\02.04 logs\\{:}_cifar_{:}_9_8_total.txt'.format(type, amount)

    return '..\\Data\\'

def getFileNameSync(taskName, type):
    if taskName == 'mnist' and type == 'KS':
        return '..\\Data\\mnist\\03.04_logs_100_sync\\{:}_mnist_100_9_8.txt'.format(type)

    if taskName == 'mnist':
        return '..\\Data\\mnist\\03.04_logs_100_sync\\{:}_mnist_100_9_8.txt'.format(type)

    if taskName == 'cifar' and type == 'KS':
        return '..\\Data\\cifar\\02.04 logs\\{:}_cifar_1000_9_8_total.txt'.format(type)

    if taskName == 'cifar':
        return '..\\Data\\cifar\\02.04 logs\\{:}_cifar_100_9_8_total.txt'.format(type)

    return '..\\Data\\'

def loadKSData(taskName):
    file = getFileName(taskName, 'KS')

    data = deserialize_labeles_list_of_arrays(file)
    return data

def loadNNData(taskName):
    file = getFileName(taskName, 'NN')

    data = deserialize_labeles_list_of_arrays(file)
    return data

def loadKSsyncData(taskName):
    file = getFileNameSync(taskName, 'KS')

    data = deserialize_labeles_list_of_arrays(file)
    return data

def loadNNsyncData(taskName):
    file = getFileNameSync(taskName, 'NN')

    data = deserialize_labeles_list_of_arrays(file)
    return data

def loadPermutationData(taskName):
    file = getFileName(taskName, 'KS_permutation')

    data = deserialize_labeles_list_of_arrays(file)
    return data
