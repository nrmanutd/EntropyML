from CodeResearch.Visualization.saveDataForVisualization import deserialize_labeles_list_of_arrays

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
