import math

from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import mnist
import numpy as np

from CodeResearch.DiviserCalculation.getDiviserFastNumba import getMaximumDiviserFastNumba
from CodeResearch.calcModelEstimations import calcModel
from CodeResearch.dataSets import loadMnist
from CodeResearch.pValueCalculator import getDataSetIndexesOfTwoClasses, getDataSetOfTwoClassesCore

def get_index_KS(objectsCount, dataSet, target, level, classesPair, isOver):
    enc = LabelEncoder()
    target = enc.fit_transform(np.ravel(target))

    attempts = 1000
    minSetCountOver = 5
    minSetCountBelow = 5

    curretCounter = 0

    print('Calculating indexes {:} level {:}'.format(('over' if isOver else 'below'), level))

    totalIdx = []

    for i in range(attempts):
        iClassIdx, jClassIdx = getDataSetIndexesOfTwoClasses(objectsCount, target, classesPair[0], classesPair[1])
        idx = np.concatenate((iClassIdx, jClassIdx))

        newSet, newTarget = getDataSetOfTwoClassesCore(dataSet, target, iClassIdx, jClassIdx)
        ks = getMaximumDiviserFastNumba(newSet, newTarget)[0]

        print('Attempt #{:}, KS: {:}'.format(i, ks))

        if isOver and ks > level:
            totalIdx = np.union1d(totalIdx, idx)

            #precision = calcModel(newSet, math.floor(objectsCount * 4 / 5), 10, newTarget)
            #print(precision['accuracy'])
            #print(precision['modelSigma'])

            curretCounter += 1

            if curretCounter >= minSetCountOver:
                return np.array(totalIdx, dtype=np.int32)

        if not isOver and ks < level:
            totalIdx = np.union1d(totalIdx, idx)

            #precision = calcModel(newSet, math.floor(objectsCount * 4 / 5), 10, newTarget)
            #print(precision['accuracy'])
            #print(precision['modelSigma'])

            curretCounter += 1

            if curretCounter >= minSetCountBelow:
                return np.array(totalIdx, dtype=np.int32)

    raise ValueError('Error: couldnt get ks {:} desired level {:}'.format(('over' if isOver else 'below'), level))

def get_objects_in_common(setA, setB):
    return np.intersect1d(setA, setB), np.setdiff1d(setA, setB), np.setdiff1d(setB, setA)

def get_index_KS_over(objectsCount, dataSet, target, level, classesPair):
    return get_index_KS(objectsCount, dataSet, target, level, classesPair, True)

def get_index_KS_below(objectsCount, dataSet, target, level, classesPair):
    return get_index_KS(objectsCount, dataSet, target, level, classesPair, False)

def get_differnt_objects(objectsCount, dataSet, target, classesPair):

    idxA = get_index_KS_over(objectsCount, dataSet, target,0.85, classesPair)
    #idxB = get_index_KS_below(objectsCount, dataSet, target, 0.75, classesPair)
    idxB = get_index_KS_below(objectsCount, dataSet, target, 0.73, classesPair)

    return get_objects_in_common(idxA, idxB)

def show_objects(intersectionIdx, onlyAIdx, onlyBIdx, dataSet, labels, taskName):
    rows = 5
    columns = 10
    numberOfObjects = min(rows * columns, max(len(intersectionIdx), len(onlyAIdx), len(onlyBIdx)))

    print('Intersection: {:}, A not B: {:}, B not A: {:}'.format(len(intersectionIdx), len(onlyAIdx), len(onlyBIdx)))

    idxes = [intersectionIdx, onlyAIdx, onlyBIdx]
    row_titles = ['Intersection', 'Only A images', 'Only B images']

    for i in range(len(idxes)):
        fig, axes = plt.subplots(rows, columns, figsize=(20, 20))
        axes[i, 0].set_ylabel(row_titles[i], fontsize=12, rotation=90, labelpad=10)

        curIdx = idxes[i]
        for j in range(min(numberOfObjects, len(curIdx))):

            image = dataSet[curIdx[j]]
            label = labels[curIdx[j]]

            rowIdx = j // columns
            colIdx = j % columns

            axes[rowIdx, colIdx].imshow(image.squeeze(), cmap='gray')
            axes[rowIdx, colIdx].set_title(f"L: {label}")
            axes[rowIdx, colIdx].axis('off')

        plt.savefig(
            '..\\AnalyzingFigures\\logs_{:}_{:}.png'.format(taskName, row_titles[i]), format='png')
        plt.close(fig)

trainX, trainY = loadMnist()
(mnistSet, mnistLabels), (testX, testY) = mnist.load_data()

intersection, onlyA, onlyB = get_differnt_objects(5000, trainX, trainY, [2, 3])
show_objects(intersection, onlyA, onlyB, mnistSet, mnistLabels, 'mnist')
