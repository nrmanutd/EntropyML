import math
import os

from sklearn import datasets
import numpy as np
import pandas as pd
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

from ucimlrepo import fetch_ucirepo

from CodeResearch.calculateAndVisualizeEmpiricalDistribution import calculateAndVisualizeEmpiricalDistribution
from CodeResearch.dataSets import loadMnist, loadCifar, loadFashionMnist, make_xor, make_spirals, make_random
from CodeResearch.estimateAndVisualizeEmpiricalDistributionDelta import estimateAndVisualizeEmpiricalDistributionDelta, \
    estimatePValuesForClassesSeparation


def empiricalDistributionById(id, t, m):
    set = fetch_ucirepo(id=id)

    dataSet = np.array(set.data.features)
    target = np.array(set.data.targets)

    if m == 'delta':
        estimateAndVisualizeEmpiricalDistributionDelta(dataSet, target, set.metadata.name, t=t)
    else:
        calculateAndVisualizeEmpiricalDistribution(dataSet, target, set.metadata.name, t=t)

    return

#empiricalDistributionById(53) #iris
#empiricalDistributionById(186) #wine

def checkCircles(l, c, m):

    elements = l
    nClasses = c

    dataSet = np.zeros((elements, 2))
    target = np.zeros(elements)

    transformedDataSet = np.zeros((elements, 2))

    bucket = math.floor(elements / nClasses)
    currentClass = 0

    for i in np.arange(elements):
        if math.floor(i / bucket) > currentClass:
            currentClass += 1

        radius = (10 - currentClass * (10 - 5) / nClasses)

        value = np.random.uniform(0, 1)
        dataSet[i, 0] = radius * math.cos(value * 2 * math.pi)
        dataSet[i, 1] = radius * math.sin(value * 2 * math.pi)

        transformedDataSet[i, 0] = radius
        transformedDataSet[i, 1] = value * 2 * math.pi

        target[i] = currentClass

    if m == 'delta':
        estimateAndVisualizeEmpiricalDistributionDelta(dataSet, target, 'circles')
        #estimateAndVisualizeEmpiricalDistributionDelta(transformedDataSet, target, 'radian_circles')
    else:
        calculateAndVisualizeEmpiricalDistribution(dataSet, target, 'circles')
        #calculateAndVisualizeEmpiricalDistribution(transformedDataSet, target, 'radian_circles')


def checkRandom(l, f, m):
    elements = l
    features = f

    dataSet = np.zeros((l, features))
    target = np.zeros(elements, dtype=int)

    for i in np.arange(elements):
        for f in np.arange(features):
            dataSet[i, f] = np.random.uniform(-10, 10)

        target[i] = 1 if np.random.uniform(-1, 1) > 0 else -1

    #estimatePValuesForClassesSeparation(dataSet, target, 'random')
    #return

    if m == 'delta':
        estimateAndVisualizeEmpiricalDistributionDelta(dataSet, target, 'random')
    else:
        calculateAndVisualizeEmpiricalDistribution(dataSet, target, 'random')


def checkHyperPlane(l, m):
    elements = l

    dataSet = np.zeros((elements, 3))
    target = np.zeros(elements, dtype=int)

    for i in np.arange(elements):
        x = np.random.uniform(-10, 10)
        y = np.random.uniform(-10, 10)
        z = np.random.uniform(-10, 10)

        dataSet[i, 0] = x
        dataSet[i, 1] = y
        dataSet[i, 2] = z

        target[i] = 1 if x + y + z > 0 else -1

    if m == 'delta':
        estimateAndVisualizeEmpiricalDistributionDelta(dataSet, target, 'hyperPlane')
    else:
        calculateAndVisualizeEmpiricalDistribution(dataSet, target, 'hyperPlane')


def checkHyperPlaneWithIntersection(l, alpha, m):
    elements = l - l%2

    dataSet = np.zeros((elements, 3))
    target = np.zeros(elements, dtype=int)
    a = 10
    center = 2 * a * alpha

    for i in np.arange(elements):

        c = 0 if i < elements/2 else center

        x = np.random.uniform(-a + c, c + a)
        y = np.random.uniform(-a, a)
        z = np.random.uniform(-a, a)

        dataSet[i, 0] = x
        dataSet[i, 1] = y
        dataSet[i, 2] = z

        target[i] = 1 if i < elements/2 else -1


    if m == 'delta':
        estimateAndVisualizeEmpiricalDistributionDelta(dataSet, target, 'hyperPlane_intersection_{0}'.format(alpha))
    else:
        calculateAndVisualizeEmpiricalDistribution(dataSet, target, 'hyperPlane_intersection_{0}'.format(alpha))


def checkMnist(t, m):
    trainX, trainY = loadMnist()

    #trainY = to_categorical(trainY, num_classes)  # One-hot encode the labels
    #testY = to_categorical(testY, num_classes)  # One-hot encode the labels

    #estimatePValuesForClassesSeparation(trainX, trainY, 'mnist', t=t)
    #return

    if m == 'delta':
        estimateAndVisualizeEmpiricalDistributionDelta(trainX, trainY, 'mnist', t=t)
    else:
        calculateAndVisualizeEmpiricalDistribution(trainX, trainY, 'mnist', t=t)
    return

def checkFashionMnist(t, m):

    trainX, train_labels = loadFashionMnist()

    if m == 'delta':
        estimateAndVisualizeEmpiricalDistributionDelta(trainX, train_labels, 'fashionmnist', t=t)
    else:
        calculateAndVisualizeEmpiricalDistribution(trainX, train_labels, 'fashionmnist', t=t)
    return

def checkCifar(t, m):
    trainX, trainY = loadCifar()

    if m == 'delta':
        estimateAndVisualizeEmpiricalDistributionDelta(trainX, trainY, 'cifar', t=t)
    else:
        calculateAndVisualizeEmpiricalDistribution(trainX, trainY, 'cifar', t=t)
    pass

def checkBasicTasks(nSamples=2000):
    # Генерация blobs
    X_blobs, y_blobs = datasets.make_blobs(n_samples=nSamples, centers=2, n_features=2, random_state=42)
    estimatePValuesForClassesSeparation(X_blobs, y_blobs, ksAttempts=10000, pAttempts=10000, mlAttempts=100, taskName='blobs')

    # Генерация moons
    X_moons, y_moons = datasets.make_moons(n_samples=nSamples, noise=0.1, random_state=42)
    estimatePValuesForClassesSeparation(X_moons, y_moons,ksAttempts=10000, pAttempts=10000, mlAttempts=100, taskName='moons')

    # Генерация circles
    X_circles, y_circles = datasets.make_circles(n_samples=nSamples, factor=0.5, noise=0.1, random_state=42)
    estimatePValuesForClassesSeparation(X_circles, y_circles,ksAttempts=10000, pAttempts=10000, mlAttempts=100, taskName='circles_sklearn')

    x_xor, y_xor = make_xor(nSamples)
    estimatePValuesForClassesSeparation(x_xor, y_xor, ksAttempts=10000, pAttempts=10000, mlAttempts=100, taskName='xor')

    x_spirals, y_spirals = make_spirals(nSamples)
    estimatePValuesForClassesSeparation(x_spirals, y_spirals, ksAttempts=10000, pAttempts=10000, mlAttempts=100, taskName='spirals')

    x_random, y_random = make_random(nSamples)
    estimatePValuesForClassesSeparation(x_random, y_random, ksAttempts=10000, pAttempts=10000, mlAttempts=100, taskName='random_sklearn')

    for sd in range(1, 10):
        X_blobs, y_blobs = datasets.make_blobs(n_samples=nSamples, centers=2, n_features=2, cluster_std=sd, random_state=42)
        estimatePValuesForClassesSeparation(X_blobs, y_blobs, ksAttempts=10000, pAttempts=10000, mlAttempts=100, taskName=f'blobs_{sd}')

def checkTask(task, *args, **kwargs):
    if task == 'circles':
        checkCircles(kwargs.get('l', None), kwargs.get('c', None), m=kwargs.get('m', None))
    elif task == 'random':
        checkRandom(kwargs.get('l', None), kwargs.get('f', None), m=kwargs.get('m', None))
    elif task == 'hyperPlane':
        checkHyperPlane(kwargs.get('l', None), m=kwargs.get('m', None))
    elif task == 'hyperPlaneI':
        checkHyperPlaneWithIntersection(kwargs.get('l', None), kwargs.get('alpha', None), m=kwargs.get('m', None))
    elif task == 'mnist':
        checkMnist(t=kwargs.get('t', None), m=kwargs.get('m', None))
    elif task == 'fashionmnist':
        checkFashionMnist(t=kwargs.get('t', None), m=kwargs.get('m', None))
    elif task == 'cifar':
        checkCifar(t=kwargs.get('t', None), m=kwargs.get('m', None))
    else:
        empiricalDistributionById(task, t=kwargs.get('t', None), m=kwargs.get('m', None))

lObj = 1000

#checkTask('hyperPlaneI', l=lObj, alpha=0, m='delta')
#checkTask('hyperPlaneI', l=lObj, alpha=0.5, m='delta')
#checkTask('hyperPlaneI', l=lObj, alpha=1, m='delta')
#checkTask('hyperPlaneI', l=lObj, alpha=2, m='delta')
#checkTask('hyperPlaneI', l=lObj, alpha=5, m='delta')

#checkTask('circles', l=lObj, c=2, m='delta')
#checkTask('circles', l=lObj, c=4, m='delta')
#checkTask('random', l=lObj, f=2, m='delta')
#checkTask('random', l=lObj, f=4, m='delta')
#checkTask('hyperPlane', l=lObj, m='delta')

#checkTask(53, m='delta') #iris
#checkTask(17, m='delta') #wisconsin
#checkTask(186, m='delta') #wine
#checkTask(602, m='delta') #dry bean

#checkTask('mnist', t=10, m='delta')
#checkTask('fashionmnist', t=10, m='delta')
#checkTask(54, m='delta') #isolet
#checkTask('cifar', t=10, m='delta')

checkBasicTasks()#blobs, moons, circles, xor, spirals, random

#todo
#1. Разобраться с проблемой с Hyperplane - почему там низкий KS получился
#2. Профилирование fast алгоритма
#3. Сделать вывод одновременно всех пар классов на одном графике.
#4. Убрать вывод логарифмов (или сделать его в отдельную папку)
#5. Запрограммировать permutations и расчет p-value
#6. Убрать логику с удвоенным объектами (необходимую для радемахера) и считать просто для каждого количества объектов. ВОзможно другую функцию использовать
#7. Держать в памяти все расчеты и перезаписывать график на каждую итерацию, чтобы не ждать всю ночь
#8. На график выводить подписи для каждой пары классов (возможно, по месту)

#гипотеза:
# 1. лосс прогнозируется как следствие флуктуаций функции распределения
# 2. accuracy прогнозируется флуктуацией функцией распределения
# 3. чтобы оценить обобщающую способность, не нужно прогонять тысячи итераций обучения, нужно посмотреть насколько переменна эмпирическая функция распределения