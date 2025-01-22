import math

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from ucimlrepo import fetch_ucirepo

from CodeResearch.calculateAndVisualizeEmpiricalDistribution import calculateAndVisualizeEmpiricalDistribution
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
        estimateAndVisualizeEmpiricalDistributionDelta(transformedDataSet, target, 'radian_circles')
    else:
        calculateAndVisualizeEmpiricalDistribution(dataSet, target, 'circles')
        calculateAndVisualizeEmpiricalDistribution(transformedDataSet, target, 'radian_circles')


def checkRandom(l, f, m):
    elements = l
    features = f

    dataSet = np.zeros((l, features))
    target = np.zeros(elements, dtype=int)

    for i in np.arange(elements):
        for f in np.arange(features):
            dataSet[i, f] = np.random.uniform(-10, 10)

        target[i] = 1 if np.random.uniform(-1, 1) > 0 else -1

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
    num_train = 60000  # there are 60000 training examples in MNIST
    num_test = 10000  # there are 10000 test examples in MNIST

    height, width, depth = 28, 28, 1  # MNIST images are 28x28 and greyscale
    num_classes = 10  # there are 10 classes (1 per digit)

    # load dataset
    (trainX, trainY), (testX, testY) = mnist.load_data()
    # reshape dataset to have a single channel

    trainX = trainX.reshape(num_train, height * width)  # Flatten data to 1D
    testX = testX.reshape(num_test, height * width)  # Flatten data to 1D
    trainX = trainX.astype('float32')
    testX = testX.astype('float32')
    trainX /= 255  # Normalise data to [0, 1] range
    testX /= 255  # Normalise data to [0, 1] range

    #trainY = to_categorical(trainY, num_classes)  # One-hot encode the labels
    #testY = to_categorical(testY, num_classes)  # One-hot encode the labels

    estimatePValuesForClassesSeparation(trainX, trainY, 'mnist', t=t)
    return

    if m == 'delta':
        estimateAndVisualizeEmpiricalDistributionDelta(trainX, trainY, 'mnist', t=t)
    else:
        calculateAndVisualizeEmpiricalDistribution(trainX, trainY, 'mnist', t=t)
    return

def checkCifar(t, m):
    num_train = 50000  # there are 60000 training examples in CIFAR
    num_test = 10000  # there are 10000 test examples in CIFAR

    height, width, depth = 32, 32, 3  # MNIST images are 32x32 and greyscale
    num_classes = 10  # there are 10 classes (1 per digit)

    # load dataset
    (trainX, trainY), (testX, testY) = cifar10.load_data()
    # reshape dataset to have a single channel

    trainX = trainX.reshape(num_train, height * width )  # Flatten data to 1D
    testX = testX.reshape(num_test, height * width)  # Flatten data to 1D
    trainX = trainX.astype('float32')
    testX = testX.astype('float32')
    trainX /= 255  # Normalise data to [0, 1] range
    testX /= 255  # Normalise data to [0, 1] range

    # trainY = to_categorical(trainY, num_classes)  # One-hot encode the labels
    # testY = to_categorical(testY, num_classes)  # One-hot encode the labels

    if m == 'delta':
        estimateAndVisualizeEmpiricalDistributionDelta(trainX, trainY, 'cifar', t=t)
    else:
        calculateAndVisualizeEmpiricalDistribution(trainX, trainY, 'cifar', t=t)
    pass

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

checkTask('mnist', t=10, m='delta')
#checkTask(54, m='delta') #isolet
#checkTask('cifar', t=10, m='delta')

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