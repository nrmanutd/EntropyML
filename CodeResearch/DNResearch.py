import numpy as np
import math
from ucimlrepo import fetch_ucirepo

from CodeResearch.calculateAndVisualizeEmpiricalDistribution import calculateAndVisualizeEmpiricalDistribution

#showRandom(150, 2)
#showCircles(150, 2)
#showCircles(1500, 2)
#showCircles(154, 4)
#showCircles(1504, 4)
#showTaskFromUciById(53) #iris
#showTaskFromUciById(186) #wine quality
#showTaskFromUciById(17) #breast cancer wisconsin
#showTaskFromUciById(602) #dry bean
#showTaskFromUciById(54) #isolet


def empiricalDistributionById(id, t):
    set = fetch_ucirepo(id=id)

    dataSet = np.array(set.data.features)
    target = np.array(set.data.targets)

    calculateAndVisualizeEmpiricalDistribution(dataSet, target, set.metadata.name, t=t)

    return

#empiricalDistributionById(53) #iris
#empiricalDistributionById(186) #wine

def checkCircles(l, c):

    elements = l
    nClasses = c

    dataSet = np.zeros((elements, 2))
    target = np.zeros(elements)
    bucket = math.floor(elements / nClasses)
    currentClass = 0

    for i in np.arange(elements):
        if math.floor(i / bucket) > currentClass:
            currentClass += 1

        radius = (10 - currentClass * (10 - 5) / nClasses)

        value = np.random.uniform(0, 1)
        dataSet[i, 0] = radius * math.cos(value * 2 * math.pi)
        dataSet[i, 1] = radius * math.sin(value * 2 * math.pi)

        target[i] = currentClass

    calculateAndVisualizeEmpiricalDistribution(dataSet, target, 'circles')


def checkRandom(l, f):
    elements = l
    features = f

    dataSet = np.zeros((l, features))
    target = np.zeros(elements, dtype=int)

    for i in np.arange(elements):
        for f in np.arange(features):
            dataSet[i, f] = np.random.uniform(-10, 10)

        target[i] = 1 if np.random.uniform(-1, 1) > 0 else -1

    calculateAndVisualizeEmpiricalDistribution(dataSet, target, 'random')


def checkHyperPlane(l):
    elements = l

    dataSet = np.zeros((l, 2))
    target = np.zeros(elements, dtype=int)

    for i in np.arange(elements):
        x = np.random.uniform(-10, 10)
        y = np.random.uniform(-10, 10)
        z = np.random.uniform(-10, 10)

        target[i] = 1 if x + y + z > 0 else -1

    calculateAndVisualizeEmpiricalDistribution(dataSet, target, 'hyperPlane')


def checkTask(task, *args, **kwargs):
    if task == 'circles':
        checkCircles(kwargs.get('l', None), kwargs.get('c', None))
    elif task == 'random':
        checkRandom(kwargs.get('l', None), kwargs.get('f', None))
    elif task == 'hyperPlane':
        checkHyperPlane(kwargs.get('l', None))
    else:
        empiricalDistributionById(task, t=kwargs.get('t', None))

lObj = 1000

#checkTask('circles', l=100, c=2)
#checkTask('circles', l=100, c=4)
#checkTask('random', l=100, f=2)
#checkTask('random', l=100, f=4)
#checkTask('hyperPlane', l=100)

#checkTask('circles', l=lObj, c=2)
#checkTask('circles', l=lObj, c=4)
#checkTask('random', l=lObj, f=2)
#checkTask('random', l=lObj, f=4)
#checkTask('hyperPlane', l=lObj)

#checkTask(53, t=20)
checkTask(186, t=5)

#todo
#+1. вывод в файл графиков (2 картинки)
#2. сохранение в файл данных (посмотреть на датафрейм). Посмотреть, как можно append делать в файл.
#+3. сделать симуляцию на модельных экспериментах двух типов: простой с разделяющей гиперплоскостью с хорошей разделимостью и плохой. Идея - посмотреть на то, как меняется функционал по оценке эмпирической функции распределения, качество алгоритма от степени перемешивания объектов
#+4. оптимизировать расчет радемахеровской сложности через сортировку данных
#5. добавить другие задачи
#+6. сделать рандомизацию по разным подмножествам для оценки общей радемахеровской сложности
#7. Добавить еще несколько моделей (svm, логит, elastic net)

#гипотеза:
# 1. лосс прогнозируется как следствие флуктуаций функции распределения
# 2. accuracy прогнозируется флуктуацией функцией распределения
# 3. чтобы оценить обобщающую способность, не нужно прогонять тысячи итераций обучения, нужно посмотреть насколько переменна эмпирическая функция распределения