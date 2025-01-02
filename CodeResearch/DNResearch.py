import numpy as np
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


def empiricalDistributionById(id):
    set = fetch_ucirepo(id=id)

    dataSet = np.array(set.data.features)
    target = np.array(set.data.targets)

    calculateAndVisualizeEmpiricalDistribution(dataSet, target, set.metadata.name)

    return

empiricalDistributionById(53) #iris
#empiricalDistributionById(186) #wine

#todo
#1. вывод в файл графиков (2 картинки)
#2. сохранение в файл данных (посмотреть на датафрейм). Посмотреть, как можно append делать в файл.
#3. сделать симуляцию на модельных экспериментах двух типов: простой с разделяющей гиперплоскостью с хорошей разделимостью и плохой. Идея - посмотреть на то, как меняется функционал по оценке эмпирической функции распределения, качество алгоритма от степени перемешивания объектов
#4. оптимизировать расчет радемахеровской сложности через сортировку данных
#5. добавить другие задачи
#6. сделать рандомизацию по разным подмножествам для оценки общей радемахеровской сложности

#гипотеза:
# 1. лосс прогнозируется как следствие флуктуаций функции распределения
# 2. accuracy прогнозируется флуктуацией функцией распределения
# 3. чтобы оценить обобщающую способность, не нужно прогонять тысячи итераций обучения, нужно посмотреть насколько переменна эмпирическая функция распределения