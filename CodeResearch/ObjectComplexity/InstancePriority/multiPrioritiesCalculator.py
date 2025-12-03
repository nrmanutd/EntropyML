import math

import numpy as np
from scipy.special import softmax

from CodeResearch.ObjectComplexity.Hardness.BaseHardnessCalculator import BaseHardnessCalculator
from CodeResearch.ObjectComplexity.InstancePriority.basePriorityCalculator import BasePriorityCalculator


class MultiPrioritiesCalculator(BasePriorityCalculator):
    def __init__(self, hardnessCalculator: BaseHardnessCalculator, alphas, useBasedPriority, useImportance, useHardness,
                 useBoth):
        self.hardnessCalculator = hardnessCalculator
        self.alphas = alphas
        self.useBoth = useBoth
        self.useHardness = useHardness
        self.useImportance = useImportance
        self.useBasedPriority = useBasedPriority

    def calculatePriority(self, dataSet, target):

        hardnessResult = self.hardnessCalculator.calculateHardness(dataSet, target)
        importance = hardnessResult[0]
        easiness = hardnessResult[1]

        importanceIdx = np.argsort(importance)[::-1]
        hardnessIdx = np.argsort(easiness)[::-1]

        resultPriorities = []
        probs = []

        for alpha in self.alphas:
            nTrain = math.ceil(alpha * len(target))

            if self.useBasedPriority:
                resultPriorities.append(range(nTrain))
                probs.append(np.full(nTrain, 1.0 / nTrain))

            if self.useImportance:
                cutIdx = importanceIdx[:nTrain]
                resultPriorities.append(cutIdx)
                probs.append(softmax(importance[cutIdx]))

            if self.useHardness:
                cutIdx = hardnessIdx[:nTrain]
                resultPriorities.append(cutIdx)
                probs.append(softmax(easiness[cutIdx]))

            if self.useBoth:
                curIdx, curProbs = MultiPrioritiesCalculator.assign_weights(importance, easiness)

                idx = curIdx[:nTrain]
                resultPriorities.append(idx)
                probs.append(curProbs[:nTrain])

        return resultPriorities, probs

    @staticmethod
    def assign_weights(importance, easiness):
        """
        Присваивает веса элементам с двумя признаками.

        Порядок:
        1. Элементы с x > 0.5 идут первыми
        2. Элементы с x <= 0.5 идут после
        3. Внутри каждой группы сортировка по убыванию x*y

        Веса: 1, 2, 3, ... N в полученном порядке.

        Parameters:
        -----------
        x_arr, y_arr : array-like
            Массивы значений от 0 до 1 одинаковой длины

        Returns:
        --------
        weights : ndarray
            Массив весов в исходном порядке элементов
        sorted_indices : ndarray
            Индексы элементов в порядке сортировки
        """
        x = np.asarray(importance)
        y = np.asarray(easiness)
        n = len(x)

        # Индексы всех элементов
        indices = np.arange(n)

        # Вычисляем x*y для всех элементов
        xy_product = x * y

        # Маска для группы x > 0.5
        mask_a = x > 0.5

        # Индексы группы A
        idx_a = indices[mask_a]
        # Индексы группы B
        idx_b = indices[~mask_a]

        # Сортируем группу A по убыванию x*y
        sorted_idx_a = idx_a[np.argsort(-xy_product[idx_a])]

        # Сортируем группу B по убыванию x*y
        sorted_idx_b = idx_b[np.argsort(-xy_product[idx_b])]

        # Объединяем: сначала группа A, потом группа B
        idx = np.concatenate([sorted_idx_a, sorted_idx_b])

        # Создаём массив весов в порядке сортировки
        probs = softmax(np.arange(n, 0, -1))

        return idx, probs
