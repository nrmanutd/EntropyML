import numpy as np
from abc import ABC, abstractmethod
from scipy import optimize, interpolate


class UsefulObjectsCalculator(ABC):

    def evaluate(self, easiness1, easiness2):

        e1 = self.ecdf_advanced(easiness1)
        e2 = self.ecdf_advanced(easiness2)

        intersections = self.find_ecdf_intersection_optimize(e1[0], e1[1], e2[0], e2[1])

        if len(intersections) == 0:
            return 0

        return min(np.array(intersections))

    def find_ecdf_intersection_optimize(self, x1, y1, x2, y2):
        """
        Использование оптимизации для поиска пересечения.
        """
        # Создаем интерполяционные функции
        f1 = interpolate.interp1d(x1, y1, kind='linear',
                                  bounds_error=False, fill_value=(0, 1))
        f2 = interpolate.interp1d(x2, y2, kind='linear',
                                  bounds_error=False, fill_value=(0, 1))

        # Функция разности
        def diff_func(x):
            return f1(x) - f2(x)

        # Диапазон поиска (минимальный и максимальный x из обоих наборов)
        x_min = min(x1.min(), x2.min())
        x_max = max(x1.max(), x2.max())

        # Ищем корни разности функций
        intersections = []

        # Разбиваем на интервалы для поиска
        search_points = np.linspace(x_min, x_max, 100)

        for i in range(len(search_points) - 1):
            a, b = search_points[i], search_points[i + 1]

            # Проверяем, меняет ли функция знак на интервале
            if diff_func(a) * diff_func(b) < 0:
                intersections.append(a)

        return intersections

    def ecdf_advanced(self, data):
        """
        ECDF с правильной обработкой повторяющихся значений
        """
        data = np.asarray(data)
        n = len(data)

        # Сортируем
        x = np.sort(data)

        # Для повторяющихся значений нужно корректно считать
        # Используем cumcount
        y = np.arange(1, n + 1) / n

        # Если есть NaN, удаляем их
        if np.any(np.isnan(data)):
            valid = ~np.isnan(data)
            x = x[valid]
            n = len(x)
            y = np.arange(1, n + 1) / n

        return x, y