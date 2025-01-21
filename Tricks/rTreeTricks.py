import numpy as np
from rtree import index
import sys

print(sys.version)

d = dict()
d[1] = '1'
d[2] = '2'
d[3] = '3'

print(next(iter(d.items())))


p = index.Property()
p.dimension = 3
# Создание индекса
idx = index.Index(properties=p)
#idx = index.Rtree('rtree')

# Добавление точек в виде прямоугольников
# Формат: id, (x_min, y_min, x_max, y_max)
points = [
    (0, (1.0, 2.0, 1.0, 1.0, 2.0, 1.0)),  # Точка с id=0
    (1, (2.0, 3.0, 2.0, 2.0, 3.0, 2.0)),  # Точка с id=1
    (2, (4.0, 5.0, 4.0, 4.0, 5.0, 4.0)),  # Точка с id=2
]

for point in points:
    idx.insert(*point)

print(idx.bounds)

query_rect = (1, 2, 1, 4, 5, 4)

# Поиск точек
results = list(idx.intersection(query_rect))
print("Points in range:", results)  # Вывод: [1, 2]

