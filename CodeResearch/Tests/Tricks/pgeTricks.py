import numpy as np
import pygeos

# Создаём массив точек
points = pygeos.points(np.random.random(10000), np.random.random(10000))

# Создаём пространственный индекс
idx = pygeos.STRtree(points)

# Прямоугольник поиска
query_rect = pygeos.box(0.2, 0.2, 0.8, 0.8)

# Поиск точек
results = idx.query(query_rect)
print("Number of points in rectangle:", len(results))
