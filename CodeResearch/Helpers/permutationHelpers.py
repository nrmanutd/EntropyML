import math

import numpy as np

def extractDataSet(x, y, nObjects, nFeatures):
    xx, yy = GetSubSet(x, y, math.floor(nObjects / 2))
    xx = GetSubSetOnFeatures(xx, nFeatures)

    return xx, yy

def permuteDataSet(newSet, newTarget):
    idx = range(0, len(newTarget))
    newIdx = np.random.permutation(idx)

    return newSet[newIdx], newTarget

def GetObjectsPerClass(target, seekingClass, nObjects):
    idx = np.nonzero(target == seekingClass)[0]

    mask = np.zeros(len(idx))
    mask[0: min(len(idx) - 1, nObjects)] = 1

    mask = np.random.permutation(mask)
    idxM = np.nonzero(mask)[0]

    return idx[idxM]

def GetSubSet(dataSet, target, nObjects):
    vClasses, parts = np.unique(target, return_counts=True)
    parts = parts / len(target)

    nParts = np.floor(nObjects * parts).astype(int)

    objectsPerClass = 2 * np.maximum(np.ones(len(nParts), dtype=int), nParts)

    subSetIdx = []

    for iClass in np.arange(len(vClasses)):
        idx = GetObjectsPerClass(target, vClasses[iClass], objectsPerClass[iClass])
        subSetIdx = subSetIdx + idx.tolist()

    return dataSet[subSetIdx], target[subSetIdx]


def GetSubSetOnFeatures(x, nFeatures):
    totalFeatures = x.shape[1]
    selectedFeatures = np.random.choice(np.arange(totalFeatures), size=nFeatures, replace=False)
    return x[:, selectedFeatures]


def getDataSetIndexesOfTwoClasses(currentObjects, target, iClass, jClass):
    iClassIdx = np.where(target == iClass)[0]
    jClassIdx = np.where(target == jClass)[0]

    # print('Total objects: {:}, iClass: {:}, jClass: {:}, currentObjects: {:}'.format(dataSet.shape[0], len(iClassIdx), len(jClassIdx), currentObjects))

    partIClass = len(iClassIdx) / (len(iClassIdx) + len(jClassIdx))

    iObjectsCount = math.ceil(partIClass * currentObjects) if partIClass < 0.5 else math.floor(
        partIClass * currentObjects)
    jObjectsCount = currentObjects - iObjectsCount

    iClassObjects = GetObjectsPerClass(target, iClass, iObjectsCount)
    jClassObjects = GetObjectsPerClass(target, jClass, jObjectsCount)

    return iClassObjects, jClassObjects


def getDataSetOfTwoClassesCore(dataSet, target, iClassObjects, jClassObjects):

    iObjectsCount = len(iClassObjects)
    jObjectsCount = len(jClassObjects)

    nFeatures = dataSet.shape[1]
    newSet = np.zeros((iObjectsCount + jObjectsCount, nFeatures))

    newSet[0:iObjectsCount, :] = dataSet[iClassObjects, :]
    newSet[iObjectsCount:(iObjectsCount + jObjectsCount), :] = dataSet[jClassObjects, :]

    newTarget = np.zeros(iObjectsCount + jObjectsCount)
    newTarget[0:iObjectsCount] = target[iClassObjects]
    newTarget[iObjectsCount: (iObjectsCount + jObjectsCount)] = target[jClassObjects]

    return newSet, newTarget


def stratified_split_indices_with_min(
        y: np.ndarray,
        train_ratio: float = 0.8,
        min_samples_per_class: int = 1,
        random_state: int = None
):
    """
    Стратифицированное разбиение с гарантией минимального количества образцов в каждом классе.

    Parameters:
    -----------
    y : np.ndarray
        Вектор меток классов
    train_ratio : float
        Доля объектов для обучения
    min_samples_per_class : int
        Минимальное количество образцов каждого класса в обучающей выборке
    random_state : int
        Seed для воспроизводимости

    Returns:
    --------
    train_indices, test_indices : Tuple[np.ndarray, np.ndarray]
    """
    if not 0 < train_ratio <= 1:
        raise ValueError(f"train_ratio должен быть между 0 и 1, получено {train_ratio}")

    if random_state is not None:
        np.random.seed(random_state)

    unique_classes, class_counts = np.unique(y, return_counts=True)

    train_indices_list = []
    test_indices_list = []

    for cls, count in zip(unique_classes, class_counts):
        class_indices = np.where(y == cls)[0]
        shuffled_indices = np.random.permutation(class_indices)

        # Вычисляем количество для train с учетом минимального требования
        n_train = max(min_samples_per_class, int(np.floor(count * train_ratio)))

        # Если недостаточно данных для разделения
        if n_train >= count:
            n_train = max(1, count - 1)  # оставляем хотя бы один в test

        train_class_indices = shuffled_indices[:n_train]
        test_class_indices = shuffled_indices[n_train:]

        train_indices_list.extend(train_class_indices)
        test_indices_list.extend(test_class_indices)

    train_indices = np.array(train_indices_list)
    test_indices = np.array(test_indices_list)

    return train_indices, test_indices

def stratified_split_indices_with_min_and_priority(
        y: np.ndarray,
        priority: np.ndarray,
        alpha: float = 0.8,
        min_samples_per_class: int = 1,
        random_state: int = None
):
    if not 0 < alpha <= 1:
        raise ValueError(f"train_ratio должен быть между 0 и 1, получено {alpha}")

    if random_state is not None:
        np.random.seed(random_state)

    unique_classes, class_counts = np.unique(y, return_counts=True)

    train_indices_list = []

    for cls, count in zip(unique_classes, class_counts):
        class_indices = np.where(y == cls)[0]
        class_priorities = priority[class_indices]

        class_priorities_idx = np.argsort(-class_priorities)

        shuffled_indices = class_indices[class_priorities_idx]

        n_train = max(min_samples_per_class, int(np.floor(count * alpha)))

        # Если недостаточно данных для разделения
        if n_train >= count:
            n_train = max(1, count - 1)  # оставляем хотя бы один в test

        train_class_indices = shuffled_indices[:n_train]
        train_indices_list.extend(train_class_indices)

    train_indices = np.array(train_indices_list)

    return train_indices
