import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy import stats

def calculateNormalityTest(x):
    if len(x) < 5000:
        return stats.shapiro(x)

    return stats.normaltest(x)

def calculateNormalityWithMeanTest(x):
    v2, pv2 = stats.ttest_1samp(x, 0)

    if len(x) < 5000:
        v1, pv1 = stats.shapiro(x)
    else:
        v1, pv1 = stats.normaltest(x)

    return min(pv1, pv2)

def perform_pca(X, n_components=10, scale=True):
    """
    Выполняет PCA и возвращает усеченную матрицу с топ-компонентами

    Parameters:
    X - матрица объекты x признаки
    n_components - количество главных компонент для сохранения
    scale -是否需要 масштабировать признаки
    """

    # Сохраняем исходные размеры
    original_shape = X.shape

    # Масштабирование признаков (рекомендуется для PCA)
    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X
        print("Масштабирование отключено")

    # Выполнение PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    return X_pca

def normalizeTarget(target):
    enc = LabelEncoder()
    normalizedTarget = enc.fit_transform(np.ravel(target))

    return normalizedTarget