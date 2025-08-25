from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
