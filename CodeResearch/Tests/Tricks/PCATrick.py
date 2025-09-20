import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from CodeResearch.dataSets import load_proteins

x, y = load_proteins("../../Data/Proteins/df_master.csv")

# Стандартизация данных (важно для PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)

# Создание и обучение PCA
pca = PCA(n_components=5)  # уменьшаем до 2 компонент
X_pca = pca.fit_transform(X_scaled)

# Результаты
print("Объясненная дисперсия:", pca.explained_variance_ratio_)
print("Суммарная объясненная дисперсия:", sum(pca.explained_variance_ratio_))

# Визуализация
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.xlabel('Первая главная компонента')
plt.ylabel('Вторая главная компонента')
plt.title('PCA проекция')
plt.colorbar()
plt.show()