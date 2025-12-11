import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from CodeResearch.LearningFramework.Learners.baseLearner import BaseLearner
from CodeResearch.LearningFramework.NeuralNetwork.PytorchHelpers import NeuralNetwork


class NNEpochLearnerPyTorch(BaseLearner):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.learning_rate = learning_rate
        self.loss_fn = nn.CrossEntropyLoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def train(self, x, y, probs):
        """Обучение модели с нуля"""
        nFeatures = x.shape[1]
        nClasses = len(np.unique(y))

        # Создаем модель
        model = self.define_model(nFeatures, nClasses).to(self.device)

        # Обучение
        model = self.update(model, x, y)
        return model

    def update(self, model, x, y):
        """Одно обновление модели (один шаг градиентного спуска)"""
        # Подготовка данных
        x_tensor, y_tensor = self._prepare_data(x, y)

        # Переводим модель в режим обучения
        model.train()

        # Forward pass
        predictions = model(x_tensor)
        loss = self.loss_fn(predictions, y_tensor)

        # Backward pass и оптимизация
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()

        return model

    def test(self, model, x, y):
        """Тестирование модели"""
        # Подготовка данных
        x_tensor, y_tensor = self._prepare_data(x, y)

        # Переводим модель в режим оценки
        model.eval()

        with torch.no_grad():
            predictions = model(x_tensor)
            _, predicted = torch.max(predictions, 1)
            correct = (predicted == y_tensor).sum().item()
            accuracy = correct / y_tensor.size(0)

        return accuracy, predicted

    def define_model(self, nFeatures, nClasses):
        """Определение архитектуры модели"""
        # Динамический выбор размера скрытого слоя
        dense = 512 if nFeatures > 20 else 16

        # Создаем модель
        model = NeuralNetwork(nFeatures, nClasses, dense).to(self.device)

        # Добавляем оптимизатор как атрибут модели
        model.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        return model

    def _prepare_data(self, x, y):
        x_tensor = torch.FloatTensor(x).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)

        return x_tensor, y_tensor

    def trainAndTest(self, x, y, probs, xt, yt):
        model = self.train(x, y, probs)
        return self.test(model, xt, yt)