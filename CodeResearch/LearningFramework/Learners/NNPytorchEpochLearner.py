import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from CodeResearch.LearningFramework.Learners.baseLearner import BaseLearner
from CodeResearch.LearningFramework.NeuralNetwork.PytorchHelpers import NeuralNetwork


class NNEpochLearnerPyTorch(BaseLearner):
    def __init__(self, nClasses: int, learning_rate=1e-3, dense=512):
        super().__init__()
        self.learning_rate = learning_rate
        self.loss_fn = nn.CrossEntropyLoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nClasses = nClasses
        self.dense = dense
        print(f"Using device: {self.device}")

    def train(self, x, y, probs):
        """Обучение модели с нуля"""
        nFeatures = x.shape[1]

        # Создаем модель
        model = self.define_model(nFeatures, self.nClasses).to(self.device)

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
        # Создаем модель
        model = NeuralNetwork(nFeatures, nClasses, self.dense).to(self.device)

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