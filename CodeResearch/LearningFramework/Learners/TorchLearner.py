import numpy as np
import torch
import torch.nn as nn
from typing import Sequence, Optional, Union
from torch.utils.data import DataLoader, TensorDataset

from CodeResearch.LearningFramework.Learners.baseLearner import BaseLearner


class TorchMLPLearner(BaseLearner):
    """
    Простой learner на PyTorch для табличных данных (классификация).
    Архитектура задаётся в конструкторе, модель не хранится в стейте —
    создаётся в train() и передаётся в test()/update().
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_sizes: Sequence[int] = (128, 64),
        activation: str = "relu",
        dropout: float = 0.0,
        lr: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 20,
        update_epochs: int = 1,
        weight_decay: float = 0.0,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        input_dim    — размерность признакового пространства
        num_classes  — число классов (для CrossEntropyLoss, обычно >= 2)
        hidden_sizes — список размеров скрытых слоёв, например [512, 256]
        activation   — 'relu' | 'tanh' | 'elu' | 'gelu'
        dropout      — p для Dropout (0.0, если не нужен)
        lr           — learning rate
        batch_size   — размер батча
        epochs       — сколько эпох в train()
        update_epochs — сколько эпох в update()
        weight_decay — L2-регуляризация
        device       — 'cpu' | 'cuda' | torch.device(...) или None (авто)
        """
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_sizes = list(hidden_sizes)
        self.activation_name = activation.lower()
        self.dropout = float(dropout)
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.update_epochs = update_epochs
        self.weight_decay = weight_decay

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    # ---------- служебные методы построения сети / оптимизатора ----------

    def _activation_module(self) -> nn.Module:
        if self.activation_name == "relu":
            return nn.ReLU()
        if self.activation_name == "tanh":
            return nn.Tanh()
        if self.activation_name == "elu":
            return nn.ELU()
        if self.activation_name == "gelu":
            return nn.GELU()
        raise ValueError(f"Unknown activation: {self.activation_name}")

    def _build_model(self) -> nn.Module:
        layers = []
        in_features = self.input_dim

        if len(self.hidden_sizes) == 0:
            # Линейный классификатор
            layers.append(nn.Linear(in_features, self.num_classes))
        else:
            for h in self.hidden_sizes:
                layers.append(nn.Linear(in_features, h))
                layers.append(self._activation_module())
                if self.dropout > 0.0:
                    layers.append(nn.Dropout(self.dropout))
                in_features = h
            layers.append(nn.Linear(in_features, self.num_classes))

        model = nn.Sequential(*layers)
        return model

    def _make_optimizer(self, model: nn.Module):
        return torch.optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    # ---------- работа с данными ----------

    @staticmethod
    def _to_tensors(x, y, probs=None, device=None):
        # x
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32))
        elif not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        else:
            x = x.float()

        # y — метки классов [0..C-1]
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y.astype(np.int64))
        elif not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.int64)
        else:
            y = y.long()

        if probs is not None:
            if isinstance(probs, np.ndarray):
                probs = torch.from_numpy(probs.astype(np.float32))
            elif not isinstance(probs, torch.Tensor):
                probs = torch.tensor(probs, dtype=torch.float32)
            else:
                probs = probs.float()
        if device is not None:
            x = x.to(device)
            y = y.to(device)
            if probs is not None:
                probs = probs.to(device)
        return x, y, probs

    def _make_loader(self, x, y, probs=None, shuffle=True):
        if probs is None:
            dataset = TensorDataset(x, y)
        else:
            dataset = TensorDataset(x, y, probs)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    # ---------- одна эпоха обучения ----------

    def _train_one_epoch(self, model, optimizer, criterion, x, y, probs=None):
        model.train()
        x, y, probs = self._to_tensors(x, y, probs, device=self.device)
        loader = self._make_loader(x, y, probs, shuffle=True)

        total_loss = 0.0
        total_examples = 0

        for batch in loader:
            if len(batch) == 2:
                xb, yb = batch
                pb = None
            else:
                xb, yb, pb = batch

            optimizer.zero_grad()
            logits = model(xb)
            losses = criterion(logits, yb)  # [batch]
            if pb is not None:
                pb = pb.view_as(losses)
                losses = losses * pb
            loss = losses.mean()
            loss.backward()
            optimizer.step()

            bs = yb.size(0)
            total_loss += loss.item() * bs
            total_examples += bs

        return total_loss / max(total_examples, 1)

    # ---------- публичные методы BaseLearner ----------

    def train(self, x, y, probs=None):
        """
        Полное обучение модели на данных (несколько эпох).
        x, y, probs — np.array или torch.Tensor.
        Возвращает обученную модель (nn.Module).
        """
        model = self._build_model().to(self.device)
        criterion = nn.CrossEntropyLoss(reduction="none")
        optimizer = self._make_optimizer(model)

        for _ in range(self.epochs):
            _ = self._train_one_epoch(model, optimizer, criterion, x, y, probs)

        return model

    def update(self, model, x, y):
        """
        Дообучение переданной модели на новых данных (несколько эпох).
        Возвращает дообученную модель.
        """
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss(reduction="none")
        optimizer = self._make_optimizer(model)

        for _ in range(self.update_epochs):
            _ = self._train_one_epoch(model, optimizer, criterion, x, y, probs=None)

        return model

    def test(self, model, x, y):
        """
        Оценка accuracy и получение предсказаний.
        Возвращает:
            accuracy: float
            predictions: np.ndarray формы (N,) с предсказанными классами.
        """
        model = model.to(self.device)
        model.eval()
        x, y, _ = self._to_tensors(x, y, probs=None, device=self.device)
        loader = self._make_loader(x, y, probs=None, shuffle=False)

        correct = 0
        total = 0
        all_preds = []

        criterion = nn.CrossEntropyLoss(reduction="none")
        total_loss = 0.0

        with torch.no_grad():
            for xb, yb in loader:
                logits = model(xb)
                losses = criterion(logits, yb)
                loss = losses.mean()

                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu().numpy())

                correct += (preds == yb).sum().item()
                total += yb.size(0)
                total_loss += loss.item() * yb.size(0)

        acc = correct / max(total, 1)
        preds_np = np.concatenate(all_preds, axis=0) if all_preds else np.array([])

        # Если средний loss тоже нужен — можно вернуть тройку (acc, preds_np, avg_loss)
        # avg_loss = total_loss / max(total, 1)
        # return acc, preds_np, avg_loss

        return acc, preds_np

    def trainAndTest(self, x, y, probs, xt, yt):
        model = self.train(x, y, probs)
        return self.test(model, xt, yt)