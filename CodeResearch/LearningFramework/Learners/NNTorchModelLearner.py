from typing import Callable, Optional, Union, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from CodeResearch.LearningFramework.Learners.TorchLearner import TorchLearner

ModelFactory = Callable[[], nn.Module]


class TorchModelLearner (TorchLearner):
    """
    Универсальный learner на PyTorch:
    - модель задаётся фабрикой model_factory: () -> nn.Module
    - поддерживает Adam/SGD, cosine/step schedulers, AMP
    - подходит для MLP/CNN (важно: x должен быть уже нужной формы для модели)
    """

    def __init__(self, model_factory: ModelFactory, lr: float = 1e-3, batch_size: int = 64, epochs: int = 20,
                 update_epochs: int = 1, weight_decay: float = 0.0, device: Optional[Union[str, torch.device]] = None,
                 optimizer_name: str = "adam", momentum: float = 0.9, nesterov: bool = True,
                 scheduler_name: str = "none", cosine_tmax: Optional[int] = None, min_lr: float = 0.0,
                 step_size: int = 60, gamma: float = 0.2, use_amp: bool = False, label_smoothing: float = 0.0):
        super().__init__(device)
        self.model_factory = model_factory
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.update_epochs = int(update_epochs)
        self.weight_decay = float(weight_decay)

        self.optimizer_name = optimizer_name.lower()
        self.momentum = float(momentum)
        self.nesterov = bool(nesterov)

        self.scheduler_name = scheduler_name.lower()
        self.cosine_tmax = cosine_tmax
        self.min_lr = float(min_lr)
        self.step_size = int(step_size)
        self.gamma = float(gamma)

        self.use_amp = bool(use_amp) and (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        self.label_smoothing = float(label_smoothing)

    # ----------------- factories -----------------

    def build_model(self) -> nn.Module:
        m = self.model_factory()
        if not isinstance(m, nn.Module):
            raise TypeError("model_factory must return torch.nn.Module")
        return m

    def _make_criterion(self) -> nn.Module:
        # reduction="none" нужно, чтобы поддержать pb-веса per-sample
        return nn.CrossEntropyLoss(reduction="none", label_smoothing=self.label_smoothing)

    def _make_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        if self.optimizer_name == "adam":
            return torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.optimizer_name == "sgd":
            return torch.optim.SGD(
                model.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                nesterov=self.nesterov,
            )
        raise ValueError("optimizer_name must be 'adam' or 'sgd'")

    def _make_scheduler(self, optimizer: torch.optim.Optimizer, total_epochs: int):
        if self.scheduler_name == "none":
            return None
        if self.scheduler_name == "cosine":
            tmax = self.cosine_tmax if self.cosine_tmax is not None else total_epochs
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tmax, eta_min=self.min_lr)
        if self.scheduler_name == "step":
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        raise ValueError("scheduler_name must be 'none', 'cosine', or 'step'")

    # ----------------- data utils -----------------

    @staticmethod
    def _to_tensor_any(x: Any, dtype: torch.dtype) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(dtype=dtype)
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(dtype=dtype)
        return torch.tensor(x, dtype=dtype)

    def _to_tensors(self, x, y, probs=None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        x = self._to_tensor_any(x, torch.float32)
        y = self._to_tensor_any(y, torch.int64)

        if probs is not None:
            probs = self._to_tensor_any(probs, torch.float32)

        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        if probs is not None:
            probs = probs.to(self.device, non_blocking=True)

        return x, y, probs

    def _make_loader(self, x: torch.Tensor, y: torch.Tensor, probs=None, shuffle=True) -> DataLoader:
        ds = TensorDataset(x, y) #if probs is None else TensorDataset(x, y, probs) #todo: check if probs really necessary here
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    # ----------------- train helpers -----------------

    def _train_one_epoch(self, model, optimizer, criterion, x, y, probs=None) -> float:
        model.train()
        x, y, probs = self._to_tensors(x, y, probs)
        loader = self._make_loader(x, y, probs, shuffle=True)

        total_loss = 0.0
        total_n = 0

        for batch in loader:
            if len(batch) == 2:
                xb, yb = batch
                pb = None
            else:
                xb, yb, pb = batch

            optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    logits = model(xb)
                    losses = criterion(logits, yb)
                    if pb is not None:
                        pb = pb.view_as(losses)
                        losses = losses * pb
                    loss = losses.mean()
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                logits = model(xb)
                losses = criterion(logits, yb)
                if pb is not None:
                    pb = pb.view_as(losses)
                    losses = losses * pb
                loss = losses.mean()
                loss.backward()
                optimizer.step()

            bs = int(yb.size(0))
            total_loss += float(loss.item()) * bs
            total_n += bs

        return total_loss / max(total_n, 1)

    # ----------------- public API -----------------

    def train(self, x, y, probs=None) -> nn.Module:
        model = self.build_model().to(self.device)
        criterion = self._make_criterion()
        optimizer = self._make_optimizer(model)
        scheduler = self._make_scheduler(optimizer, total_epochs=self.epochs)

        for _ in range(self.epochs):
            _ = self._train_one_epoch(model, optimizer, criterion, x, y, probs)
            if scheduler is not None:
                scheduler.step()

        return model

    def update(self, model: nn.Module, x, y) -> nn.Module:
        model = model.to(self.device)
        criterion = self._make_criterion()
        optimizer = self._make_optimizer(model)
        scheduler = self._make_scheduler(optimizer, total_epochs=self.update_epochs)

        for _ in range(self.update_epochs):
            _ = self._train_one_epoch(model, optimizer, criterion, x, y, probs=None)
            if scheduler is not None:
                scheduler.step()

        return model

    def test(self, model: nn.Module, x, y):
        model = model.to(self.device)
        model.eval()

        x, y, _ = self._to_tensors(x, y, probs=None)
        loader = self._make_loader(x, y, probs=None, shuffle=False)

        correct = 0
        total = 0
        all_preds = []

        criterion = self._make_criterion()

        with torch.no_grad():
            for xb, yb in loader:
                logits = model(xb)
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.detach().cpu().numpy())

                correct += int((preds == yb).sum().item())
                total += int(yb.size(0))

        acc = correct / max(total, 1)
        preds_np = np.concatenate(all_preds, axis=0) if all_preds else np.array([])
        return acc, preds_np

    def trainAndTest(self, x, y, probs, xt, yt):
        model = self.train(x, y, probs)
        return self.test(model, xt, yt)
