import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from CodeResearch.LearningFramework.Learners.NNTorchModelLearner import TorchModelLearner

FeatureFn = Callable[[nn.Module, torch.Tensor], torch.Tensor]

@dataclass
class BossTrainResult:
    """
    Результат стандартного BossScoringLearner.train().
    """

    # Scoring-модель после initial training.
    model: nn.Module

    # Средний EL2N каждого объекта по всем scoring-эпохам.
    # Shape: [N].
    difficulty: np.ndarray

    # Features предпоследнего слоя после последней эпохи.
    # Shape: [N, feature_dim].
    features: np.ndarray

    # Исходные метки, shape [N].
    labels: np.ndarray

    # Заполняются только в trainAndTestOnEachEpoch().
    accuracies: list[Optional[float]] = field(default_factory=list)
    predictions: list[Optional[np.ndarray]] = field(default_factory=list)

@dataclass
class BossTrainTestResult:
    """
    Результат BossScoringLearner.trainAndTest().
    """

    train_result: BossTrainResult
    accuracy: float
    predictions: np.ndarray

class BossModelLearner(TorchModelLearner):
    """
    Initial-training learner для опубликованного BOSS.

    После каждой эпохи:
    1. выполняет отдельный проход по train set с shuffle=False;
    2. вычисляет EL2N каждого объекта;
    3. прибавляет его к накопленной сумме.

    После последней эпохи:
    1. делит сумму EL2N на число эпох;
    2. извлекает features предпоследнего слоя.

    Стандартный train() возвращает BossTrainResult,
    а не только nn.Module.
    """

    def __init__(
        self,
        *args,
        feature_fn: Optional[FeatureFn] = None,
        score_batch_size: Optional[int] = None,
        normalize_el2n: bool = True,
        verbose_scores: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.feature_fn = feature_fn

        self.score_batch_size = (
            self.batch_size
            if score_batch_size is None
            else int(score_batch_size)
        )

        if self.score_batch_size <= 0:
            raise ValueError("score_batch_size must be positive")

        self.normalize_el2n = bool(normalize_el2n)
        self.verbose_scores = bool(verbose_scores)

    # =========================================================
    # Вспомогательные методы
    # =========================================================

    def _make_boss_loader(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> DataLoader:
        """
        Порядок объектов должен быть постоянным, поскольку
        score[i] должен соответствовать исходному объекту i.
        """
        return DataLoader(
            TensorDataset(x, y),
            batch_size=self.score_batch_size,
            shuffle=False,
        )

    @staticmethod
    def _unwrap_model(model: nn.Module) -> nn.Module:
        """
        Получение исходной модели из torch.compile wrapper.
        """
        return getattr(model, "_orig_mod", model)

    def _compute_difficulty(
        self,
        model: nn.Module,
        x,
        y,
    ) -> np.ndarray:
        """
        Вычисляет EL2N всех объектов при текущем состоянии модели.

        D_i = ||softmax(f(x_i)) - one_hot(y_i)||_2

        Если normalize_el2n=True, дополнительно делит на sqrt(2),
        переводя теоретический диапазон EL2N в [0, 1].
        """
        x_tensor, y_tensor, _ = self._to_tensors(
            x,
            y,
            probs=None,
        )

        loader = self._make_boss_loader(
            x_tensor,
            y_tensor,
        )

        was_training = model.training
        model.eval()

        result = []

        with torch.inference_mode():
            for xb, yb in loader:
                if self.use_amp:
                    with torch.amp.autocast("cuda"):
                        logits = model(xb)
                else:
                    logits = model(xb)

                if not isinstance(logits, torch.Tensor):
                    raise TypeError(
                        "model(x) must return a tensor of logits"
                    )

                if logits.ndim != 2:
                    raise ValueError(
                        "Expected logits with shape [B, C], "
                        f"got {tuple(logits.shape)}"
                    )

                # Softmax лучше считать в float32, даже если
                # forward выполнялся через AMP.
                probabilities = torch.softmax(
                    logits.float(),
                    dim=1,
                )

                targets = F.one_hot(
                    yb,
                    num_classes=probabilities.shape[1],
                ).to(probabilities.dtype)

                difficulty = torch.linalg.vector_norm(
                    probabilities - targets,
                    ord=2,
                    dim=1,
                )

                if self.normalize_el2n:
                    difficulty = difficulty / math.sqrt(2.0)

                result.append(difficulty.cpu())

        model.train(was_training)

        if not result:
            return np.empty(0, dtype=np.float32)

        return (
            torch.cat(result)
            .numpy()
            .astype(np.float32, copy=False)
        )

    def _forward_features(
        self,
        model: nn.Module,
        xb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Извлекает выход предпоследнего слоя.
        """
        if self.feature_fn is not None:
            return self.feature_fn(model, xb)

        forward_features = getattr(
            model,
            "forward_features",
            None,
        )

        if callable(forward_features):
            return forward_features(xb)

        raise AttributeError(
            "Pass feature_fn=... or implement "
            "model.forward_features(x)."
        )

    def _extract_features(
        self,
        model: nn.Module,
        x,
        y,
    ) -> np.ndarray:
        """
        Извлекает final-epoch features для всех объектов.
        """
        feature_model = self._unwrap_model(model)

        x_tensor, y_tensor, _ = self._to_tensors(
            x,
            y,
            probs=None,
        )

        loader = self._make_boss_loader(
            x_tensor,
            y_tensor,
        )

        was_training = feature_model.training
        feature_model.eval()

        result = []

        with torch.inference_mode():
            for xb, _ in loader:
                if self.use_amp:
                    with torch.amp.autocast("cuda"):
                        features = self._forward_features(
                            feature_model,
                            xb,
                        )
                else:
                    features = self._forward_features(
                        feature_model,
                        xb,
                    )

                if not isinstance(features, torch.Tensor):
                    raise TypeError(
                        "Feature function must return torch.Tensor"
                    )

                if features.ndim < 2:
                    raise ValueError(
                        "Features must have shape [B, ...]"
                    )

                # Например, [B, C, 1, 1] -> [B, C].
                features = features.flatten(start_dim=1)

                result.append(
                    features.float().cpu()
                )

        feature_model.train(was_training)

        if not result:
            return np.empty((0, 0), dtype=np.float32)

        return (
            torch.cat(result, dim=0)
            .numpy()
            .astype(np.float32, copy=False)
        )

    # =========================================================
    # Общая реализация обучения
    # =========================================================

    def _train_boss(
        self,
        x,
        y,
        probs=None,
        xt=None,
        yt=None,
    ) -> BossTrainResult:
        """
        Общая реализация для train() и
        trainAndTestOnEachEpoch().
        """
        if self.epochs <= 0:
            raise ValueError(
                "BOSS initial training requires epochs > 0"
            )

        model = self.build_model().to(self.device)

        if self.shouldCompile:
            model = torch.compile(model)

        criterion = self._make_criterion()
        optimizer = self._make_optimizer(model)
        scaler = self._make_scaler()

        scheduler = self._make_scheduler(
            optimizer,
            total_epochs=self.epochs,
        )

        n_objects = len(y)

        # Храним только сумму по эпохам.
        difficulty_sum = np.zeros(
            n_objects,
            dtype=np.float64,
        )

        accuracies = []
        predictions = []

        start_time = time.time()

        for epoch in range(self.epochs):
            accuracy, epoch_predictions = self._train_one_epoch(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                criterion=criterion,
                x=x,
                y=y,
                probs=probs,
                xtest=xt,
                ytest=yt,
            )

            epoch_difficulty = self._compute_difficulty(
                model=model,
                x=x,
                y=y,
            )

            if epoch_difficulty.shape != (n_objects,):
                raise RuntimeError(
                    "Unexpected difficulty shape: "
                    f"{epoch_difficulty.shape}; "
                    f"expected {(n_objects,)}"
                )

            difficulty_sum += epoch_difficulty

            accuracies.append(accuracy)
            predictions.append(epoch_predictions)

            if self.verbose_scores:
                message = (
                    f"Epoch {epoch + 1}/{self.epochs}, "
                    f"time={time.time() - start_time:.1f}s, "
                    f"mean difficulty="
                    f"{epoch_difficulty.mean():.6f}"
                )

                if accuracy is not None:
                    message += f", test accuracy={accuracy:.6f}"

                print(message)

            if scheduler is not None:
                scheduler.step()

        mean_difficulty = (
            difficulty_sum / float(self.epochs)
        ).astype(np.float32)

        # В BOSS features берутся после последней
        # initial-training эпохи.
        features = self._extract_features(
            model=model,
            x=x,
            y=y,
        )

        if isinstance(y, torch.Tensor):
            labels = y.detach().cpu().numpy().astype(np.int64, copy=False)
        else:
            labels = np.asarray(y, dtype=np.int64)

        return BossTrainResult(
            model=model,
            difficulty=mean_difficulty,
            features=features,
            accuracies=accuracies,
            predictions=predictions,
            labels=labels
        )

    # =========================================================
    # Реализация стандартного интерфейса BaseLearner
    # =========================================================

    def train(
        self,
        x,
        y,
        probs=None,
    ) -> BossTrainResult:
        """
        Стандартный train(), но результатом является структура:

            BossTrainResult(
                model,
                difficulty,
                features,
                ...
            )
        """
        return self._train_boss(
            x=x,
            y=y,
            probs=probs,
            xt=None,
            yt=None,
        )

    def trainAndTest(
        self,
        x,
        y,
        probs,
        xt,
        yt,
    ) -> BossTrainTestResult:
        """
        Нельзя использовать реализацию TorchModelLearner,
        потому что self.train() теперь возвращает структуру,
        а не непосредственно модель.
        """
        train_result = self.train(
            x=x,
            y=y,
            probs=probs,
        )

        accuracy, predictions = self.test(
            train_result.model,
            xt,
            yt,
        )

        return BossTrainTestResult(
            train_result=train_result,
            accuracy=accuracy,
            predictions=predictions,
        )

    def trainAndTestOnEachEpoch(
        self,
        x,
        y,
        probs,
        xt,
        yt,
    ) -> BossTrainResult:
        """
        Возвращает тот же BossTrainResult, но поля
        accuracies и predictions заполнены для каждой эпохи.
        """
        return self._train_boss(
            x=x,
            y=y,
            probs=probs,
            xt=xt,
            yt=yt,
        )