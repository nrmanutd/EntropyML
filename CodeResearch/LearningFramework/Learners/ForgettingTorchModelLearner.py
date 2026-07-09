import time
from typing import Optional, Dict, Any

import numpy as np
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from CodeResearch.LearningFramework.Learners.NNTorchModelLearner import TorchModelLearner

class ForgettingTorchModelLearner(TorchModelLearner):
    """
    TorchModelLearner с расчетом Example Forgetting score.

    Что возвращает train():
        model, scores, stats

    где:
        scores[i] — forgetting score для объекта x[i], y[i]

    Интерпретация score:
        - обычный объект: score = число forgetting events;
        - never-learned объект: score > max(forgetting_counts),
          то есть он считается сложнее всех объектов, которые хотя бы раз
          были выучены.

    Forgetting event:
        объект раньше был классифицирован правильно,
        а при следующем появлении в mini-batch стал классифицирован неправильно.

    Важно:
        - score считается только для объектов, участвующих в обучении;
        - порядок scores соответствует порядку x, y, переданных в train().
    """

    def __init__(
        self,
        *args,
        never_learned_policy: str = "max_plus_one_with_margin",
        **kwargs,
    ):
        """
        never_learned_policy:
            "max_plus_one":
                never-learned объектам присваивается max(forgetting_counts) + 1

            "max_plus_one_with_margin":
                never-learned объектам присваивается score выше всех остальных,
                а внутри never-learned группы используется tie-breaker по margin:
                чем более отрицательный последний margin, тем выше score.
        """
        super().__init__(*args, **kwargs)

        valid_policies = {"max_plus_one", "max_plus_one_with_margin"}
        if never_learned_policy not in valid_policies:
            raise ValueError(
                f"never_learned_policy must be one of {valid_policies}, "
                f"got {never_learned_policy}"
            )

        self.never_learned_policy = never_learned_policy

        self.forgetting_scores_ = None
        self.forgetting_stats_ = None

        self._prev_acc = None
        self._last_acc = None
        self._forgetting_counts = None
        self._learning_counts = None
        self._presentations = None
        self._was_ever_learned = None
        self._first_learned_presentation = None
        self._last_margin = None

    # ------------------------------------------------------------------
    # Forgetting state
    # ------------------------------------------------------------------

    def _init_forgetting_state(self, n: int):
        """
        Инициализация статистик для N обучающих объектов.
        """
        self._prev_acc = np.zeros(n, dtype=np.int8)
        self._last_acc = np.zeros(n, dtype=np.int8)

        self._forgetting_counts = np.zeros(n, dtype=np.int32)
        self._learning_counts = np.zeros(n, dtype=np.int32)

        self._presentations = np.zeros(n, dtype=np.int32)
        self._was_ever_learned = np.zeros(n, dtype=bool)
        self._first_learned_presentation = np.full(n, -1, dtype=np.int32)

        # margin = logit(correct class) - max logit(other classes)
        self._last_margin = np.full(n, np.nan, dtype=np.float32)

    def _update_forgetting_statistics(self, idxb, logits, yb):
        """
        Обновляет forgetting statistics для текущего mini-batch.

        Важно:
            вызывается ДО optimizer.step().
        """
        with torch.no_grad():
            preds = torch.argmax(logits.detach(), dim=1)
            curr_acc_t = (preds == yb).to(torch.int8)

            idx_np = idxb.detach().cpu().numpy().astype(np.int64)
            curr_acc_np = curr_acc_t.detach().cpu().numpy().astype(np.int8)

            prev_acc_np = self._prev_acc[idx_np]

            forgetting_mask = (prev_acc_np == 1) & (curr_acc_np == 0)
            learning_mask = (prev_acc_np == 0) & (curr_acc_np == 1)

            self._forgetting_counts[idx_np[forgetting_mask]] += 1
            self._learning_counts[idx_np[learning_mask]] += 1

            self._presentations[idx_np] += 1

            first_learned_mask = (
                (curr_acc_np == 1)
                & (~self._was_ever_learned[idx_np])
            )

            if np.any(first_learned_mask):
                first_idx = idx_np[first_learned_mask]
                self._first_learned_presentation[first_idx] = self._presentations[first_idx]

            self._was_ever_learned[idx_np[curr_acc_np == 1]] = True

            self._prev_acc[idx_np] = curr_acc_np
            self._last_acc[idx_np] = curr_acc_np

            self._last_margin[idx_np] = self._compute_margin(logits, yb)

    @staticmethod
    def _compute_margin(logits, yb):
        """
        margin = logit(correct class) - max logit(other classes)

        Если margin < 0, объект классифицирован неправильно.
        Чем margin более отрицательный, тем увереннее ошибка модели.
        """
        logits_cpu = logits.detach().float().cpu()
        y_cpu = yb.detach().long().cpu()

        batch_size = logits_cpu.shape[0]
        row_idx = torch.arange(batch_size)

        correct_logits = logits_cpu[row_idx, y_cpu]

        other_logits = logits_cpu.clone()
        other_logits[row_idx, y_cpu] = -float("inf")
        max_other_logits = other_logits.max(dim=1).values

        margin = correct_logits - max_other_logits

        return margin.numpy().astype(np.float32)

    # ------------------------------------------------------------------
    # Loader with indices
    # ------------------------------------------------------------------

    def _make_loader_with_indices(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        probs=None,
        shuffle: bool = True,
    ) -> DataLoader:
        """
        Создает DataLoader, который возвращает не только xb, yb,
        но и idxb — индексы объектов внутри исходного x, y.

        Без idxb невозможно корректно записать forgetting statistics
        обратно в scores[i].
        """
        idx = torch.arange(x.shape[0], dtype=torch.long, device=x.device)

        if probs is None:
            ds = TensorDataset(x, y, idx)

            return DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=shuffle,
            )

        ds = TensorDataset(x, y, probs, idx)

        # WeightedRandomSampler лучше кормить CPU double weights.
        if isinstance(probs, torch.Tensor):
            weights = probs.detach().cpu().double()
        else:
            weights = torch.as_tensor(probs, dtype=torch.double)

        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(ds),
            replacement=False,
        )

        return DataLoader(
            ds,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=False,
        )

    # ------------------------------------------------------------------
    # Train helpers
    # ------------------------------------------------------------------

    def _train_one_epoch(
        self,
        model,
        optimizer,
        scaler,
        criterion,
        x,
        y,
        probs=None,
        xtest=None,
        ytest=None,
    ):
        """
        Один epoch обучения + обновление forgetting statistics.

        Возвращает:
            accuracies, predictions

        Совместимо с базовой логикой _trainModel:
            r = self._train_one_epoch(...)
            accuracies.append(r[0])
            predictions.append(r[1])
        """
        model.train()

        x, y, probs = self._to_tensors(x, y, probs)
        loader = self._make_loader_with_indices(x, y, probs, shuffle=True)

        total_loss = 0.0
        total_n = 0

        for batch in loader:
            if len(batch) == 3:
                xb, yb, idxb = batch
                pb = None
            elif len(batch) == 4:
                xb, yb, pb, idxb = batch
            else:
                raise RuntimeError(
                    f"Unexpected batch format: len(batch)={len(batch)}"
                )

            optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(xb)
                    losses = criterion(logits, yb)

                    if pb is not None:
                        pb = pb.view_as(losses)
                        losses = losses * pb

                    loss = losses.mean()

                # В статье correctness фиксируется при предъявлении объекта.
                # Поэтому обновляем forgetting ДО optimizer.step().
                self._update_forgetting_statistics(idxb, logits, yb)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            else:
                logits = model(xb)
                losses = criterion(logits, yb)

                if pb is not None:
                    pb = pb.view_as(losses)
                    losses = losses * pb

                loss = losses.mean()

                # В статье correctness фиксируется при предъявлении объекта.
                # Поэтому обновляем forgetting ДО optimizer.step().
                self._update_forgetting_statistics(idxb, logits, yb)

                loss.backward()
                optimizer.step()

            bs = int(yb.size(0))
            total_loss += float(loss.item()) * bs
            total_n += bs

        accuracies = None
        predictions = None

        if xtest is not None:
            was_training = model.training
            accuracies, predictions = self.test(model, xtest, ytest)
            model.train(was_training)

        return accuracies, predictions

    # ------------------------------------------------------------------
    # Score construction
    # ------------------------------------------------------------------

    def _build_forgetting_scores(self) -> np.ndarray:
        """
        Строит итоговый score.

        Обычные объекты:
            score = number of forgetting events

        Never-learned объекты:
            score строго выше всех остальных.
        """
        scores = self._forgetting_counts.astype(np.float32).copy()

        never_learned = ~self._was_ever_learned

        if np.any(never_learned):
            max_score = float(scores.max()) if scores.size > 0 else 0.0
            base_score = max_score + 1.0

            if self.never_learned_policy == "max_plus_one":
                scores[never_learned] = base_score

            elif self.never_learned_policy == "max_plus_one_with_margin":
                # Для never-learned объектов последний margin должен быть < 0.
                # Чем margin более отрицательный, тем объект сложнее.
                severity = -self._last_margin[never_learned]
                severity = np.nan_to_num(
                    severity,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).astype(np.float32)

                if severity.size > 0 and severity.max() > severity.min():
                    severity = (severity - severity.min()) / (
                        severity.max() - severity.min()
                    )
                else:
                    severity = np.zeros_like(severity, dtype=np.float32)

                # Все never-learned выше всех learned,
                # но внутри группы есть небольшой порядок по severity.
                scores[never_learned] = base_score + severity

        return scores.astype(np.float32)

    def _build_forgetting_stats(self, scores: np.ndarray) -> Dict[str, Any]:
        """
        Возвращает подробную диагностику по каждому объекту.
        """
        return {
            "scores": scores.copy(),
            "forgetting_counts": self._forgetting_counts.copy(),
            "learning_counts": self._learning_counts.copy(),
            "was_ever_learned": self._was_ever_learned.copy(),
            "never_learned": (~self._was_ever_learned).copy(),
            "presentations": self._presentations.copy(),
            "first_learned_presentation": self._first_learned_presentation.copy(),
            "last_acc": self._last_acc.copy(),
            "last_margin": self._last_margin.copy(),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, x, y, probs=None):
        """
        Обучает модель и возвращает forgetting scores.

        Возвращает:
            model, scores, stats
        """
        model = self.build_model().to(self.device)

        (
            model,
            optimizer,
            scaler,
            accuracies,
            predictions,
            scores,
            stats,
        ) = self._trainModel(
            model=model,
            x=x,
            y=y,
            probs=probs,
            epochs=self.epochs,
            xt=None,
            yt=None,
        )

        return model, scores, stats

    def update(self, model, x, y):
        """
        Дообучает существующую модель и возвращает forgetting scores
        за период update.

        Возвращает:
            model, scores, stats

        Важно:
            это НЕ продолжает старые forgetting statistics.
            Для update statistics считаются заново на переданном x, y.
        """
        model = model.to(self.device)

        (
            model,
            optimizer,
            scaler,
            accuracies,
            predictions,
            scores,
            stats,
        ) = self._trainModel(
            model=model,
            x=x,
            y=y,
            probs=None,
            epochs=self.update_epochs,
            xt=None,
            yt=None,
        )

        return model, scores, stats

    def trainAndTest(self, x, y, probs, xt, yt):
        """
        Совместимая с базовым классом сигнатура.

        Возвращает:
            acc, preds_np, scores, stats
        """
        model, scores, stats = self.train(x, y, probs)

        acc, preds_np = self.test(model, xt, yt)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return acc, preds_np, scores, stats

    def trainAndTestOnEachEpoch(self, x, y, probs, xt, yt):
        """
        Обучает модель, тестирует после каждой эпохи,
        дополнительно возвращает forgetting scores.

        Возвращает:
            model, accuracies, predictions, scores, stats
        """
        model = self.build_model().to(self.device)

        (
            model,
            optimizer,
            scaler,
            accuracies,
            predictions,
            scores,
            stats,
        ) = self._trainModel(
            model=model,
            x=x,
            y=y,
            probs=probs,
            epochs=self.epochs,
            xt=xt,
            yt=yt,
        )

        return model, accuracies, predictions, scores, stats

    def _trainModel(
        self,
        model,
        x,
        y,
        probs,
        epochs,
        xt,
        yt,
        optimizer: Optional[Optimizer] = None,
        scaler=None,
    ):
        """
        Переопределенная версия TorchModelLearner._trainModel.

        Возвращает:
            model,
            optimizer,
            scaler,
            accuracies,
            predictions,
            scores,
            stats
        """
        if self.shouldCompile:
            model = torch.compile(model)

        n = int(len(y))
        self._init_forgetting_state(n)

        criterion = self._make_criterion()

        if optimizer is None:
            optimizer = self._make_optimizer(model)

        if scaler is None:
            scaler = self._make_scaler()

        scheduler = self._make_scheduler(optimizer, total_epochs=epochs)

        accuracies = []
        predictions = []

        t1 = time.time()

        for epoch_idx in range(epochs):
            acc, pred = self._train_one_epoch(
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

            if acc is not None:
                print(
                    f"Epoch #{epoch_idx + 1}({epochs}) "
                    f"{time.time() - t1:.2f}s: {acc}"
                )

            accuracies.append(acc)
            predictions.append(pred)

            if scheduler is not None:
                scheduler.step()

        scores = self._build_forgetting_scores()
        stats = self._build_forgetting_stats(scores)

        self.forgetting_scores_ = scores
        self.forgetting_stats_ = stats

        return model, optimizer, scaler, accuracies, predictions, scores, stats