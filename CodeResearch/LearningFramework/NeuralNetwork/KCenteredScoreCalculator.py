from CodeResearch.LearningFramework.NeuralNetwork.BaseScoreCalculator import BaseScoreCalculator
import math
import numpy as np
import torch
import torch.nn as nn

class KCenteredScoreCalculator(BaseScoreCalculator):
    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def calculateScore(self, model, batches, device):
        """
        K-Center Greedy score calculator.

        Возвращает:
            scores: np.ndarray [N], dtype float32

        Логика:
        - строим эмбеддинги всех объектов;
        - выбираем top-alpha объектов через k-center greedy;
        - score выбранного объекта = расстояние до ближайшего уже выбранного центра
          на момент добавления;
        - первый объект не имеет настоящего acquisition score, поэтому ему задается
          score = score второго объекта + epsilon;
        - всем объектам вне top-alpha присваивается 0.

        Важно:
        - scores идут в порядке прохождения объектов через batches;
        - yb не используется;
        - если batches имеет shuffle=True, порядок scores будет соответствовать
          порядку выдачи batches, а не исходному порядку датасета.
        """
        model.eval()

        chunk_size = 8192
        eps = 1e-6

        embeddings = []

        use_explicit_features = hasattr(model, "features")

        hook_handle = None
        captured = {}

        if not use_explicit_features:
            last_linear = self._find_last_linear(model)

            if last_linear is None:
                raise TypeError(
                    "Не удалось найти nn.Linear слой для извлечения эмбеддингов. "
                    "Добавь model.features или используй модель с финальным nn.Linear."
                )

            def _hook(module, inputs, output):
                feat = inputs[0]

                if isinstance(feat, (tuple, list)):
                    feat = feat[0]

                if feat.ndim > 2:
                    feat = feat.flatten(1)

                captured["feat"] = feat.detach()

            hook_handle = last_linear.register_forward_hook(_hook)

        try:
            with torch.inference_mode():
                for batch in batches:
                    if len(batch) == 3:
                        xb, yb, _idx = batch
                    else:
                        xb, yb = batch

                    xb = xb.to(device, non_blocking=True)

                    if use_explicit_features:
                        feat = self._extract_features_default(model, xb)
                    else:
                        captured.clear()
                        _ = model(xb)

                        if "feat" not in captured:
                            raise RuntimeError(
                                "Forward hook не смог извлечь вход в последний Linear слой."
                            )

                        feat = captured["feat"]

                    embeddings.append(feat.detach().float().cpu())

        finally:
            if hook_handle is not None:
                hook_handle.remove()

        if len(embeddings) == 0:
            return np.zeros(0, dtype=np.float32)

        X_cpu = torch.cat(embeddings, dim=0).float()
        n = X_cpu.shape[0]

        scores = np.zeros(n, dtype=np.float32)

        if n == 0 or self.alpha <= 0:
            return scores

        # alpha <= 1 трактуем как долю датасета.
        # alpha > 1 трактуем как абсолютное число объектов.
        if self.alpha <= 1:
            k = int(math.ceil(self.alpha * n))
        else:
            k = int(self.alpha)

        k = max(0, min(k, n))

        if k == 0:
            return scores

        # Пробуем перенести эмбеддинги на device.
        # Если не помещаются в GPU-память, считаем на CPU.
        try:
            X = X_cpu.to(device=device, dtype=torch.float32)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                X = X_cpu
            else:
                raise

        selected_indices = []
        selected_scores = []

        with torch.inference_mode():
            # ------------------------------------------------------------
            # 1. Инициализация первого центра
            # ------------------------------------------------------------
            # В оригинальной постановке обычно есть начальное множество s0.
            # Тогда первый новый объект выбирается как самый дальний от s0.
            #
            # В твоей сигнатуре s0 нет, поэтому первый центр нужно выбрать
            # отдельным соглашением. Берем объект, ближайший к среднему
            # эмбеддингу: это детерминированная и обычно стабильная инициализация.
            # ------------------------------------------------------------
            mean = X.mean(dim=0)
            dist_to_mean = self._sq_dist_to_point(X, mean, chunk_size)
            first_idx = int(torch.argmin(dist_to_mean).item())

            selected_indices.append(first_idx)

            selected_mask = torch.zeros(n, dtype=torch.bool, device=X.device)
            selected_mask[first_idx] = True

            # min_dist[i] = расстояние от i-го объекта до ближайшего
            # уже выбранного центра.
            min_dist = self._sq_dist_to_point(X, X[first_idx], chunk_size)

            # Уже выбранные точки больше выбирать нельзя.
            min_dist[selected_mask] = -1.0

            # ------------------------------------------------------------
            # 2. Score первого объекта
            # ------------------------------------------------------------
            # У первого объекта нет настоящего k-center score, потому что
            # до него множество выбранных центров было пустым.
            #
            # Поэтому задаем ему score чуть выше максимального score,
            # который получит следующий объект.
            # ------------------------------------------------------------
            if n > 1:
                first_score = float(min_dist.max().item()) + eps
            else:
                first_score = 1.0

            selected_scores.append(first_score)

            # ------------------------------------------------------------
            # 3. K-Center Greedy
            # ------------------------------------------------------------
            for _ in range(1, k):
                # Берем объект, максимально удаленный от ближайшего
                # уже выбранного центра.
                next_idx = int(torch.argmax(min_dist).item())
                next_score = float(min_dist[next_idx].item())

                selected_indices.append(next_idx)
                selected_scores.append(next_score)

                selected_mask[next_idx] = True

                # Обновляем расстояние до ближайшего выбранного центра:
                # min_dist[i] = min(
                #     старое расстояние до выбранных центров,
                #     расстояние до нового центра
                # )
                new_dist = self._sq_dist_to_point(X, X[next_idx], chunk_size)
                min_dist = torch.minimum(min_dist, new_dist)

                # Уже выбранные точки повторно выбирать нельзя.
                min_dist[selected_mask] = -1.0

        for idx, score in zip(selected_indices, selected_scores):
            scores[idx] = score

        return scores

    @staticmethod
    def _extract_features_default(model, xb):
        """
        Извлечение эмбеддингов под архитектуру:

            model.features -> model.pool -> flatten -> model.head

        Возвращает признаки перед линейной головой.
        """
        feat = model.features(xb)

        if isinstance(feat, (tuple, list)):
            feat = feat[0]

        if hasattr(model, "pool") and model.pool is not None:
            feat = model.pool(feat)

        if feat.ndim > 2:
            feat = feat.flatten(1)

        return feat

    @staticmethod
    def _find_last_linear(model):
        """
        Находит последний nn.Linear в модели.

        Если model.features отсутствует, используем forward hook на последний
        Linear и берем его input как penultimate embedding.
        """
        last_linear = None

        for module in model.modules():
            if isinstance(module, nn.Linear):
                last_linear = module

        return last_linear

    @staticmethod
    def _sq_dist_to_point(X, point, chunk_size):
        """
        Squared L2 distance от всех строк X до одной точки point.

        Почему squared L2:
        - обычный L2 и squared L2 дают одинаковый порядок выбора;
        - squared L2 быстрее, потому что не нужен sqrt.

        Аргументы:
            X: torch.Tensor [N, D]
            point: torch.Tensor [D] или [1, D]
            chunk_size: размер чанка для экономии памяти

        Возвращает:
            dist: torch.Tensor [N]
        """
        if point.ndim == 1:
            point = point.unsqueeze(0)

        n = X.shape[0]
        dist = torch.empty(n, device=X.device, dtype=torch.float32)

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            diff = X[start:end] - point
            dist[start:end] = (diff * diff).sum(dim=1)

        return dist