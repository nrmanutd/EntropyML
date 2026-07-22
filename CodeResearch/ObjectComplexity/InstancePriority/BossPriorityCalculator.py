import heapq
import math
from typing import Sequence

import numpy as np

from CodeResearch.Helpers.Logger.BaseLogger import BaseLogger
from CodeResearch.LearningFramework.DataProcessing.BaseDataProcessor import BaseDataProcessor
from CodeResearch.LearningFramework.Learners.BossModelLearner import BossTrainResult
from CodeResearch.ObjectComplexity.InstancePriority.basePriorityCalculator import BasePriorityCalculator


class BossPriorityCalculator(BasePriorityCalculator):
    def __init__(self, betas, learnerCreator, logger: BaseLogger):
        self.logger = logger
        self.betas = betas
        self.learnerCreator = learnerCreator

    def calculatePriority(self, dataSet, target):
        learner = self.learnerCreator()

        self.logger.logDebug('Training learner for boss ranging...')
        result = learner.train(dataSet, target, np.full(len(target), 1.0 / len(target)))
        self.logger.logDebug('Model is trained')

        resultPriorities = []
        probs = []

        for beta in self.betas:
            curNTrain = math.ceil(beta * len(target))
            self.logger.logDebug(f'Calculating boss indexes for beta {beta}...')
            curIdxes = self.calculateBossPriority(result, curNTrain, beta)
            self.logger.logDebug(f'Finished calculating boss indexes for {beta}.')

            resultPriorities.append(curIdxes)
            probs.append(np.full(curNTrain, 1.0 / curNTrain))

        return resultPriorities, probs

    def calculateBossPriority(self, result: BossTrainResult, curNTrain: int, beta: float):
        mean_difficulty = result.difficulty.mean()
        a = 1.0 + mean_difficulty + 0.0002 * curNTrain
        b = 2.0 + 0.0001 * curNTrain

        cutOffMap = {0.05: 0.5, 0.1: 0.5, 0.2: 0.4, 0.5: 0.2}

        return self.select_boss_subset(a, b, curNTrain, result, cutOffMap[beta])

    def select_boss_subset(self, a: float, b: float, subset_size: int, result: BossTrainResult, beta: float = 0.0) -> np.ndarray:
        """
    Отбирает subset с помощью BOSS.

    Objective:
        F(S) = sum_{i in V} max_{j in S}
               Sim(z_i, z_j) * I_j,

    где:
        I_j = Beta(D_j; a, b).

    Перед greedy-selection глобально удаляется beta-доля
    наиболее трудных объектов:

        V' = V \\ {beta * |V| hardest samples}.

    Удаленные объекты не могут быть выбраны в S, но остаются
    среди объектов V, которые выбранный subset должен покрывать.

    Отбор выполняется class-balanced: общий бюджет распределяется
    между классами, а внутри каждого класса запускается
    Beta-weighted facility-location greedy.

    Parameters
    ----------
    a, b:
        Параметры Beta-scoring function.

    subset_size:
        Абсолютное количество выбираемых объектов.

    result:
        Результат обучения BossScoringLearner:
            result.difficulty: [N], difficulty в [0, 1];
            result.features:   [N, feature_dim];
            result.labels:     [N].

    beta:
        Доля наиболее трудных объектов, исключаемых из кандидатов.

        Например:
            beta=0.0  — ничего не удалять;
            beta=0.1  — удалить 10% объектов с максимальным D_i.

        Должно выполняться:
            0 <= beta < 1.

    Returns
    -------
    selected_indices:
        np.ndarray формы [subset_size], dtype int64.
        Индексы соответствуют исходным x и y.
    """

        features = np.asarray(result.features, dtype=np.float32)
        difficulty = np.asarray(result.difficulty, dtype=np.float64)
        labels = np.asarray(result.labels, dtype=np.int64)

        self._validate_boss_inputs(
            a=a,
            b=b,
            beta=beta,
            subset_size=subset_size,
            features=features,
            difficulty=difficulty,
            labels=labels,
        )

        n_objects = features.shape[0]

        # ---------------------------------------------------------
        # 1. Beta importance для всех объектов
        # ---------------------------------------------------------

        importance = self._calculate_beta_importance(
            difficulty=difficulty,
            a=a,
            b=b,
        )

        # ---------------------------------------------------------
        # 2. Глобально удаляем beta * N наиболее трудных объектов
        # ---------------------------------------------------------

        prune_count = int(math.floor(beta * n_objects))

        candidate_mask = np.ones(
            n_objects,
            dtype=bool,
        )

        if prune_count > 0:
            # Stable sort делает результат детерминированным
            # при одинаковых difficulty.
            hardest_indices = np.argsort(
                -difficulty,
                kind="stable",
            )[:prune_count]

            candidate_mask[hardest_indices] = False

        candidate_count = int(candidate_mask.sum())

        if subset_size > candidate_count:
            raise ValueError(
                f"After pruning {prune_count} objects, only "
                f"{candidate_count} candidates remain, but "
                f"subset_size={subset_size} was requested."
            )

        # ---------------------------------------------------------
        # 3. Class-balanced распределение бюджета
        # ---------------------------------------------------------

        classes = np.unique(labels)

        candidate_class_counts = np.asarray(
            [
                np.count_nonzero(
                    candidate_mask & (labels == class_label)
                )
                for class_label in classes
            ],
            dtype=np.int64,
        )

        class_budgets = self._allocate_class_balanced_budgets(
            class_counts=candidate_class_counts,
            subset_size=subset_size,
        )

        selected_global: list[int] = []

        # ---------------------------------------------------------
        # 4. Greedy selection внутри каждого класса
        # ---------------------------------------------------------

        for class_label, class_budget in zip(
            classes,
            class_budgets,
        ):
            class_budget = int(class_budget)

            if class_budget == 0:
                continue

            # Покрывать нужно все объекты класса, включая удаленные.
            coverage_indices = np.flatnonzero(
                labels == class_label
            )

            # Выбирать можно только из V'.
            candidate_indices = np.flatnonzero(
                (labels == class_label) & candidate_mask
            )

            if class_budget > candidate_indices.size:
                raise RuntimeError(
                    f"Class {class_label}: requested budget "
                    f"{class_budget}, but only "
                    f"{candidate_indices.size} candidates remain."
                )

            selected_candidate_positions = (
                self._boss_lazy_greedy_rectangular(
                    coverage_features=features[coverage_indices],
                    candidate_features=features[candidate_indices],
                    candidate_importance=importance[candidate_indices],
                    subset_size=class_budget,
                )
            )

            selected_class_global_indices = candidate_indices[
                selected_candidate_positions
            ]

            selected_global.extend(
                selected_class_global_indices.tolist()
            )

        selected_indices = np.asarray(
            selected_global,
            dtype=np.int64,
        )

        if selected_indices.shape != (subset_size,):
            raise RuntimeError(
                "BOSS returned an incorrect number of objects: "
                f"{selected_indices.size}; expected {subset_size}."
            )

        if np.unique(selected_indices).size != subset_size:
            raise RuntimeError(
                "BOSS returned duplicate indices."
            )

        if not np.all(candidate_mask[selected_indices]):
            raise RuntimeError(
                "BOSS selected at least one pruned object."
            )

        return selected_indices

    def _calculate_beta_importance(self,
            difficulty: np.ndarray,
            a: float,
            b: float,
            eps: float = 1e-7,
    ) -> np.ndarray:
        """
        Вычисляет ненормированную Beta density:

            I(D) ∝ D^(a - 1) * (1 - D)^(b - 1).

        Результат нормируется так, чтобы max(I) = 1.
        Это не меняет результат greedy selection.
        """
        d = np.clip(
            difficulty,
            eps,
            1.0 - eps,
        )

        # Считаем в log-space для численной устойчивости.
        log_importance = (
                (a - 1.0) * np.log(d)
                + (b - 1.0) * np.log1p(-d)
        )

        log_importance -= np.max(log_importance)

        importance = np.exp(log_importance)

        if not np.all(np.isfinite(importance)):
            raise FloatingPointError(
                "Beta importance contains NaN or infinity"
            )

        return importance.astype(
            np.float32,
            copy=False,
        )

    def _allocate_class_balanced_budgets(self,
            class_counts: Sequence[int],
            subset_size: int,
    ) -> np.ndarray:
        """
        Максимально равномерно распределяет subset_size
        между классами.

        Учитывает ситуацию, когда в некотором классе объектов
        меньше, чем требуемый бюджет этого класса.
        """
        counts = np.asarray(
            class_counts,
            dtype=np.int64,
        )

        num_classes = counts.size

        if num_classes == 0:
            raise ValueError("No classes found")

        budgets = np.full(
            num_classes,
            subset_size // num_classes,
            dtype=np.int64,
        )

        remainder = subset_size % num_classes

        if remainder > 0:
            budgets[:remainder] += 1

        # Не можем выбрать больше объектов, чем есть в классе.
        budgets = np.minimum(
            budgets,
            counts,
        )

        remaining = int(
            subset_size - budgets.sum()
        )

        # Перераспределяем остаток классам,
        # в которых еще есть свободные объекты.
        while remaining > 0:
            capacity = counts - budgets

            available_classes = np.flatnonzero(
                capacity > 0
            )

            if available_classes.size == 0:
                raise ValueError(
                    "subset_size exceeds the number "
                    "of available objects"
                )

            num_to_add = min(
                remaining,
                available_classes.size,
            )

            selected_classes = available_classes[
                               :num_to_add
                               ]

            budgets[selected_classes] += 1
            remaining -= num_to_add

        return budgets

    def _boss_lazy_greedy_rectangular(self,
            coverage_features: np.ndarray,
            candidate_features: np.ndarray,
            candidate_importance: np.ndarray,
            subset_size: int,
    ) -> np.ndarray:
        """
        Lazy-greedy максимизация BOSS facility-location objective.

        В отличие от квадратной версии:

            coverage_features:
                все объекты, которые требуется покрывать;
                включая объекты, удаленные cutoff-механизмом.

            candidate_features:
                только объекты из V', которые разрешено выбирать.

        Матрица имеет форму:

            [num_coverage_objects, num_candidates].

        Parameters
        ----------
        coverage_features:
            Shape [N_coverage, feature_dim].

        candidate_features:
            Shape [N_candidates, feature_dim].

        candidate_importance:
            Beta importance разрешенных кандидатов.
            Shape [N_candidates].

        subset_size:
            Количество кандидатов, которое требуется выбрать.

        Returns
        -------
        selected_candidate_positions:
            Локальные индексы относительно candidate_features.
        """
        coverage_features = np.asarray(
            coverage_features,
            dtype=np.float32,
        )

        candidate_features = np.asarray(
            candidate_features,
            dtype=np.float32,
        )

        candidate_importance = np.asarray(
            candidate_importance,
            dtype=np.float32,
        )

        if coverage_features.ndim != 2:
            raise ValueError(
                "coverage_features must have shape [N, D]"
            )

        if candidate_features.ndim != 2:
            raise ValueError(
                "candidate_features must have shape [M, D]"
            )

        if (
                coverage_features.shape[1]
                != candidate_features.shape[1]
        ):
            raise ValueError(
                "Coverage and candidate feature dimensions differ."
            )

        n_candidates = candidate_features.shape[0]

        if candidate_importance.shape != (n_candidates,):
            raise ValueError(
                "candidate_importance must have shape "
                f"{(n_candidates,)}, got "
                f"{candidate_importance.shape}."
            )

        if not 0 <= subset_size <= n_candidates:
            raise ValueError(
                "subset_size must be between 0 and "
                "the number of candidates."
            )

        if subset_size == 0:
            return np.empty(0, dtype=np.int64)

        # ---------------------------------------------------------
        # L2-нормализация признаков
        # ---------------------------------------------------------

        coverage_norms = np.linalg.norm(
            coverage_features,
            axis=1,
            keepdims=True,
        )

        candidate_norms = np.linalg.norm(
            candidate_features,
            axis=1,
            keepdims=True,
        )

        normalized_coverage = (
                coverage_features
                / np.maximum(coverage_norms, 1e-12)
        )

        normalized_candidates = (
                candidate_features
                / np.maximum(candidate_norms, 1e-12)
        )

        # ---------------------------------------------------------
        # Cosine similarity:
        # [N_coverage, N_candidates]
        # ---------------------------------------------------------

        weighted_similarity = (
                normalized_coverage
                @ normalized_candidates.T
        ).astype(
            np.float32,
            copy=False,
        )

        # Для монотонности facility-location objective.
        np.clip(
            weighted_similarity,
            0.0,
            1.0,
            out=weighted_similarity,
        )

        # W[i, j] = Sim(i, j) * I_j.
        weighted_similarity *= candidate_importance[
                               np.newaxis,
                               :
                               ]

        # Текущее покрытие объектов из V.
        current_coverage = np.zeros(
            coverage_features.shape[0],
            dtype=np.float32,
        )

        # При S = empty marginal gain — сумма столбца.
        initial_gains = weighted_similarity.sum(
            axis=0,
            dtype=np.float64,
        )

        heap: list[tuple[float, int]] = [
            (-float(initial_gains[j]), int(j))
            for j in range(n_candidates)
        ]

        heapq.heapify(heap)

        selected: list[int] = []

        while len(selected) < subset_size:
            if not heap:
                raise RuntimeError(
                    "Lazy-greedy heap became empty before "
                    "the requested subset was selected."
                )

            _, candidate = heapq.heappop(heap)

            candidate_coverage = weighted_similarity[
                                 :,
                                 candidate,
                                 ]

            true_gain = float(
                np.maximum(
                    candidate_coverage - current_coverage,
                    0.0,
                ).sum(dtype=np.float64)
            )

            next_upper_bound = (
                -heap[0][0]
                if heap
                else -np.inf
            )

            tolerance = 1e-10 * max(
                1.0,
                abs(true_gain),
                abs(next_upper_bound)
                if np.isfinite(next_upper_bound)
                else 1.0,
            )

            if true_gain + tolerance >= next_upper_bound:
                selected.append(candidate)

                np.maximum(
                    current_coverage,
                    candidate_coverage,
                    out=current_coverage,
                )
            else:
                heapq.heappush(
                    heap,
                    (-true_gain, candidate),
                )

        return np.asarray(
            selected,
            dtype=np.int64)

    def _validate_boss_inputs(self,
            *,
            a: float,
            b: float,
            beta: float,
            subset_size: int,
            features: np.ndarray,
            difficulty: np.ndarray,
            labels: np.ndarray,
    ) -> None:
        if not np.isfinite(a) or a <= 0.0:
            raise ValueError("a must be finite and positive")

        if not np.isfinite(b) or b <= 0.0:
            raise ValueError("b must be finite and positive")

        if not np.isfinite(beta):
            raise ValueError("beta must be finite")

        if not 0.0 <= beta < 1.0:
            raise ValueError("beta must be in [0, 1)")

        if features.ndim != 2:
            raise ValueError(
                "result.features must have shape [N, D]"
            )

        if difficulty.ndim != 1:
            raise ValueError(
                "result.difficulty must have shape [N]"
            )

        if labels.ndim != 1:
            raise ValueError(
                "result.labels must have shape [N]"
            )

        n_objects = features.shape[0]

        if difficulty.size != n_objects:
            raise ValueError(
                "features and difficulty contain "
                "different numbers of objects"
            )

        if labels.size != n_objects:
            raise ValueError(
                "features and labels contain "
                "different numbers of objects"
            )

        if not 0 < subset_size <= n_objects:
            raise ValueError(
                "subset_size must be in [1, N]"
            )

        if not np.all(np.isfinite(features)):
            raise ValueError(
                "features contain NaN or infinity"
            )

        if not np.all(np.isfinite(difficulty)):
            raise ValueError(
                "difficulty contains NaN or infinity"
            )

        if np.any(difficulty < 0.0) or np.any(
                difficulty > 1.0
        ):
            raise ValueError(
                "difficulty must be normalized to [0, 1]"
            )

        remaining_count = (
                n_objects
                - int(math.floor(beta * n_objects))
        )

        if subset_size > remaining_count:
            raise ValueError(
                f"subset_size={subset_size} exceeds the "
                f"{remaining_count} objects remaining after "
                f"beta={beta} pruning."
            )