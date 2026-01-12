from CodeResearch.Helpers.Logger.BaseLogger import BaseLogger

def should_stop(scores_list, logger:BaseLogger, window=5, spearman_thr=0.95, overlap_thr=0.90, frac=0.05, largest=True) -> bool:
    if len(scores_list) < window + 1:
            return False

    reps = stability_report(scores_list[-(window + 1):], frac=frac, largest=largest)
    result = all((r["spearman"] >= spearman_thr) and (r["topk_overlap"] >= overlap_thr) for r in reps)

    if result:
        logger.logDebug('=============')
        for r in reps:
            logger.logDebug(f'Spearman: {r["spearman"]}, top-k: {r["topk_overlap"]}')
        logger.logDebug('=============')

    return result

import numpy as np

def rankdata_average(a: np.ndarray) -> np.ndarray:
    """
    Ранги с обработкой ties (средний ранг), 1..N.
    Без scipy.
    """
    a = np.asarray(a)
    n = a.size
    order = np.argsort(a, kind="mergesort")  # stable
    ranks = np.empty(n, dtype=np.float64)

    i = 0
    while i < n:
        j = i
        # ищем блок равных значений в отсортированном порядке
        while j + 1 < n and a[order[j + 1]] == a[order[i]]:
            j += 1
        # средний ранг для ties (1-indexed)
        avg_rank = 0.5 * ((i + 1) + (j + 1))
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1

    return ranks


def spearman_rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    """
    Spearman correlation = Pearson correlation of ranks.
    """
    rx = x
    ry = y

    rx = rx - rx.mean()
    ry = ry - ry.mean()

    denom = np.sqrt((rx * rx).sum()) * np.sqrt((ry * ry).sum())
    if denom == 0:
        return np.nan
    return float((rx * ry).sum() / denom)


def topk_overlap(x: np.ndarray, y: np.ndarray, frac: float = 0.05, largest: bool = True) -> float:
    """
    Доля пересечения top-k (или bottom-k) объектов.
    frac=0.05 -> top 5%.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = x.size
    k = max(1, int(round(frac * n)))

    if largest:
        ix = np.argpartition(-x, k - 1)[:k]
        iy = np.argpartition(-y, k - 1)[:k]
    else:
        ix = np.argpartition(x, k - 1)[:k]
        iy = np.argpartition(y, k - 1)[:k]

    return float(len(set(ix.tolist()).intersection(iy.tolist())) / k)


def stability_report(scores_list, frac=0.05, largest=True):
    """
    scores_list: list of numpy arrays shape (N,)
    Возвращает список метрик устойчивости между последовательными прогонами.
    """
    reps = []
    for t in range(1, len(scores_list)):
        a = scores_list[t-1]
        b = scores_list[t]

        #a = rankdata_average(a)
        #b = rankdata_average(b)

        reps.append({
            "t": t,
            "spearman": spearman_rank_corr(a, b),
            "topk_overlap": topk_overlap(a, b, frac=frac, largest=largest),
        })
    return reps


# ---- пример использования ----
# допустим, ты копишь importance после каждого прогона:
# importance_runs = [imp_run1, imp_run2, ..., imp_runR]  # каждый imp_run shape (N,)

# reps = stability_report(importance_runs, frac=0.05, largest=True)
# for r in reps:
#     print(r["t"], "spearman=", r["spearman"], "top5% overlap=", r["topk_overlap"])