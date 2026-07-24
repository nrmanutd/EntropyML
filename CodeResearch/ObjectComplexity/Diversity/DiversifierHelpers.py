import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, vmap, grad


def calculateDelta(g_list, mode):
    """
    g_list: list[Tensor[B,D]] (батчи) ИЛИ Tensor[N,D]
    Возвращает np.ndarray [N]
    """
    # 1) склейка в эпоху
    if isinstance(g_list, torch.Tensor):
        G_epoch = g_list
    else:
        G_epoch = torch.cat(g_list, dim=0)  # [N, D] на GPU/CPU

    # 2) direction-only (нормируем строки)
    G_norm = F.normalize(G_epoch, p=2, dim=1)

    # --- к среднему по эпохе (как было) ---
    if mode == "l2_relative":
        g_mean = G_epoch.mean(dim=0, keepdim=True)  # [1, D]
        diff = G_epoch - g_mean
        eps = 1e-8
        denom = g_mean.norm(p=2, dim=1).clamp_min(eps)  # [1]
        scores = diff.norm(p=2, dim=1) / denom  # [N]
        return scores.detach().cpu().numpy()

    elif mode == "cosine":
        g_mean = G_norm.mean(dim=0, keepdim=True)  # [1, D]
        mean_dir = F.normalize(g_mean, p=2, dim=1)  # [1, D]
        cos = (G_norm * mean_dir).sum(dim=1).clamp(-1.0, 1.0)  # [N]
        scores = 1.0 - cos
        return scores.detach().cpu().numpy()

    # --- отличие от предыдущего объекта ---
    elif mode == "prev_cosine":
        N = G_norm.size(0)
        if N == 0:
            return torch.empty(0).cpu().numpy()

        cos_prev = (G_norm[1:] * G_norm[:-1]).sum(dim=1).clamp(-1.0, 1.0)  # [N-1]
        scores = torch.empty(N, device=G_norm.device)
        scores[0] = 0.0
        scores[1:] = 1.0 - cos_prev
        return scores.detach().cpu().numpy()

    elif mode == "prev_l2":
        N = G_epoch.size(0)
        if N == 0:
            return torch.empty(0).cpu().numpy()

        diffs = (G_epoch[1:] - G_epoch[:-1]).norm(p=2, dim=1)  # [N-1]
        scores = torch.empty(N, device=G_epoch.device)
        scores[0] = 0.0
        scores[1:] = diffs
        return scores.detach().cpu().numpy()

    elif mode == "grad_norm":
        scores = G_epoch.norm(p=2, dim=1)
        return scores.detach().cpu().numpy()

    # --- НОВОЕ: вклад как изменение направления running mean ---
    elif mode == "running_mean_cosine":
        N = G_norm.size(0)
        if N == 0:
            return torch.empty(0).cpu().numpy()

        # cumulative mean direction mu_i
        cumsum = torch.cumsum(G_norm, dim=0)  # [N, D]
        denom = torch.arange(1, N + 1, device=G_norm.device, dtype=G_norm.dtype).unsqueeze(1)  # [N,1]
        mu = F.normalize(cumsum / denom, p=2, dim=1)  # [N, D]

        cos_prev = (mu[1:] * mu[:-1]).sum(dim=1).clamp(-1.0, 1.0)  # [N-1]
        scores = torch.empty(N, device=G_norm.device)
        scores[0] = 0.0
        scores[1:] = 1.0 - cos_prev
        return scores.detach().cpu().numpy()
    elif mode == "centered_grad_norm":
        # score_i = ||g_i - mean||_2
        g_mean = G_epoch.mean(dim=0, keepdim=True)  # [1, D]
        diff = G_epoch - g_mean  # [N, D]
        scores = diff.norm(p=2, dim=1)  # [N]
        return scores.detach().cpu().numpy()

    elif mode == "orth_to_mean_norm":
        # score_i = ||g_i - proj_{mean_dir}(g_i)||_2
        # где mean_dir = mean / ||mean||
        eps = 1e-8
        g_mean = G_epoch.mean(dim=0, keepdim=True)  # [1, D]
        mean_dir = g_mean / g_mean.norm(p=2, dim=1, keepdim=True).clamp_min(eps)  # [1, D]

        # proj scalar: [N,1] = <g_i, mean_dir>
        proj = (G_epoch * mean_dir).sum(dim=1, keepdim=True)  # [N, 1]
        residual = G_epoch - proj * mean_dir  # [N, D]
        scores = residual.norm(p=2, dim=1)  # [N]
        return scores.detach().cpu().numpy()
    else:
        raise ValueError(
            "mode must be 'cosine', 'l2_relative', 'prev_cosine', 'prev_l2', or 'running_mean_cosine'"
        )


def per_sample_grads_vmap(model, xb, yb):
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    def single_loss(params, buffers, x, y):
        logits = functional_call(model, (params, buffers), (x.unsqueeze(0),))
        return F.cross_entropy(logits, y.unsqueeze(0), reduction="mean")

    grad_fn = grad(single_loss)

    grads_pytree = vmap(grad_fn, in_dims=(None, None, 0, 0))(params, buffers, xb, yb)
    # grads_pytree: словарь параметров -> тензор с leading dim = B

    # Превратить pytree в матрицу [B, D]
    grads_list = []
    for name, g in grads_pytree.items():
        grads_list.append(g.reshape(g.size(0), -1))  # [B, ...] -> [B, d_name]
    G = torch.cat(grads_list, dim=1)  # [B, D]
    # G = F.normalize(G, dim=1)  # direction-only (опционально)
    return G

def per_sample_grads_vmap_full(model, xb, yb, names=None):
    """
    Возвращает:
      G: Tensor[B, D] — per-sample grads для ВСЕЙ модели
      names: list[str] — порядок параметров
    """
    items = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    default_names = [n for n, _ in items]
    if names is None:
        names = default_names
    else:
        assert names == default_names, "names не совпали с текущим порядком model.named_parameters()"

    params = {n: p for n, p in items}
    buffers = dict(model.named_buffers())

    def single_loss(params, buffers, x, y):
        logits = functional_call(model, (params, buffers), (x.unsqueeze(0),))
        return F.cross_entropy(logits, y.unsqueeze(0), reduction="mean")

    grad_fn = grad(single_loss)
    grads_pytree = vmap(grad_fn, in_dims=(None, None, 0, 0))(params, buffers, xb, yb)

    grads_list = []
    for n in names:
        g = grads_pytree[n]                    # [B, ...]
        grads_list.append(g.reshape(g.size(0), -1))
    G = torch.cat(grads_list, dim=1)           # [B, D]
    return G, names

def per_sample_grads_last_layer_loop(model, xb, yb):
    # forward
    feat = model.features(xb)
    feat = model.pool(feat).flatten(1) # [B, H]

    logits = model.head(feat)  # [B, C]
    losses = F.cross_entropy(logits, yb, reduction="none")

    params = [p for p in model.head.parameters() if p.requires_grad]  # только last layer

    grads = []
    for i in range(losses.size(0)):
        gi = torch.autograd.grad(losses[i], params, retain_graph=True)
        gi_vec = torch.cat([g.flatten() for g in gi])
        # gi_vec = F.normalize(gi_vec, dim=0)
        grads.append(gi_vec)
    return torch.stack(grads, dim=0)

@torch.no_grad()
def per_sample_grads_head_linear_closed_form(model, xb, yb):
    """
    Per-sample градиенты ТОЛЬКО для головы, если голова = nn.Linear(H, C).
    Возвращает матрицу G: [B, D_head], где D_head = C*H (+ C если bias).
    Никакого autograd, никакого retain_graph -> память стабильна.
    """
    head = model.head
    if not isinstance(head, torch.nn.Linear):
        raise TypeError("This closed-form works only when model.head is nn.Linear(H, C).")

    # 1) признаки (backbone) без графа
    feat = model.features(xb)
    feat = model.pool(feat).flatten(1)          # [B, H]

    # 2) logits и softmax
    logits = head(feat)                         # [B, C]
    p = torch.softmax(logits, dim=1)            # [B, C]

    # 3) one-hot и g = p - y_onehot
    C = logits.size(1)
    y_onehot = F.one_hot(yb, num_classes=C).to(p.dtype)  # [B, C]
    g = p - y_onehot                                     # [B, C]

    # 4) per-sample grad по W: outer(g_i, feat_i) => [B, C, H]
    grad_W = g.unsqueeze(2) * feat.unsqueeze(1)          # [B, C, H]
    grad_W = grad_W.reshape(grad_W.size(0), -1)          # [B, C*H]

    # 5) per-sample grad по bias: [B, C]
    if head.bias is not None:
        grad_b = g                                       # [B, C]
        G = torch.cat([grad_W, grad_b], dim=1)           # [B, C*H + C]
    else:
        G = grad_W                                       # [B, C*H]

    return G

@torch.no_grad()
def centered_grad_norm_head_linear_two_pass(model, batches, device):
    """
    batches: итератор по батчам, возвращает (xb, yb, idx) или (xb, yb)
            важно: порядок в pass1 и pass2 должен совпадать.
    Возвращает scores numpy [N] для centered_grad_norm по голове (W и bias).
    """
    model.eval()
    head = model.head
    if not isinstance(head, nn.Linear):
        raise TypeError("Works only for model.head = nn.Linear(H, C)")

    # ---------- PASS 1: считаем средний градиент по W и по bias ----------
    sum_M = None          # [C, H]
    sum_g = None          # [C] (для bias)
    n_total = 0

    for batch in batches:
        if len(batch) == 3:
            xb, yb, _idx = batch
        else:
            xb, yb = batch

        n_total += len(yb)
        continue

        xb = torch.as_tensor(xb, dtype=torch.float32, device=device)
        yb = torch.as_tensor(yb, dtype=torch.int64, device=device)

        feat = model.features(xb)
        feat = model.pool(feat).flatten(1)          # [B, H]
        logits = head(feat)                         # [B, C]

        g = torch.softmax(logits, dim=1)            # [B, C]
        g[torch.arange(g.size(0), device=device), yb] -= 1.0  # p - onehot

        # sum_M += g^T @ feat  (это сумма outer по всем объектам батча)
        # g: [B,C], feat:[B,H] -> (g.T @ feat): [C,H]
        M_batch = g.transpose(0, 1) @ feat          # [C, H]

        if sum_M is None:
            sum_M = M_batch
            sum_g = g.sum(dim=0)                    # [C]
        else:
            sum_M += M_batch
            sum_g += g.sum(dim=0)

        n_total += g.size(0)

        # освобождение временных
        del xb, yb, feat, logits, g, M_batch

    #mean_M = 0 * sum_M / max(n_total, 1)                # [C, H] #todo: remove it!
    #mean_g = 0 * sum_g / max(n_total, 1)                # [C]  (для bias)#todo: remove it!
    #norm_mean_M2 = (mean_M * mean_M).sum()          # скаляр #todo: revert it!
    #norm_mean_g2 = (mean_g * mean_g).sum()          # скаляр #todo: revert it!

    # ---------- PASS 2: считаем scores без materialize grad_W ----------
    scores = np.empty(n_total, dtype=np.float32)
    pos = 0

    # важно: batches должен быть "перезапускаемым" (или заранее сохранён список батчей)
    for batch in batches:
        if len(batch) == 3:
            xb, yb, _idx = batch
        else:
            xb, yb = batch

        xb = torch.as_tensor(xb, dtype=torch.float32, device=device)
        yb = torch.as_tensor(yb, dtype=torch.int64, device=device)

        feat = model.features(xb)
        feat = model.pool(feat).flatten(1)          # [B, H]
        logits = head(feat)                         # [B, C]

        g = torch.softmax(logits, dim=1)            # [B, C]
        g[torch.arange(g.size(0), device=device), yb] -= 1.0

        # ||outer||_F^2 = ||g||^2 * ||feat||^2
        g2 = (g * g).sum(dim=1)                     # [B]
        f2 = (feat * feat).sum(dim=1)               # [B]
        norm_outer2 = g2 * f2                       # [B]

        # <outer, mean_M> = g^T (mean_M feat)
        #Mf = feat @ mean_M.t()                      # [B, C]  (feat:[B,H], mean_M^T:[H,C]) #todo: revert it!
        #inner = (g * Mf).sum(dim=1)                 # [B] #todo: revert it!

        # centered norm по W
        #score2 = norm_outer2 + norm_mean_M2 - 2.0 * inner  # [B]
        score2 = norm_outer2 #todo: revert it!

        # если учитываем bias: grad_b = g
        # centered bias term: ||g - mean_g||^2 = ||g||^2 + ||mean_g||^2 - 2 g·mean_g
        if head.bias is not None:
            #inner_b = (g * mean_g.unsqueeze(0)).sum(dim=1)   # [B] #todo: revert it!
            #score2 = score2 + g2 + norm_mean_g2 - 2.0 * inner_b #todo: revert it!
            score2 = score2 + g2#todo: comment it!

        score = torch.sqrt(torch.clamp(score2, min=0.0))      # [B]

        bsz = score.numel()
        scores[pos:pos+bsz] = score.detach().cpu().numpy()
        pos += bsz

        #del xb, yb, feat, logits, g, g2, f2, norm_outer2, Mf, inner, score2, score #todo: revert it!
        del xb, yb, feat, logits, g, g2, f2, norm_outer2, score2, score

    return scores

def _unpack_batch(batch):
    """
    Поддерживает batch = (xb, yb) или (xb, yb, idx).
    idx игнорируется, потому что порядок scores сохраняется как порядок прохождения batches.
    """
    if len(batch) == 3:
        xb, yb, _idx = batch
    elif len(batch) == 2:
        xb, yb = batch
    else:
        raise ValueError("Batch must be (xb, yb) or (xb, yb, idx).")
    return xb, yb


def _extract_features_default(model, xb):
    """
    Под твою архитектуру:
        model.features -> model.pool -> flatten -> model.head
    """
    feat = model.features(xb)

    if hasattr(model, "pool") and model.pool is not None:
        feat = model.pool(feat)

    feat = feat.flatten(1)
    return feat


@torch.no_grad()
def grad_norm_head_linear_one_pass(
    model,
    batches,
    device,
    *,
    include_bias=True,
    feature_fn=None,
    input_dtype=torch.float32,
):
    """
    Считает per-object norm градиента loss по линейной голове model.head.

    Возвращает:
        scores: np.ndarray [N], dtype float32

    Важно:
    - scores идут строго в порядке прохождения объектов через batches.
    - Если нужен порядок датасета, dataloader должен иметь shuffle=False.
    - Если dataloader shuffle=True, возвращаемый порядок будет порядком текущего прохода.
    - Функция не использует autograd; градиент по голове считается аналитически.
    - Работает для model.head = nn.Linear(H, C).

    Для каждого объекта:
        logits = head(features)
        g = softmax(logits) - onehot(y)

        ||grad_W||^2 = ||g||^2 * ||features||^2
        ||grad_b||^2 = ||g||^2, если include_bias=True и bias существует

        score = sqrt(||grad_W||^2 + ||grad_b||^2)
    """
    model.eval()

    if not hasattr(model, "head"):
        raise AttributeError("Expected model.head to exist.")

    head = model.head
    if not isinstance(head, nn.Linear):
        raise TypeError("Works only for model.head = nn.Linear(H, C).")

    if feature_fn is None:
        feature_fn = _extract_features_default

    use_bias = include_bias and (head.bias is not None)

    scores_list = []

    for batch in batches:
        xb, yb = _unpack_batch(batch)

        xb = xb.to(device=device, dtype=input_dtype, non_blocking=True)
        yb = yb.to(device=device, dtype=torch.long, non_blocking=True)

        feat = feature_fn(model, xb)          # [B, H]
        logits = head(feat)                  # [B, C]

        probs = F.softmax(logits, dim=1)     # [B, C]
        g = probs.clone()
        g[torch.arange(g.size(0), device=device), yb] -= 1.0  # p - onehot

        g2 = (g * g).sum(dim=1)              # [B]
        f2 = (feat * feat).sum(dim=1)        # [B]

        score2 = g2 * f2                     # norm grad_W squared

        if use_bias:
            score2 = score2 + g2             # add norm grad_b squared

        scores = torch.sqrt(torch.clamp(score2, min=0.0))

        scores_list.append(scores.detach().cpu().numpy().astype(np.float32))

        del xb, yb, feat, logits, probs, g, g2, f2, score2, scores

    if len(scores_list) == 0:
        return np.empty(0, dtype=np.float32)

    return np.concatenate(scores_list, axis=0)

@torch.no_grad()
def prediction_correct_and_grad_norm_head_linear_one_pass(
    model,
    batches,
    device,
    *,
    include_bias=True,
    feature_fn=None,
    input_dtype=torch.float32,
):
    """
    За один проход по датасету вычисляет:

        h_scores:
            индикатор правильной классификации:
            1.0, если argmax(logits) == y, иначе 0.0.

        grad_norm_scores:
            per-object норму градиента cross-entropy loss
            по параметрам линейной головы model.head.

    Возвращает:
        h_scores:         np.ndarray [N], dtype float32
        grad_norm_scores: np.ndarray [N], dtype float32

    Порядок объектов соответствует порядку прохождения batches.

    Для линейной головы:
        logits = W f + b
        g = softmax(logits) - onehot(y)

        ||grad_W||^2 = ||g||^2 ||f||^2

        ||grad_b||^2 = ||g||^2,
        если include_bias=True и bias присутствует.
    """
    model.eval()

    if not hasattr(model, "head"):
        raise AttributeError("Expected model.head to exist.")

    head = model.head

    if not isinstance(head, nn.Linear):
        raise TypeError(
            "This function works only for model.head = nn.Linear(H, C)."
        )

    if feature_fn is None:
        feature_fn = _extract_features_default

    use_bias = include_bias and (head.bias is not None)

    result_list = []

    for batch in batches:
        xb, yb = _unpack_batch(batch)

        xb = xb.to(
            device=device,
            dtype=input_dtype,
            non_blocking=True,
        )
        yb = yb.to(
            device=device,
            dtype=torch.long,
            non_blocking=True,
        )

        # Один общий forward pass.
        feat = feature_fn(model, xb)          # [B, H]
        logits = head(feat)                   # [B, C]

        # ---------------------------------------------------------
        # 1. Correctness / h
        # ---------------------------------------------------------
        pred = torch.argmax(logits, dim=1)

        h_scores = (pred == yb).to(torch.float32)  # [B]

        # ---------------------------------------------------------
        # 2. GradNorm по линейной голове
        # ---------------------------------------------------------
        probs = F.softmax(logits, dim=1)      # [B, C]

        # Для g = p - onehot(y):
        #
        # ||g||^2
        # = sum_c p_c^2 - 2 p_y + 1
        #
        # Это позволяет не создавать clone(probs) и не изменять
        # элемент целевого класса in-place.
        p_target = probs.gather(
            dim=1,
            index=yb.unsqueeze(1),
        ).squeeze(1)                          # [B]

        g2 = probs.square().sum(dim=1) - 2.0 * p_target + 1.0
        g2 = torch.clamp(g2, min=0.0)         # защита от round-off

        f2 = feat.square().sum(dim=1)          # [B]

        if use_bias:
            # g2 * f2 + g2 = g2 * (f2 + 1)
            grad_score2 = g2 * (f2 + 1.0)
        else:
            grad_score2 = g2 * f2

        grad_norm_scores = torch.sqrt(
            torch.clamp(grad_score2, min=0.0)
        )

        # Один перенос GPU -> CPU вместо двух отдельных переносов.
        batch_result = torch.stack(
            [h_scores, grad_norm_scores],
            dim=1,
        )

        result_list.append(
            batch_result.cpu().numpy().astype(
                np.float32,
                copy=False,
            )
        )

        del (
            xb,
            yb,
            feat,
            logits,
            pred,
            h_scores,
            probs,
            p_target,
            g2,
            f2,
            grad_score2,
            grad_norm_scores,
            batch_result,
        )

    if len(result_list) == 0:
        empty = np.empty(0, dtype=np.float32)
        return empty, empty.copy()

    result = np.concatenate(result_list, axis=0)

    h_scores = result[:, 0]
    grad_norm_scores = result[:, 1]

    return h_scores, grad_norm_scores

@torch.no_grad()
def prediction_correct_head_linear_one_pass(
    model,
    batches,
    device,
    *,
    include_bias=True,
    feature_fn=None,
    input_dtype=torch.float32,
):
    """
    Возвращает per-object индикатор правильной классификации по линейной голове model.head.

    Возвращает:
        scores: np.ndarray [N], dtype float32

    Где:
        1.0 — целевой класс угадан:
              argmax_c p(c | x) == y

        0.0 — целевой класс не угадан:
              argmax_c p(c | x) != y

    Важно:
    - scores идут строго в порядке прохождения объектов через batches.
    - Если нужен порядок датасета, dataloader должен иметь shuffle=False.
    - Если dataloader shuffle=True, возвращаемый порядок будет порядком текущего прохода.
    - Функция не использует autograd.
    - Работает для model.head = nn.Linear(H, C).

    Примечание:
    - include_bias оставлен в сигнатуре для совместимости, но здесь не используется.
    - Для argmax softmax считать не нужно:
      argmax(softmax(logits)) == argmax(logits).
    """
    model.eval()

    if not hasattr(model, "head"):
        raise AttributeError("Expected model.head to exist.")

    head = model.head
    if not isinstance(head, nn.Linear):
        raise TypeError("Works only for model.head = nn.Linear(H, C).")

    if feature_fn is None:
        feature_fn = _extract_features_default

    scores_list = []

    for batch in batches:
        xb, yb = _unpack_batch(batch)

        xb = xb.to(device=device, dtype=input_dtype, non_blocking=True)
        yb = yb.to(device=device, dtype=torch.long, non_blocking=True)

        feat = feature_fn(model, xb)        # [B, H]
        logits = head(feat)                # [B, C]

        pred = torch.argmax(logits, dim=1) # [B]

        # 1.0 если угадали, 0.0 если ошиблись
        scores = (pred == yb).to(torch.float32)

        scores_list.append(scores.detach().cpu().numpy().astype(np.float32))

        del xb, yb, feat, logits, pred, scores

    if len(scores_list) == 0:
        return np.empty(0, dtype=np.float32)

    return np.concatenate(scores_list, axis=0)

@torch.no_grad()
def el2n_scores(
    model,
    batches,
    device,
    *,
    num_classes=None,
    dtype=torch.float32,
):
    """
    Считает EL2N score для каждого объекта:

        EL2N_i = ||softmax(model(x_i)) - onehot(y_i)||_2

    Важно:
    - scores возвращаются ровно в порядке прохождения объектов через batches.
    - Если return_indices=True и batch содержит idx, дополнительно возвращаются idx.
    - В отличие от gradient norm, здесь используется обычный model(xb),
      а не model.features + model.head.

    Parameters
    ----------
    model : torch.nn.Module
    batches : iterable
        Возвращает (xb, yb) или (xb, yb, idx).
    device : str or torch.device
    num_classes : int or None
        Если None, берется из logits.shape[1].
    return_indices : bool
    dtype : torch dtype

    Returns
    -------
    scores : np.ndarray [N]
    indices : np.ndarray [N], optional
    """
    model.eval()

    scores_list = []
    idx_list = []

    for batch in batches:
        xb, yb = _unpack_batch(batch)

        xb = xb.to(device=device, dtype=dtype, non_blocking=True)
        yb = yb.to(device=device, dtype=torch.long, non_blocking=True)

        logits = model(xb)  # [B, C]

        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        probs = F.softmax(logits, dim=1)

        C = probs.size(1) if num_classes is None else num_classes
        y_onehot = F.one_hot(yb, num_classes=C).to(dtype=probs.dtype)

        score = torch.norm(probs - y_onehot, p=2, dim=1)

        scores_list.append(score.detach().cpu().numpy().astype(np.float32))

    scores = np.concatenate(scores_list, axis=0) if scores_list else np.empty(0, dtype=np.float32)

    return scores


@torch.no_grad()
def grad_norm_times_positive_cos_to_train_mean_grad_head_linear_two_pass(
    model,
    train_batches,
    batches,
    device,
    eps: float = 1e-12,
):
    """
    Считает для каждого объекта из batches score:

        score_i = ||grad_i|| * max(cos(grad_i, mean_grad_train), 0)

    где:
        grad_i          — per-object градиент loss по линейной голове model.head
                          для объекта из batches;
        mean_grad_train — средний градиент по объектам из train_batches,
                          то есть по той совокупности, на которой модель уже обучалась.

    Эквивалентно:

        score_i = max(<grad_i, mean_grad_train> / ||mean_grad_train||, 0)

    Возвращает:
        scores: np.ndarray [N], dtype float32

    Порядок scores соответствует порядку объектов в batches.
    """

    model.eval()

    head = model.head
    if not isinstance(head, nn.Linear):
        raise TypeError("Works only for model.head = nn.Linear(H, C)")

    def _unpack_batch(batch):
        if len(batch) == 3:
            xb, yb, _idx = batch
        else:
            xb, yb = batch
        return xb, yb

    def _extract_features(model, xb):
        feat = model.features(xb)

        if hasattr(model, "pool") and model.pool is not None:
            feat = model.pool(feat)

        feat = feat.flatten(1)
        return feat

    # ------------------------------------------------------------------
    # PASS 1: mean_grad_train по train_batches
    # ------------------------------------------------------------------
    sum_M = None   # [C, H] = sum_i outer(g_i, feat_i)
    sum_g = None   # [C]    = sum_i g_i для bias
    n_train = 0

    for batch in train_batches:
        xb, yb = _unpack_batch(batch)

        xb = torch.as_tensor(xb, dtype=torch.float32, device=device)
        yb = torch.as_tensor(yb, dtype=torch.int64, device=device)

        feat = _extract_features(model, xb)          # [B, H]
        logits = head(feat)                          # [B, C]

        g = torch.softmax(logits, dim=1)             # [B, C]
        g[torch.arange(g.size(0), device=device), yb] -= 1.0

        M_batch = g.transpose(0, 1) @ feat           # [C, H]
        g_batch = g.sum(dim=0)                       # [C]

        if sum_M is None:
            sum_M = M_batch
            sum_g = g_batch
        else:
            sum_M += M_batch
            sum_g += g_batch

        n_train += g.size(0)

        del xb, yb, feat, logits, g, M_batch, g_batch

    if n_train == 0:
        raise ValueError("train_batches is empty: cannot compute mean train gradient.")

    mean_M = sum_M / n_train                         # [C, H]
    mean_g = sum_g / n_train                         # [C]

    norm_mean2 = (mean_M * mean_M).sum()

    if head.bias is not None:
        norm_mean2 = norm_mean2 + (mean_g * mean_g).sum()

    norm_mean = torch.sqrt(torch.clamp(norm_mean2, min=0.0))
    norm_mean = torch.clamp(norm_mean, min=eps)

    # ------------------------------------------------------------------
    # PASS 2: score_i для объектов из batches
    # ------------------------------------------------------------------
    scores_list = []

    for batch in batches:
        xb, yb = _unpack_batch(batch)

        xb = torch.as_tensor(xb, dtype=torch.float32, device=device)
        yb = torch.as_tensor(yb, dtype=torch.int64, device=device)

        feat = _extract_features(model, xb)          # [B, H]
        logits = head(feat)                          # [B, C]

        g = torch.softmax(logits, dim=1)             # [B, C]
        g[torch.arange(g.size(0), device=device), yb] -= 1.0

        # <grad_W(i), mean_M>
        Mf = feat @ mean_M.t()                       # [B, C]
        inner = (g * Mf).sum(dim=1)                  # [B]

        # bias term: <grad_b(i), mean_g>
        if head.bias is not None:
            inner_b = (g * mean_g.unsqueeze(0)).sum(dim=1)
            inner = inner + inner_b

        # score_i = ||grad_i|| * max(cos(grad_i, mean_grad_train), 0)
        #         = max(<grad_i, mean_grad_train> / ||mean_grad_train||, 0)
        score = torch.clamp(inner / norm_mean, min=0.0)  # [B]

        scores_list.append(score.detach().cpu().numpy().astype(np.float32))

        del xb, yb, feat, logits, g, Mf, inner, score

    if len(scores_list) == 0:
        return np.empty(0, dtype=np.float32)

    return np.concatenate(scores_list, axis=0).astype(np.float32)


@torch.no_grad()
def chg_shapley_to_train_mean_grad_head_linear_two_pass(
    model,
    train_batches,
    batches,
    device,
):
    """
    Рассчитывает CHG Shapley score для объектов-кандидатов из batches.

    Используемая адаптация формулы из статьи:

        x_i = h_i * grad_i,
        h_i = loss_i,

        alpha = mean_{i in train_batches}(x_i).

    Здесь:
        grad_i — per-object градиент CrossEntropy loss
                 по параметрам линейной головы model.head;

        h_i    — индивидуальный CrossEntropy loss объекта,
                 не участвующий в backpropagation;

        N      — множество объектов-кандидатов из batches;

        alpha  — средний CHG-градиент уже выбранного
                 обучающего множества train_batches.

    Utility кандидатов:

        U(S) = ||alpha||^2
               - ||mean_{i in S}(x_i) - alpha||^2.

    Возвращает:
        scores: np.ndarray [N], dtype float32.

    Значения могут быть как положительными, так и отрицательными.
    Для отбора наиболее ценных объектов нужно сортировать scores
    по убыванию.

    Важно:
        1. batches проходит дважды и должен быть повторно итерируемым,
           например DataLoader, а не одноразовым generator.
        2. Для соответствия scores порядку датасета используйте
           shuffle=False.
        3. Между двумя проходами не должны применяться случайные
           аугментации, меняющие представление объектов.
    """

    model.eval()

    head = model.head
    if not isinstance(head, nn.Linear):
        raise TypeError(
            "Works only for model.head = nn.Linear(H, C)"
        )

    # ------------------------------------------------------------------
    # Вспомогательные функции
    # ------------------------------------------------------------------

    def _unpack_batch(batch):
        if len(batch) == 3:
            xb, yb, _idx = batch
        else:
            xb, yb = batch

        return xb, yb

    def _extract_features(model, xb):
        """
        Извлечение признаков для архитектуры:

            model.features
            -> model.pool, если он существует
            -> flatten
            -> model.head
        """
        feat = model.features(xb)

        if hasattr(model, "pool") and model.pool is not None:
            feat = model.pool(feat)

        return feat.flatten(1)

    def _get_chg_components(xb, yb):
        """
        Возвращает:

            feat: [B, H]

            weighted_g: [B, C]

        Для CrossEntropy:

            g_i = softmax(logits_i) - one_hot(y_i)

        является градиентом loss по logits.

        CHG использует:

            x_i = h_i * grad_i,

        поэтому weighted_g соответствует h_i * g_i.
        """

        feat = _extract_features(model, xb)          # [B, H]
        logits = head(feat)                          # [B, C]

        log_probs = torch.log_softmax(logits, dim=1)

        row_idx = torch.arange(
            logits.size(0),
            device=logits.device,
        )

        # h_i = индивидуальный CrossEntropy loss.
        # Функция находится под @torch.no_grad(), поэтому h_i
        # автоматически не участвует в дифференцировании.
        hardness = -log_probs[row_idx, yb]           # [B]

        # g_i = d loss_i / d logits_i
        g = log_probs.exp()                          # softmax
        g[row_idx, yb] -= 1.0                       # [B, C]

        # Для каждого объекта это h_i * градиент по logits.
        weighted_g = g * hardness.unsqueeze(1)       # [B, C]

        return feat, weighted_g

    def _per_object_x_norm2(feat, weighted_g):
        """
        x_i состоит из градиента по W и, при наличии, по bias:

            x_W(i) = weighted_g_i outer feat_i
            x_b(i) = weighted_g_i

        Поэтому:

            ||x_W(i)||_F^2
                = ||weighted_g_i||^2 * ||feat_i||^2.
        """

        g_norm2 = (weighted_g * weighted_g).sum(dim=1)
        feat_norm2 = (feat * feat).sum(dim=1)

        norm2 = g_norm2 * feat_norm2

        if head.bias is not None:
            norm2 = norm2 + g_norm2

        return norm2

    def _chg_coefficients(n):
        """
        Коэффициенты формулы (4).

        Итоговая формула представляется как:

            phi_j =
                c_x2       * ||x_j||^2
              + c_Tx       * <T, x_j>
              + c_T2       * ||T||^2
              + c_Q        * Q
              + c_xalpha   * <x_j, alpha>
              + c_Talpha   * <T, alpha>,

        где:

            T = sum_i x_i,
            Q = sum_i ||x_i||^2.

        Формула статьи содержит (n - 2) в знаменателях.
        Для n=1 и n=2 ниже используются непосредственно
        вычисленные точные коэффициенты.
        """

        if n < 1:
            raise ValueError("n must be positive")

        if n == 1:
            # phi_j = U({j}) = 2<x_j, alpha> - ||x_j||^2
            return (
                -1.0,   # c_x2
                0.0,    # c_Tx
                0.0,    # c_T2
                0.0,    # c_Q
                2.0,    # c_xalpha
                0.0,    # c_Talpha
            )

        if n == 2:
            return (
                -0.75,  # c_x2
                -0.25,  # c_Tx
                0.0,    # c_T2
                0.375,  # c_Q
                2.0,    # c_xalpha
                -0.5,   # c_Talpha
            )

        # Гармонические суммы вычисляются в float64.
        k = np.arange(1, n + 1, dtype=np.float64)

        H1 = np.sum(1.0 / k, dtype=np.float64)
        H2 = np.sum(1.0 / (k * k), dtype=np.float64)

        inv_n = 1.0 / n

        # Повторяющееся выражение из формулы (4):
        #
        # 2 * sum(1/k) - 2 * sum(1/k^2) - 1 + 1/n
        B = 2.0 * H1 - 2.0 * H2 - 1.0 + inv_n

        denominator_1 = n * (n - 1)
        denominator_2 = n * (n - 1) * (n - 2)

        c_x2 = (
            -H2 / n
            + (
                2.0 * H1
                - 3.0 * H2
                + inv_n
            ) / denominator_1
            + 2.0 * B / denominator_2
        )

        c_Tx = (
            -2.0
            * (
                H1
                - H2
                - inv_n
                + inv_n * inv_n
            )
            / ((n - 1) * (n - 2))
        )

        c_T2 = B / denominator_2

        c_Q = (
            (H2 - inv_n) / denominator_1
            - B / denominator_2
        )

        c_xalpha = (
            2.0
            * (H1 - inv_n)
            / (n - 1)
        )

        c_Talpha = (
            -2.0
            * (H1 - 1.0)
            / denominator_1
        )

        return (
            float(c_x2),
            float(c_Tx),
            float(c_T2),
            float(c_Q),
            float(c_xalpha),
            float(c_Talpha),
        )

    # ==================================================================
    # PASS A: вычисляем alpha по уже выбранному train_batches
    # ==================================================================

    alpha_M_sum = None   # sum_i h_i * outer(g_i, feat_i), [C, H]
    alpha_g_sum = None   # sum_i h_i * g_i, [C]
    n_train = 0

    for batch in train_batches:
        xb, yb = _unpack_batch(batch)

        xb = torch.as_tensor(
            xb,
            dtype=torch.float32,
            device=device,
        )
        yb = torch.as_tensor(
            yb,
            dtype=torch.int64,
            device=device,
        ).view(-1)

        feat, weighted_g = _get_chg_components(xb, yb)

        M_batch = weighted_g.transpose(0, 1) @ feat
        g_batch = weighted_g.sum(dim=0)

        if alpha_M_sum is None:
            alpha_M_sum = M_batch
            alpha_g_sum = g_batch
        else:
            alpha_M_sum += M_batch
            alpha_g_sum += g_batch

        n_train += weighted_g.size(0)

        del (
            xb,
            yb,
            feat,
            weighted_g,
            M_batch,
            g_batch,
        )

    if n_train == 0:
        raise ValueError(
            "train_batches is empty: cannot compute alpha."
        )

    alpha_M = alpha_M_sum / n_train                  # [C, H]
    alpha_g = alpha_g_sum / n_train                  # [C]

    # ==================================================================
    # PASS 1 по кандидатам:
    #
    # T = sum_i x_i
    # Q = sum_i ||x_i||^2
    # ==================================================================

    T_M = None                                       # [C, H]
    T_g = None                                       # [C]

    # Скалярные агрегаты сохраняем в float64.
    Q = torch.zeros(
        (),
        dtype=torch.float64,
        device=device,
    )

    n_candidates = 0

    for batch in batches:
        xb, yb = _unpack_batch(batch)

        xb = torch.as_tensor(
            xb,
            dtype=torch.float32,
            device=device,
        )
        yb = torch.as_tensor(
            yb,
            dtype=torch.int64,
            device=device,
        ).view(-1)

        feat, weighted_g = _get_chg_components(xb, yb)

        M_batch = weighted_g.transpose(0, 1) @ feat
        g_batch = weighted_g.sum(dim=0)

        if T_M is None:
            T_M = M_batch
            T_g = g_batch
        else:
            T_M += M_batch
            T_g += g_batch

        x_norm2 = _per_object_x_norm2(
            feat,
            weighted_g,
        )

        Q += x_norm2.double().sum()
        n_candidates += weighted_g.size(0)

        del (
            xb,
            yb,
            feat,
            weighted_g,
            M_batch,
            g_batch,
            x_norm2,
        )

    if n_candidates == 0:
        return np.empty(0, dtype=np.float32)

    # ||T||^2
    T_norm2 = (
        T_M.double() * T_M.double()
    ).sum()

    # <T, alpha>
    T_alpha = (
        T_M.double() * alpha_M.double()
    ).sum()

    if head.bias is not None:
        T_norm2 = T_norm2 + (
            T_g.double() * T_g.double()
        ).sum()

        T_alpha = T_alpha + (
            T_g.double() * alpha_g.double()
        ).sum()

    (
        c_x2,
        c_Tx,
        c_T2,
        c_Q,
        c_xalpha,
        c_Talpha,
    ) = _chg_coefficients(n_candidates)

    # Эти три слагаемых одинаковы для всех j.
    # Они не влияют на ранжирование, но включаются, чтобы вернуть
    # полный Shapley value из теоремы.
    common_offset = (
        c_T2 * T_norm2
        + c_Q * Q
        + c_Talpha * T_alpha
    )

    # ==================================================================
    # PASS 2 по кандидатам:
    #
    # Для каждого j нужны:
    #   ||x_j||^2
    #   <T, x_j>
    #   <x_j, alpha>
    # ==================================================================

    scores_list = []
    n_scored = 0

    for batch in batches:
        xb, yb = _unpack_batch(batch)

        xb = torch.as_tensor(
            xb,
            dtype=torch.float32,
            device=device,
        )
        yb = torch.as_tensor(
            yb,
            dtype=torch.int64,
            device=device,
        ).view(-1)

        feat, weighted_g = _get_chg_components(xb, yb)

        # --------------------------------------------------------------
        # ||x_j||^2
        # --------------------------------------------------------------
        x_norm2 = _per_object_x_norm2(
            feat,
            weighted_g,
        ).double()

        # --------------------------------------------------------------
        # <T, x_j>
        #
        # Для весов головы:
        #
        # <outer(weighted_g_j, feat_j), T_M>
        #   = <weighted_g_j, T_M @ feat_j>
        # --------------------------------------------------------------
        T_projection = feat @ T_M.t()                # [B, C]
        T_x = (
            weighted_g * T_projection
        ).sum(dim=1)

        # --------------------------------------------------------------
        # <x_j, alpha>
        # --------------------------------------------------------------
        alpha_projection = feat @ alpha_M.t()        # [B, C]
        x_alpha = (
            weighted_g * alpha_projection
        ).sum(dim=1)

        if head.bias is not None:
            T_x = T_x + (
                weighted_g * T_g.unsqueeze(0)
            ).sum(dim=1)

            x_alpha = x_alpha + (
                weighted_g * alpha_g.unsqueeze(0)
            ).sum(dim=1)

        # --------------------------------------------------------------
        # Полная формула CHG Shapley
        # --------------------------------------------------------------
        score = (
            c_x2 * x_norm2
            + c_Tx * T_x.double()
            + c_xalpha * x_alpha.double()
            + common_offset
        )

        scores_list.append(
            score.cpu().numpy().astype(np.float32)
        )

        n_scored += score.numel()

        del (
            xb,
            yb,
            feat,
            weighted_g,
            x_norm2,
            T_projection,
            T_x,
            alpha_projection,
            x_alpha,
            score,
        )

    if n_scored != n_candidates:
        raise RuntimeError(
            "The second pass over batches returned a different "
            "number of objects. batches must be re-iterable; "
            "do not pass a one-shot generator."
        )

    return np.concatenate(
        scores_list,
        axis=0,
    ).astype(np.float32)

@torch.no_grad()
def cosine_to_mean_grad_head_linear_two_pass(model, batches, device, eps: float = 1e-12):
    """
    batches: итератор по батчам, возвращает (xb, yb, idx) или (xb, yb)
            важно: порядок в pass1 и pass2 должен совпадать.
    Возвращает scores numpy [N] = cos(grad_i, mean_grad) по голове (W и bias),
    где grad_i — градиент по (W,bias) для одного объекта, mean_grad — средний по датасету.

    Реализация без materialize per-sample grad_W:
      grad_W(i) = outer(g_i, feat_i), где g_i = p_i - onehot(y_i)
      grad_b(i) = g_i
    """
    model.eval()
    head = model.head
    if not isinstance(head, nn.Linear):
        raise TypeError("Works only for model.head = nn.Linear(H, C)")

    # ---------- PASS 1: считаем средний градиент по W и по bias ----------
    sum_M = None          # [C, H]  = sum_i (g_i^T @ feat_i)
    sum_g = None          # [C]     = sum_i g_i (для bias)
    n_total = 0

    for batch in batches:
        if len(batch) == 3:
            xb, yb, _idx = batch
        else:
            xb, yb = batch

        xb = torch.as_tensor(xb, dtype=torch.float32, device=device)
        yb = torch.as_tensor(yb, dtype=torch.int64, device=device)

        feat = model.features(xb)
        feat = model.pool(feat).flatten(1)          # [B, H]
        logits = head(feat)                         # [B, C]

        g = torch.softmax(logits, dim=1)            # [B, C]
        g[torch.arange(g.size(0), device=device), yb] -= 1.0  # p - onehot

        M_batch = g.transpose(0, 1) @ feat          # [C, H]

        if sum_M is None:
            sum_M = M_batch
            sum_g = g.sum(dim=0)                    # [C]
        else:
            sum_M += M_batch
            sum_g += g.sum(dim=0)

        n_total += g.size(0)

        del xb, yb, feat, logits, g, M_batch

    mean_M = sum_M / max(n_total, 1)                # [C, H]
    mean_g = sum_g / max(n_total, 1)                # [C]

    # ||mean_grad|| for W and bias
    norm_mean2 = (mean_M * mean_M).sum()
    if head.bias is not None:
        norm_mean2 = norm_mean2 + (mean_g * mean_g).sum()
    norm_mean = torch.sqrt(torch.clamp(norm_mean2, min=0.0))
    norm_mean = torch.clamp(norm_mean, min=eps)

    # ---------- PASS 2: считаем cos(grad_i, mean_grad) ----------
    scores = np.empty(n_total, dtype=np.float32)
    pos = 0

    # важно: batches должен быть "перезапускаемым" (или заранее сохранён список батчей)
    for batch in batches:
        if len(batch) == 3:
            xb, yb, _idx = batch
        else:
            xb, yb = batch

        xb = torch.as_tensor(xb, dtype=torch.float32, device=device)
        yb = torch.as_tensor(yb, dtype=torch.int64, device=device)

        feat = model.features(xb)
        feat = model.pool(feat).flatten(1)          # [B, H]
        logits = head(feat)                         # [B, C]

        g = torch.softmax(logits, dim=1)            # [B, C]
        g[torch.arange(g.size(0), device=device), yb] -= 1.0

        # <grad_W(i), mean_M> = <outer(g_i, feat_i), mean_M> = g_i^T (mean_M feat_i)
        Mf = feat @ mean_M.t()                      # [B, C]
        inner_W = (g * Mf).sum(dim=1)               # [B]
        inner = inner_W

        # bias term: <grad_b(i), mean_g> = g_i · mean_g
        if head.bias is not None:
            inner_b = (g * mean_g.unsqueeze(0)).sum(dim=1)  # [B]
            inner = inner + inner_b

        # ||grad_i||^2 = ||outer||_F^2 + (||g_i||^2 if bias)
        g2 = (g * g).sum(dim=1)                     # [B]
        f2 = (feat * feat).sum(dim=1)               # [B]
        norm_outer2 = g2 * f2                       # [B]
        norm_i2 = norm_outer2
        if head.bias is not None:
            norm_i2 = norm_i2 + g2

        norm_i = torch.sqrt(torch.clamp(norm_i2, min=0.0))
        norm_i = torch.clamp(norm_i, min=eps)

        cos = inner / (norm_i * norm_mean)          # [B]
        cos = torch.clamp(cos, min=-1.0, max=1.0)

        bsz = cos.numel()
        scores[pos:pos+bsz] = cos.detach().cpu().numpy()
        pos += bsz

        del xb, yb, feat, logits, g, Mf, inner_W, inner, g2, f2, norm_outer2, norm_i2, norm_i, cos

    return scores

@torch.no_grad()
def centered_grad_norm_head_linear_two_pass_entropy_loss(model, batches, device):
    """
    batches: итератор по батчам, возвращает (xb, yb, idx) или (xb, yb)
            важно: порядок в pass1 и pass2 должен совпадать.

    Возвращает:
        scores numpy [N]

    Смысл score:
        score_i = ||grad_i|| * max_c softmax(logits_i)[c]

    где:
        ||grad_i|| — норма градиента CE loss по линейной голове model.head
                    для истинной метки y_i;

        max_c softmax(logits_i)[c] — вероятность лучшего класса по версии модели,
                                     то есть confidence самой модели.

    Важно:
    - batches должен быть перезапускаемым, потому что функция проходит по нему два раза.
    - Фактически centered-часть в текущей реализации отключена:
      mean_M / mean_g не считаются, используется обычная per-object grad norm.
    """
    model.eval()

    head = model.head
    if not isinstance(head, nn.Linear):
        raise TypeError("Works only for model.head = nn.Linear(H, C)")

    # ---------- PASS 1: считаем только общее число объектов ----------
    n_total = 0

    for batch in batches:
        if len(batch) == 3:
            xb, yb, _idx = batch
        else:
            xb, yb = batch

        n_total += len(yb)

    # ---------- PASS 2: считаем grad norm * model confidence ----------
    scores = np.empty(n_total, dtype=np.float32)
    confidence_all = np.empty(n_total, dtype=np.float32)

    pos = 0

    for batch in batches:
        if len(batch) == 3:
            xb, yb, _idx = batch
        else:
            xb, yb = batch

        xb = torch.as_tensor(xb, dtype=torch.float32, device=device)
        yb = torch.as_tensor(yb, dtype=torch.int64, device=device)

        feat = model.features(xb)
        feat = model.pool(feat).flatten(1)          # [B, H]
        logits = head(feat)                         # [B, C]

        probs = torch.softmax(logits, dim=1)        # [B, C]

        # Вероятность лучшего класса по версии модели:
        # max_c p(c | x)
        best_prob = probs.max(dim=1).values         # [B]

        # Градиент CE loss по logits:
        # g = p - onehot(y)
        g = probs.clone()
        g[torch.arange(g.size(0), device=device), yb] -= 1.0

        # ||grad_W||_F^2 = ||g||^2 * ||feat||^2
        g2 = (g * g).sum(dim=1)                     # [B]
        f2 = (feat * feat).sum(dim=1)               # [B]
        score2 = g2 * f2                            # [B]

        # Если есть bias: ||grad_b||^2 = ||g||^2
        if head.bias is not None:
            score2 = score2 + g2

        grad_norm = torch.sqrt(torch.clamp(score2, min=0.0))  # [B]

        bsz = grad_norm.numel()

        scores[pos:pos + bsz] = grad_norm.detach().cpu().numpy().astype(np.float32)
        confidence_all[pos:pos + bsz] = best_prob.detach().cpu().numpy().astype(np.float32)

        pos += bsz

        del xb, yb, feat, logits, probs, best_prob, g
        del g2, f2, score2, grad_norm

    return scores * confidence_all

def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    B, C = logits.shape

    log_p = F.log_softmax(logits, dim=1)
    p = log_p.exp()
    H = - (p * log_p).sum(dim=1)

    Hn = H / torch.log(torch.tensor(float(C), device=logits.device))

    return Hn.clamp(0.0, 1.0)

def easiness_from_logits(logits: torch.Tensor, yb) -> torch.Tensor:
    p_true = torch.softmax(logits, dim=1)
    loss_i = p_true[torch.arange(p_true.size(0), device=logits.device), yb]

    return loss_i

@torch.no_grad()
def snapshot_all_named_params(model: torch.nn.Module):
    """
    Возвращает:
      w: Tensor[D] — все trainable параметры в одном векторе
      names: list[str] — порядок параметров (важно для согласования с градиентами)
    """
    items = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    names = [n for n, _ in items]
    w = torch.cat([p.detach().flatten().clone() for _, p in items], dim=0)
    return w, names

@torch.no_grad()
def direction_from_two_models(model_before, model_after, eps=1e-12):
    """
    m = normalize(theta_after - theta_before)
    Возвращает:
      m: Tensor[D] (unit direction)
      names: list[str] (порядок параметров)
    """
    w0, names0 = snapshot_all_named_params(model_before)
    w1, names1 = snapshot_all_named_params(model_after)
    assert names0 == names1, "Порядок параметров изменился — так быть не должно."
    delta = w1 - w0
    m = F.normalize(delta, dim=0, eps=eps)
    return m, names0

@torch.no_grad()
def direction_from_two_models_after_snapshots(w0, names0, w1, names1, eps=1e-12):

    assert names0 == names1, "Порядок параметров изменился — так быть не должно."
    delta = w1 - w0
    m = F.normalize(delta, dim=0, eps=eps)
    return m, names0

def proj_and_orth_norm(G, m, eps=1e-12):
    """
    G: [B, D] per-sample gradients
    m: [D] unit direction (delta-weights normalized)
    Возвращает:
      proj: [B]  <g_i, m>
      orth: [B]  ||g_i - proj_i * m|| = sqrt(||g||^2 - proj^2)
    """
    m = m.to(G.device)
    proj = (G * m.unsqueeze(0)).sum(dim=1)                  # [B]
    norm2 = (G * G).sum(dim=1)                              # [B]
    orth2 = (norm2 - proj * proj).clamp_min(0.0)
    orth = torch.sqrt(orth2 + eps)
    return proj, orth