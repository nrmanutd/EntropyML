import torch
import numpy as np
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

    mean_M = sum_M / max(n_total, 1)                # [C, H]
    mean_g = sum_g / max(n_total, 1)                # [C]  (для bias)
    norm_mean_M2 = (mean_M * mean_M).sum()          # скаляр
    norm_mean_g2 = (mean_g * mean_g).sum()          # скаляр

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
        Mf = feat @ mean_M.t()                      # [B, C]  (feat:[B,H], mean_M^T:[H,C])
        inner = (g * Mf).sum(dim=1)                 # [B]

        # centered norm по W
        score2 = norm_outer2 + norm_mean_M2 - 2.0 * inner  # [B]

        # если учитываем bias: grad_b = g
        # centered bias term: ||g - mean_g||^2 = ||g||^2 + ||mean_g||^2 - 2 g·mean_g
        if head.bias is not None:
            inner_b = (g * mean_g.unsqueeze(0)).sum(dim=1)   # [B]
            score2 = score2 + g2 + norm_mean_g2 - 2.0 * inner_b

        score = torch.sqrt(torch.clamp(score2, min=0.0))      # [B]

        bsz = score.numel()
        scores[pos:pos+bsz] = score.detach().cpu().numpy()
        pos += bsz

        del xb, yb, feat, logits, g, g2, f2, norm_outer2, Mf, inner, score2, score

    return scores

@torch.no_grad()
def centered_grad_norm_head_linear_two_pass_entropy_loss(model, batches, device):
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

    mean_M = sum_M / max(n_total, 1)                # [C, H]
    mean_g = sum_g / max(n_total, 1)                # [C]  (для bias)
    norm_mean_M2 = (mean_M * mean_M).sum()          # скаляр
    norm_mean_g2 = (mean_g * mean_g).sum()          # скаляр

    # ---------- PASS 2: считаем scores без materialize grad_W ----------
    scores = np.empty(n_total, dtype=np.float32)
    pos = 0
    loss_all = np.empty(n_total, dtype=np.float32)

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

        pred = torch.argmax(logits, dim=1)
        loss_i = F.cross_entropy(logits, yb, reduction='none')

        g = torch.softmax(logits, dim=1)            # [B, C]
        g[torch.arange(g.size(0), device=device), yb] -= 1.0

        # ||outer||_F^2 = ||g||^2 * ||feat||^2
        g2 = (g * g).sum(dim=1)                     # [B]
        f2 = (feat * feat).sum(dim=1)               # [B]
        norm_outer2 = g2 * f2                       # [B]

        # <outer, mean_M> = g^T (mean_M feat)
        Mf = feat @ mean_M.t()                      # [B, C]  (feat:[B,H], mean_M^T:[H,C])
        inner = (g * Mf).sum(dim=1)                 # [B]

        # centered norm по W
        score2 = norm_outer2 + norm_mean_M2 - 2.0 * inner  # [B]

        # если учитываем bias: grad_b = g
        # centered bias term: ||g - mean_g||^2 = ||g||^2 + ||mean_g||^2 - 2 g·mean_g
        if head.bias is not None:
            inner_b = (g * mean_g.unsqueeze(0)).sum(dim=1)   # [B]
            score2 = score2 + g2 + norm_mean_g2 - 2.0 * inner_b

        score = torch.sqrt(torch.clamp(score2, min=0.0))      # [B]

        bsz = score.numel()
        scores[pos:pos+bsz] = score.detach().cpu().numpy()
        loss_all[pos:pos+bsz] = loss_i.cpu().numpy()
        pos += bsz

        del xb, yb, feat, logits, g, g2, f2, norm_outer2, Mf, inner, score2, score, loss_i

    return scores, loss_all

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