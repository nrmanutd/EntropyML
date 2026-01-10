import torch
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
    feat = model[:-1](xb)  # [B, H]
    logits = model[-1](feat)  # [B, C]
    losses = F.cross_entropy(logits, yb, reduction="none")

    params = [p for p in model[-1].parameters() if p.requires_grad]  # только last layer

    grads = []
    for i in range(losses.size(0)):
        gi = torch.autograd.grad(losses[i], params, retain_graph=True)
        gi_vec = torch.cat([g.flatten() for g in gi])
        # gi_vec = F.normalize(gi_vec, dim=0)
        grads.append(gi_vec)
    return torch.stack(grads, dim=0)

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