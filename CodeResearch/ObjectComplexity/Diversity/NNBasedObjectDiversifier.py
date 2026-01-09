import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, vmap, grad

from CodeResearch.Helpers.Logger.BaseLogger import BaseLogger
from CodeResearch.LearningFramework.Learners.TorchLearner import TorchMLPLearner
from CodeResearch.ObjectComplexity.Diversity.BaseObjectDiversifier import BaseObjectDiversifier


class NNBasedObjectDiversifier(BaseObjectDiversifier):
    def __init__(self, learner: TorchMLPLearner, samplerFactory, epochs, logger: BaseLogger):

        self.logger = logger
        self.learner = learner
        self.epochs = epochs
        self.samplerFactory = samplerFactory

    def calculateObjectDiversity(self, dataSet, target, baseDataSet, baseTarget):
        self.logger.logDebug('Estimating object diversity...')

        sampler = self.samplerFactory(dataSet, target)
        currentModel = None

        device = self.learner.device
        all_epochs_scores = []

        if baseDataSet is not None and len(baseTarget) != 0:
            currentModel = self.learner.train(baseDataSet, baseTarget, np.full(len(baseTarget), 1.0 / len(baseTarget)))

        for epoch in range(self.epochs):
            self.logger.logDebug(f'Estimating diversity for epoch #{epoch} of {self.epochs}...')

            batches = sampler.sample()
            g_list = []

            for xx, yy in batches:
                # ---- 2) Перевод данных в torch на нужный device ----
                xb = torch.as_tensor(xx, dtype=torch.float32, device=device)
                yb = torch.as_tensor(yy, dtype=torch.int64, device=device)

                if currentModel is not None:
                    # ---- 3) Временно переключаем режим на eval() для стабильного измерения ----
                    was_training = currentModel.training
                    currentModel.eval()

                    # важно: НЕ оборачивай это в torch.no_grad(), т.к. градиенты нужны
                    #G_batch = self.per_sample_grads_last_layer_loop(currentModel, xb, yb)
                    G_batch = self.per_sample_grads_vmap(currentModel, xb, yb)

                    # ---- 4) Вернуть режим как было ----
                    currentModel.train(was_training)
                else:
                    tempModel = self.learner.build_model()
                    tempModel = tempModel.to(device)
                    tempModel.eval()

                    #G_batch = self.per_sample_grads_last_layer_loop(tempModel, xb, yb)
                    G_batch = self.per_sample_grads_vmap(tempModel, xb, yb)

                g_list.append(G_batch.detach())

                probs = np.full(len(yy), 1.0 / len(yy))

                currentModel = self.learner.train(xx, yy, probs) if currentModel is None else self.learner.update(
                    currentModel, xx, yy)

            g_delta = self.calculateDelta(g_list, mode='grad_norm')
            all_epochs_scores.append(g_delta)

        final_scores = np.mean(np.stack(all_epochs_scores, axis=0), axis=0)

        self.logger.logDebug('Object diversity estimated.')

        return final_scores

    def calculateDelta(self, g_list, mode):
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

    def per_sample_grads_vmap(self, model, xb, yb):
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
        #G = F.normalize(G, dim=1)  # direction-only (опционально)
        return G

    def per_sample_grads_last_layer_loop(self, model, xb, yb):
        # forward
        feat = model[:-1](xb)  # [B, H]
        logits = model[-1](feat)  # [B, C]
        losses = F.cross_entropy(logits, yb, reduction="none")

        params = [p for p in model[-1].parameters() if p.requires_grad]  # только last layer

        grads = []
        for i in range(losses.size(0)):
            gi = torch.autograd.grad(losses[i], params, retain_graph=True)
            gi_vec = torch.cat([g.flatten() for g in gi])
            #gi_vec = F.normalize(gi_vec, dim=0)
            grads.append(gi_vec)
        return torch.stack(grads, dim=0)