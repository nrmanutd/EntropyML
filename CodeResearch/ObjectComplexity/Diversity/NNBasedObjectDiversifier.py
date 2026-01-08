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

    def calculateObjectDiversity(self, dataSet, target):

        self.logger.logDebug('Estimating object diversity...')

        sampler = self.samplerFactory(dataSet, target)
        currentModel = None

        device = self.learner.device
        all_epochs_scores = []

        for epoch in range(self.epochs):
            self.logger.logDebug(f'Estimating diversity for epoch #{epoch} of {self.epochs}...')

            batches = sampler.sample()
            g_list = []

            for xx, yy in batches:
                probs = np.full(len(yy), 1.0 / len(yy))

                currentModel = self.learner.train(xx, yy, probs) if currentModel is None else self.learner.update(
                    currentModel, xx, yy)

                # ---- 2) Перевод данных в torch на нужный device ----
                xb = torch.as_tensor(xx, dtype=torch.float32, device=device)
                yb = torch.as_tensor(yy, dtype=torch.int64, device=device)

                # ---- 3) Временно переключаем режим на eval() для стабильного измерения ----
                was_training = currentModel.training
                currentModel.eval()

                # важно: НЕ оборачивай это в torch.no_grad(), т.к. градиенты нужны
                G_batch = self.per_sample_grads_last_layer_loop(currentModel, xb, yb)
                # или: G_batch = self.per_sample_grads_vmap(currentModel, xb, yb)

                # ---- 4) Вернуть режим как было ----
                currentModel.train(was_training)

                g_list.extend(G_batch.detach())

            g_delta = self.calculateDelta(g_list)
            all_epochs_scores.append(g_delta)

        final_scores = np.mean(np.stack(all_epochs_scores, axis=0), axis=0)

        self.logger.logDebug('Object diversity estimated.')

        return final_scores

    def calculateDelta(self, g_list, mode="cosine"):
        G_epoch = torch.cat(g_list, dim=0)  # [N, D] на GPU

        # --- 2) средний вектор за эпоху ---
        g_mean = G_epoch.mean(dim=0, keepdim=True)  # [1, D]

        # --- 3) diversity относительно среднего (на GPU) ---
        if mode == "l2_relative":
            # diversity_i = ||g_i - mean|| / ||mean||
            diff = G_epoch - g_mean
            eps = 1e-8
            denom = g_mean.norm(p=2, dim=1).clamp_min(eps)  # [1]
            scores = diff.norm(p=2, dim=1) / denom  # [N]

        elif mode == "cosine":
            # diversity_i = 1 - cos(g_i, mean_dir)
            mean_dir = F.normalize(g_mean, p=2, dim=1)  # [1, D]
            # если G_epoch уже нормирован (как у тебя), то это просто dot
            cos = (G_epoch * mean_dir).sum(dim=1).clamp(-1.0, 1.0)  # [N]
            scores = 1.0 - cos

        else:
            raise ValueError("mode must be 'cosine' or 'l2_relative'")

        # --- 4) вернуть numpy ---
        return scores.detach().cpu().numpy()

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
        G = F.normalize(G, dim=1)  # direction-only (опционально)
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
            gi_vec = F.normalize(gi_vec, dim=0)
            grads.append(gi_vec)
        return torch.stack(grads, dim=0)