import torch
import numpy as np
from torch import nn as nn

from CodeResearch.LearningFramework.Learners.NNTorchModelLearner import TorchModelLearner


class NNUpdatableTorchModelLearner(TorchModelLearner):
    def train(self, x, y, probs=None):
        model = self.build_model().to(self.device)
        model, optimizer, a, p = self._trainModel(model, x, y, probs, self.epochs, None, None)

        return model, optimizer, self.scaler

    def update(self, m, x, y):
        model = m[0]
        optimizer = m[1]
        scaler = m[2]

        new_opt = type(optimizer)(model.parameters(), **optimizer.defaults)
        new_opt.load_state_dict(optimizer.state_dict())

        new_scaler = torch.cuda.amp.GradScaler(enabled=scaler.is_enabled())
        new_scaler.load_state_dict(scaler.state_dict())

        self.scaler = new_scaler

        model = model.to(self.device)
        model, optimizer, a, p = self._trainModel(model, x, y, None, self.update_epochs, None, None, new_opt)

        return model, optimizer

    def test(self, m, x, y):
        model = m[0]

        model = model.to(self.device)
        model.eval()

        x, y, _ = self._to_tensors(x, y, probs=None)
        loader = self._make_loader(x, y, probs=None, shuffle=False)

        correct = 0
        total = 0
        all_preds = []

        with torch.no_grad():
            for xb, yb in loader:
                logits = model(xb)
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.detach().cpu().numpy())

                correct += int((preds == yb).sum().item())
                total += int(yb.size(0))

        acc = correct / max(total, 1)
        preds_np = np.concatenate(all_preds, axis=0) if all_preds else np.array([])
        return acc, preds_np

    def trainAndTest(self, x, y, probs, xt, yt):
        model = self.train(x, y, probs)
        acc, preds_np = self.test(model, xt, yt)

        del model
        torch.cuda.empty_cache()

        return acc, preds_np