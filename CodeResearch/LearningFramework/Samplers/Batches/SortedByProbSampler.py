import numpy as np
import torch
from torch.utils.data import Sampler


class SortedByProbSampler(Sampler):
    def __init__(self, probs):
        super().__init__()
        if isinstance(probs, np.ndarray):
            probs = torch.from_numpy(probs)
        elif not isinstance(probs, torch.Tensor):
            probs = torch.tensor(probs)

        probs = probs.view(-1)
        self.indicies = torch.argsort(probs, descending=True).tolist()

    def __iter__(self):
        return iter(self.indicies)

    def __len__(self):
        return len(self.indicies)