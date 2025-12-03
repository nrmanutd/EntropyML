import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, nFeatures, nClasses, dense):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(nFeatures, dense),
            nn.ReLU(),
            nn.Linear(dense, dense),
            nn.ReLU(),
            nn.Linear(dense, nClasses)
        )

    def forward(self, x):
        return self.layers(x)