import numpy as np
from abc import ABC, abstractmethod


class BaseLearner(ABC):
    @abstractmethod
    def train(self, x, y, probs):
        pass

    @abstractmethod
    def update(self, model, x, y):
        pass

    @abstractmethod
    def test(self, model, x, y):
        pass

    @abstractmethod
    def trainAndTest(self, x, y, probs, xt, yt):
        model = self.train(x, y, probs)
        return self.test(model, xt, yt)

    def extendSet(self, xList, yList):
        extended_x = np.concatenate(xList, axis=0) if xList[1] is not None else xList[0]
        extended_y = np.concatenate(yList) if yList[1] is not None else yList[0]

        return extended_x, extended_y