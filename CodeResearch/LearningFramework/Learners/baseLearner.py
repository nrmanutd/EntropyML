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