from abc import ABC, abstractmethod


class BaseLearner(ABC):
    @abstractmethod
    def train(self, x, y):
        pass

    @abstractmethod
    def test(self, model, x, y):
        pass