from abc import ABC, abstractmethod


class BaseSampler(ABC):

    @abstractmethod
    def sample(self, seed=None):
        pass
