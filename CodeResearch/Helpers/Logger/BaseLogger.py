import time
from abc import ABC, abstractmethod

class BaseLogger(ABC):
    def __init__(self):
        self.startTime = time.time()
        self.prevTime = self.startTime

    @abstractmethod
    def logConcreteObject(self, object):
        pass

    def logWithTime(self, object, prefix):
        delta = time.time() - self.prevTime

        print(f'[{prefix}] {time.time() - self.startTime}s (+{delta}s): {str(object)}')
        self.prevTime = delta + self.prevTime

    def logDebug(self, object):
        self.logWithTime(object, 'Debug')

    def logWarn(self, object):
        self.logWithTime(object, 'Warn')

    def logError(self, object):
        self.logWithTime(object, 'Error')