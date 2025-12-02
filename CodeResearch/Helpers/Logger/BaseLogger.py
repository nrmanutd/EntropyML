import time
from abc import ABC, abstractmethod

class BaseLogger(ABC):
    @abstractmethod
    def logConcreteObject(self, object):
        pass

    def logWithTime(self, object, prefix):
        str = self.logConcreteObject(object)
        print(f'[{prefix}]{time.time()}: {str}')

    def logDebug(self, object):
        self.logWithTime(object, 'Debug')

    def logWarn(self, object):
        self.logWithTime(object, 'Warn')

    def logError(self, object):
        self.logWithTime(object, 'Error')