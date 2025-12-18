from CodeResearch.Helpers.Logger.BaseLogger import BaseLogger


class SimpleLogger(BaseLogger):
    def logConcreteObject(self, object):
        return self.logDebug(object)