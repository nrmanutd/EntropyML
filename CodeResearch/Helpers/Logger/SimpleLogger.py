from CodeResearch.Helpers.Logger.BaseLogger import BaseLogger


class SimpleLogger(BaseLogger):
    def logConcreteObject(self, object):
        self.logDebug(object)