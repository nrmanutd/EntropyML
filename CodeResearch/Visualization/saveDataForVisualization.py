import numpy as np
import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def saveDataForVisualization(data):
    classesPair = data['classes']
    iStep = data['step']
    taskName = data['taskName']

    fileName = 'PValuesFigures\\PValuesLogs\\pValues_pairs_{0}_{1}_{2}.txt'.format(taskName, classesPair, iStep)

    with open(fileName, "w") as write:
        json.dump(data, write, cls=NumpyEncoder)