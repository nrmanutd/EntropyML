import json


def saveDataForVisualization(data):
    classesPair = data['classes']
    iStep = data['step']
    taskName = data['taskName']

    fileName = 'PValuesFigures\\pValues_pairs_{0}_{1}_{2}.txt'.format(taskName, classesPair, iStep)
    with open(fileName, 'w') as convert_file:
        convert_file.write(str(data))