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


def serialize_labeled_list_of_arrays(arrays, names, task_name, n_attempts, filename):
    """
    Сериализует список массивов, их названия, taskName и nAttempts в JSON файл.

    Параметры:
    - arrays: список одномерных массивов (списки или numpy массивы)
    - names: список строк с названиями массивов
    - task_name: строка, название задачи
    - n_attempts: число, количество попыток
    - filename: имя файла для сохранения
    """
    if len(arrays) != len(names):
        raise ValueError("Количество массивов и названий должно совпадать")

    # Преобразуем массивы в списки (если это numpy массивы)
    data = {
        "taskName": task_name,
        "nAttempts": n_attempts,
        "names": names,
        "arrays": [arr.tolist() if isinstance(arr, np.ndarray) else list(arr) for arr in arrays]
    }

    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)  # indent=4 для красивого форматирования


def deserialize_labeles_list_of_arrays(filename):
    """
    Десериализует данные из JSON файла, возвращает:
    - arrays: список массивов
    - names: список названий
    - task_name: название задачи
    - n_attempts: количество попыток
    """

    with open(filename, 'r') as f:
        data = json.load(f)

    return (
        data["arrays"],
        data["names"],
        data["taskName"],
        data["nAttempts"]
    )