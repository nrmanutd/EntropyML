import os
import re

def find_files_with_regex(directory, pattern):
    """
    Находит файлы в указанной директории, соответствующие регулярному выражению
    """
    regex = re.compile(pattern)
    matching_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if regex.search(file):
                matching_files.append(os.path.join(root, file))

    return matching_files

def getLastFile(logsFolder, pattern):

    files = find_files_with_regex(logsFolder, pattern)
    maxObjects = 0
    bestFile = files[0]

    p = r".*_\d+_\d+_(\d+)\.txt$"

    for file in files:
        match = re.search(p, file)
        num1 = int(match.groups()[0])

        if num1 > maxObjects:
            bestFile = file
            maxObjects = num1

    return bestFile

def getLastFilesFromFolder(folder, pattern):
    files = find_files_with_regex(folder, pattern)
    files = getLastFiles(files)
    return files

def getLastFiles(files):
    data = {}
    pattern = r"^.*KS.*_.*_.*_(\d+)_(\d+)_(\d+).txt$"

    for file in files:
        match = re.search(pattern, file)
        num1, num2, num3 = match.groups()
        key = "{0}_{1}".format(num1, num2)

        num3 = int(num3)

        if key not in data:
            data[key] = {'file': file, 'objects': num3}
        else:
            if data[key]['objects'] < num3:
                data[key] = {'file': file, 'objects': num3}

    files = []
    for k, v in data.items():
        files.append(v['file'])

    return files