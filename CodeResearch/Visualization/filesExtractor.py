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