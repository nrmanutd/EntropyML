import numpy as np

def extractDataToStrings(x):
    columns_to_join = ['name', 'cat_level_1']
    result = x[columns_to_join].astype(str).agg(';'.join, axis=1).tolist()

    return np.array(result)
