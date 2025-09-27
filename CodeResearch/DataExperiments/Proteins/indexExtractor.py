import numpy as np
import pandas as pd

from CodeResearch.Visualization.filesExtractor import getLastFile
from CodeResearch.Visualization.saveDataForVisualization import deserialize_labeles_list_of_arrays, \
    serialize_labeled_list_of_arrays
from CodeResearch.dataSets import load_proteins

def getProteinsComplexities(folder, taskName, iterations):
    frequenciesFile = getLastFile(folder, f"^KS_frequency_{taskName}_{iterations}_\d+_\d+_\d+.txt$")
    indexesFile = getLastFile(folder, f"^KS_indexes_{taskName}_{iterations}_\d+_\d+_\d+.txt$")

    frequencesData = deserialize_labeles_list_of_arrays(frequenciesFile)
    indexesData = deserialize_labeles_list_of_arrays(indexesFile)

    d = pd.read_csv("../../Data/Proteins/df_master.csv")
    selected_columns = 'ligand_id'
    ligand_id = d[selected_columns].to_numpy()

    ligands = ligand_id[indexesData[0][0]]
    frequences = frequencesData[0][0]

    df = pd.DataFrame({
        'ligand_id': ligands,
        'frequency': frequences
    })

    return df


