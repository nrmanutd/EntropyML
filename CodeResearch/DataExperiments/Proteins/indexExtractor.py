import numpy as np
import pandas as pd
from CodeResearch.Visualization.saveDataForVisualization import deserialize_labeles_list_of_arrays, \
    serialize_labeled_list_of_arrays
from CodeResearch.dataSets import load_proteins

data = deserialize_labeles_list_of_arrays('KS_indexes_proteins_full_200_1_0_6327.txt')

d = pd.read_csv("../../Data/Proteins/df_master.csv")
selected_columns = 'ligand_id'
ligand_id = d[selected_columns].to_numpy()

ligands = ligand_id[data[0][0]]
serialize_labeled_list_of_arrays([ligands], data[1], data[2], data[3], 'ligands.txt')
