import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm
from time import perf_counter

data_dir = './keren/keren/'
files = os.listdir(data_dir)

# It seems to me that each X array contains a separate image of the cells. 

# I am a little confused as to how to extract each individual cell from the
# image. Ah OK nevermind, this is what the y array is for (I think)

# Need to make a training dataset with expression information for each cell. 
training_data = {'cell_number':[], 'marker':[], 'expression':[], 'cell_type':[]}

# for file in tqdm(files, desc='Loading files...'):
for file in [f for f in files if f != 'meta.yaml']:
    npz = np.load(data_dir + file, allow_pickle=True)
    # Because npz['cell_types'] has shape (), need to use item() to extract the
    # dictionary. 
    X, y, cell_types = npz['X'], npz['y'], npz['cell_types'].item()

    # X array has shape (1, 2048, 2048, 51). Last index is channels (i.e. markers)
    # y array has shape (1, 2048, 2048, 2) 
    # cell_types array is a dictionary mapping the mask indices to the cell types.

    # First, normalize the data...
    X = X / np.max(X)

    # Probably should use the whole-cell segments, not nuclear...
    y = y[:, :, :, 0]
   
    ncells, nmarkers = np.max(y), X.shape[-1]
    for i in tqdm(range(ncells), desc='Loading cell data...'):
        for j in range(nmarkers):

            cell_data = X[:, :, :, j][np.where(y == i)]
            # Add stuff to the dataset. 
            training_data['cell_number'].append(i)
            training_data['marker'].append(j)
            training_data['expression'].append(np.mean(cell_data))
            training_data['cell_type'].append(cell_types[i])
    break

# Write all the cleaned-up data to a CSV file to avoid re-generating. 
training_data = pd.DataFrame(training_data)
training_data.to_csv('assignment_01_data_cleaned.csv')

# Create a marker expression panel (how different markers correllate with
# different cell types)
