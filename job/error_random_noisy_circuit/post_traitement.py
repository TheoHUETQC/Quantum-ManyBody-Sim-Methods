from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os

# for the colormap
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Define the directory to save figures
save_dir = 'figures'
os.makedirs(save_dir, exist_ok=True) # Create the directory if it doesn't exist

# --- Paramètres ---
nqubits_list = [5, 6, 7, 8]#, 9]
lambda_list = np.arange(0, 0.4 + 0.05, 0.05)
runs = range(10)

maxdim_list = [2 ** i for i in range(2,6+1)]
maxsize_list = [4 ** i for i in range(2,6+1)]

def get_data(run, nqubits, method_type, lambda_val, max_val):
    """Récupère la colonne spécifique du CSV"""
    col_name = f"gammaN={lambda_val:.1f}, max{'_size' if 'pp' in method_type else 'dim'}={max_val}"

    path = f"/content/drive/MyDrive/resultat_simu_osaka/random_circuit_noisy_error/run_{run}/results_N_{nqubits}-error_{method_type}.csv"
    df = pd.read_csv(path)
    return df[col_name]
