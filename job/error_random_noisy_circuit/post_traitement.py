from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os

# for the colormap
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import ast

def convert_to_list(data_series):
  return data_series.apply(ast.literal_eval).tolist()

# Define the directory to save figures
save_dir = 'figures'
os.makedirs(save_dir, exist_ok=True) # Create the directory if it doesn't exist

# --- Paramètres ---
nqubits_list = [5, 6, 7, 8, 9]
lambda_list = np.arange(0, 0.4 + 0.05, 0.05)
runs = range(10)

maxdim_list = [2 ** i for i in range(2,6+1)]
maxsize_list = [4 ** i for i in range(2,6+1)]

thermalisation_pp = 25
thermalisation_mpo = 35

def get_data(run, nqubits, method_type, lambda_val):
    """Récupère la colonne spécifique du CSV
    method_type = "pp" or "mpo"
    """
    col_name = f"error gammaN={lambda_val:.1f}"

    path = f"/content/drive/MyDrive/resultat_simu_osaka/random_circuit_noisy_error/run_{run}/results_N_{nqubits}-error_{method_type}.csv"
    df = pd.read_csv(path)
    return df[col_name]

# --- Calcul et Plot ---
for nqubits in nqubits_list:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    cmap = plt.get_cmap('viridis')
    norm = Normalize(vmin=lambda_list.min(), vmax=lambda_list.max())
    for lambda_val in lambda_list:
        all_pp_error = []
        all_mpo_error = []
        for r in runs:
            pp_error = np.mean(convert_to_list(get_data(r+1, nqubits, "pp", lambda_val)), axis=1)
            mpo_error = np.mean(convert_to_list(get_data(r+1, nqubits, "mpo", lambda_val)), axis=1)

            all_pp_error.append(pp_error)
            all_mpo_error.append(mpo_error)

        # Calcul stats sur les runs
        pp_error_means = (np.mean(all_pp_error, axis=0))
        pp_error_errs = (np.std(all_pp_error, axis=0) / np.sqrt(len(runs)))

        mpo_error_means = (np.mean(all_mpo_error, axis=0))
        mpo_error_errs = (np.std(all_mpo_error, axis=0) / np.sqrt(len(runs)))

        # Plot
        color = cmap(norm(lambda_val))
        axes[0].errorbar(maxsize_list, pp_error_means, yerr=pp_error_errs, marker='o', label=f"λ={lambda_val:.2f}", capsize=3, color=color)
        axes[1].errorbar(maxdim_list, mpo_error_means, yerr=mpo_error_errs, marker='s', capsize=3, color=color)

    axes[0].set_title(f"nqubits={nqubits}, overlap error (Pauli)")
    axes[1].set_title("overlap error (MPO)")
    axes[0].set_xlabel("Max size (N_p)")
    axes[1].set_xlabel("max bond dim (Chi)")
    for ax in axes:
        ax.set_ylabel("Error")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xscale('log')

    sm = ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=axes, pad=0.02)
    cbar.set_label("Gamma * N", rotation=270, labelpad=15)

    plt.show()

    fig_name = f"overlap_error_nqubits_{nqubits}.png"
    fig_path = os.path.join(save_dir, fig_name)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)