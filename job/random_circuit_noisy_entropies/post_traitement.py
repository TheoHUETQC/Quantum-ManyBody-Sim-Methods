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
nqubits_list = [7, 8]#, 9, 15, 20]
lambda_list = np.arange(0, 0.4 + 0.05, 0.05)
runs = range(10)

maxdim_list = [2 ** i for i in range(2,7+1)]
maxsize_list = [2 ** i for i in range(5,10+1)]

def get_data(run, nqubits, entropy_type, lambda_val, max_val):
    """Récupère la colonne spécifique du CSV"""
    col_name = f"gammaN={lambda_val:.1f}, max{'_size' if 'Renyi' in entropy_type else 'dim'}={max_val}"

    path = f"/content/drive/MyDrive/resultat_simu_osaka/random_circuit_noisy_entropies/run_{run}/results_N_{nqubits}-{entropy_type}.csv"
    df = pd.read_csv(path)
    return df[col_name]

thermalisation_pp = 25
thermalisation_mpo = 35

# --- Calcul et Plot ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
cmap = plt.get_cmap('viridis')
norm = Normalize(vmin=lambda_list.min(), vmax=lambda_list.max())

for maxdim, max_size in zip(maxdim_list, maxsize_list):
    for lambda_val in lambda_list:
        re_means, oe_means = [], []
        re_errs, oe_errs = [], []

        for nqubits in nqubits_list:
            all_re = []
            all_oe = []

            for r in runs:
                re_series = get_data(r+1, nqubits, "Renyi_entropy", lambda_val, max_size)
                oe_series = get_data(r+1, nqubits, "Operator_Entanglement", lambda_val, maxdim)

                # Moyenne temporelle après thermalisation
                all_re.append(re_series[thermalisation_pp:].mean())
                all_oe.append(oe_series[thermalisation_mpo:].mean())

            # Calcul stats sur les runs
            re_means.append(np.mean(all_re))
            re_errs.append(np.std(all_re) / np.sqrt(len(runs)))

            oe_means.append(np.mean(all_oe))
            oe_errs.append(np.std(all_oe) / np.sqrt(len(runs)))

        # Plot
        color = cmap(norm(lambda_val))
        axes[0].errorbar(nqubits_list, re_means, yerr=re_errs, marker='o', label=f"λ={lambda_val:.2f}", capsize=3, color=color)
        axes[1].errorbar(nqubits_list, oe_means, yerr=oe_errs, marker='s', capsize=3, color=color)

axes[0].set_title("Average Renyi Entropy (Pauli)")
axes[1].set_title("Average Operator Entanglement (MPO)")
for ax in axes:
    ax.set_xlabel("Number of Qubits (N)")
    ax.set_ylabel("Entropy")
    ax.grid(True, linestyle='--', alpha=0.6)

sm = ScalarMappable(cmap=cmap, norm=norm)
cbar = fig.colorbar(sm, ax=axes, pad=0.02)
cbar.set_label("Gamma * N", rotation=270, labelpad=15)

plt.show()

fig_name = "average_entropy_vs_nqubits.png"
fig_path = os.path.join(save_dir, fig_name)
fig.savefig(fig_path, dpi=150)
plt.close(fig)