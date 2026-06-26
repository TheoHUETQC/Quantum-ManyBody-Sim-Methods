from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os

# Define the directory to save figures
save_dir = 'figures'
os.makedirs(save_dir, exist_ok=True) # Create the directory if it doesn't exist

# --- Paramètres ---
nqubits_list = [7, 9, 11]
colors = ["green", "blue", "orange"]
markers = [">", "s", "o"]

lambda_list = np.arange(0, 0.36 + 0.04, 0.04)
runs = range(10)

def get_data(run, nqubits, lambda_val, entropy_type):
    """Récupère la colonne spécifique du CSV
    entropy_type = OE or OSE"""
    col_name = f"gammaN={lambda_val}"

    path = f"/content/drive/MyDrive/resultat_simu_osaka/random_circuit_noisy_M2/run_{run}/results_N_{nqubits}-{entropy_type}.csv"
    df = pd.read_csv(path)
    return df[col_name]

# --- Calcul et Plot ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

for i in range(len(nqubits_list)):
  nqubits = nqubits_list[i]
  color = colors[i]
  marker = markers[i]

  OSE_means = []
  OSE_errs = []
  OE_means = []
  OE_errs = []

  for lambda_val in lambda_list:
    all_OSE = []
    all_OE = []

    for r in runs:
      # Moyenne temporelle après thermalisation
      all_OSE.append(get_data(r+1, nqubits, lambda_val, "OSE")[4*nqubits]/nqubits)
      all_OE.append(get_data(r+1, nqubits, lambda_val, "OE")[4*nqubits]/nqubits)

    # Calcul stats sur les runs
    OSE_means.append(np.mean(all_OSE))
    OSE_errs.append(np.std(all_OSE) / np.sqrt(len(runs)))
    OE_means.append(np.mean(all_OE))
    OE_errs.append(np.std(all_OE) / np.sqrt(len(runs)))

  # Plot
  axes[0].errorbar(lambda_list, OSE_means, yerr=OSE_errs, marker=marker, label=f"N={nqubits}", capsize=3, color=color)
  axes[1].errorbar(lambda_list, OE_means, yerr=OE_errs, marker=marker, label=f"N={nqubits}", capsize=3, color=color)

axes[0].vlines(x=0.22, ymin=0.75, ymax=1.3, colors="gray", linestyles="dashed")

axes[0].set_title("OSE (Pauli)")
axes[0].set_ylabel(r'$\overline{M^{(k=2)}}(t=4N)/N$')
axes[0].legend()

axes[1].set_title("OE (MPO)")
axes[1].set_ylabel(r'$\overline{S_{0E}^{(k=2)}}(t=4N)/N$')

for axe in axes:
  axe.set_xlabel(r'$\gamma  N$')

fig.show()

"""fig_name = "average_OSE_vs_gammaN.png"
fig_path = os.path.join(save_dir, fig_name)
fig.savefig(fig_path, dpi=150)
plt.close(fig)"""