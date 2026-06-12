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
nqubits_list = [7, 9]#, 11]
colors = ["green", "blue", "orange"]
markers = [">", "s", "o"]

lambda_list = np.arange(0, 0.36 + 0.04, 0.04)
runs = range(10)

def get_data(run, nqubits, lambda_val):
    """Récupère la colonne spécifique du CSV"""
    col_name = f"gammaN={lambda_val}"

    path = f"/content/drive/MyDrive/resultat_simu_osaka/random_circuit_noisy_M2/run_{run}/results_N_{nqubits}-Renyi_entropy.csv"
    df = pd.read_csv(path)
    return df[col_name]

# --- Calcul et Plot ---
plt.figure(figsize=(14, 6))

for i in range(len(nqubits_list)):
  nqubits = nqubits_list[i]
  color = colors[i]
  marker = markers[i]

  re_means = []
  re_errs = []

  for lambda_val in lambda_list:
    all_re = []

    for r in runs:
      # Moyenne temporelle après thermalisation
      all_re.append(get_data(r+1, nqubits, lambda_val)[4*nqubits]/nqubits)

    # Calcul stats sur les runs
    re_means.append(np.mean(all_re))
    re_errs.append(np.std(all_re) / np.sqrt(len(runs)))

  # Plot
  plt.errorbar(lambda_list, re_means, yerr=re_errs, marker=marker, label=f"N={nqubits}", capsize=3, color=color)

plt.vlines(x=0.22, ymin=1.2, ymax=1.3, colors="gray", linestyles="dashed")

plt.title("Renyi Entropy (Pauli)")

plt.xlabel("Gamma * N")
plt.ylabel("M(k=2)(t=4N)/N")
plt.legend()

plt.show()

"""fig_name = "average_entropy_vs_gammaN.png"
fig_path = os.path.join(save_dir, fig_name)
fig.savefig(fig_path, dpi=150)
plt.close(fig)"""