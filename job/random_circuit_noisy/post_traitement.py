from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os

# for the colormap
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

"""# List of parameters that vary in our simulation"""

nqubits_list = [5, 6, 7, 8 ]#, 9, 10]
gamma_list = np.arange(0, 0.4, 0.02)
runs = 10

"""# Compute error bar"""

def mean_error(entropy):
  K = len(entropy)
  if K < 2:
    return entropy[0], 0  # Impossible de calculer une erreur avec un seul point

  e_mean = np.mean(entropy, axis=0)
  e_std = np.std(entropy, axis=0)

  return e_mean,  e_std / np.sqrt(K)

"""# Compute data convergence"""

def estimate_convergence(valeurs, tolerance=1e-3, fallback_window=0.1):
    """
    estimate_convergence(valeurs, tolerance=1e-3, fallback_window=0.1)

    Determines the convergence value of a curve (list of values).

    Arguments:
    - values: list or np.array, the y-coordinates of the curve.
    - tolerance: float, the variation threshold below which convergence is considered to have occurred.
    - end_window: float, the proportion of the end of the curve to be analyzed (0.1 = the last 10% of values).

    Returns:
    - float: the convergence value if found.
    - None: if the curve does not appear to converge.
    """
    if len(valeurs) < 10:
        raise ValueError("The list of values is too short to analyze convergence.")

    # Conversion en array numpy pour faciliter les calculs
    y = np.array(valeurs)
    n = len(y)

    # On prend les index pour la fin de la courbe
    window_size = int(n * fallback_window)
    if window_size < 3:
        window_size = 3 # Sécurité pour les listes courtes

    # On sépare la fin de la courbe en deux sous-segments pour comparer leur stabilité
    fin_1 = y[-window_size : -window_size // 2]
    fin_2 = y[-window_size // 2 :]

    moyenne_1 = np.mean(fin_1)
    moyenne_2 = np.mean(fin_2)

    # Écart entre les deux moyennes de fin
    variation = abs(moyenne_2 - moyenne_1)

    # Optionnel : On vérifie aussi que la pente globale sur la fin est proche de zéro
    # pour éviter de valider une courbe qui monte encore de façon très linéaire et lente
    pente = abs(y[-1] - y[-window_size]) / window_size

    if variation < tolerance and pente < tolerance:
        # La valeur de convergence estimée est la moyenne des dernières valeurs
        return float(moyenne_2)
    else:
        print("The curve does not converge")
        return None
    
"""# Retrieval of simulation data"""

def csv_to_df(run, nqubits, entropy) :
  """
  entropy = "Renyi_entropy" or "Operator_Entanglement"
  """
  file_name = f"results_N_{nqubits}-{entropy}.csv"
  path= f"/content/drive/MyDrive/resultat_simu_osaka/random_circuit_noisy/results/run_{run}/"+file_name
  return pd.read_csv(path)

df_dict = {}
for nqubits in nqubits_list :
  print(f"nqubits : {nqubits}")
  for gamma in gamma_list :
    print(f"- gamma : {gamma}")
    renyi_entropy, operator_entanglement = [], []
    for run in range(1, runs+1) :
      renyi_entropy.append(csv_to_df(run, nqubits, "Renyi_entropy")[f"gamma={gamma}"])
      operator_entanglement.append(csv_to_df(run, nqubits, "Operator_Entanglement")[f"gamma={gamma}"])

    re_mean, re_error = mean_error(renyi_entropy)
    oe_mean, oe_error = mean_error(operator_entanglement)

    df_dict[(nqubits, gamma)] = pd.DataFrame({"Layer" : csv_to_df(run, nqubits, "Renyi_entropy")["Layer"],
                                  "Renyi_entropy" : re_mean,
                                  "Renyi_entropy_error" : re_error,
                                  "Operator_Entanglement" : oe_mean,
                                  "Operator_Entanglement_error" : oe_error})

"""# Thermalisation parameter

It is at this level that we consider our system to be stable
"""

thermalisation_layer_pp = 25
thermalisation_layer_mpo = 35

"""# Plot entropies vs layer"""

# Define the directory to save figures
save_dir = 'figures'
os.makedirs(save_dir, exist_ok=True) # Create the directory if it doesn't exist

cmap = plt.get_cmap('viridis') # inferno, rainbow, RdBu, Blues
norm = Normalize(vmin=gamma_list.min(), vmax=gamma_list.max())

for nqubits in nqubits_list:
  fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

  for gamma in gamma_list:
    df = df_dict[(nqubits, gamma)]
    axes[0].errorbar(df["Layer"], df["Renyi_entropy"], yerr=df["Renyi_entropy_error"], capsize=3, color=cmap(norm(gamma)))
    axes[1].errorbar(df["Layer"], df["Operator_Entanglement"], yerr=df["Operator_Entanglement_error"], capsize=3, color=cmap(norm(gamma)))

  for i in range(2):
    axes[i].set_xlabel("layers")
    axes[i].set_ylabel("Entropy")
    axes[i].grid(True, linestyle='--', alpha=0.6)

  axes[0].axvline(x=thermalisation_layer_pp, color='red',linestyle='--', label="Thermalisation")
  axes[1].axvline(x=thermalisation_layer_mpo, color='red',linestyle='--', label="Thermalisation")

  axes[0].set_title(f"nqubits = {nqubits}, Renyi entropy (Pauli)")
  axes[1].set_title("Operator entanglement (MPO)")

  sm = ScalarMappable(cmap=cmap, norm=norm)
  sm.set_array(gamma_list)
  cbar = fig.colorbar(sm, ax=axes)
  cbar.set_label("gamma", loc='center', rotation=270, labelpad=15)

  plt.legend()
  plt.show()

  # Save the figure
  fig_name = f"entropy_vs_layer_nqubits_{nqubits}.png"
  fig_path = os.path.join(save_dir, fig_name)
  fig.savefig(fig_path, dpi=150)
  plt.close(fig)

"""# Plot average entropies vs number of qubits, Complexity"""

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

cmap = plt.get_cmap('viridis')
norm = Normalize(vmin=gamma_list.min(), vmax=gamma_list.max())

for gamma in gamma_list:
  renyi_entropy_average, operator_entanglement_average = [], []
  renyi_entropy_error, operator_entanglement_error = [], []

  for nqubits in nqubits_list:
    df = df_dict[(nqubits, gamma)]

    re_mean, re_error = mean_error(df["Renyi_entropy"][:thermalisation_layer_pp])
    oe_mean, oe_error = mean_error(df["Operator_Entanglement"][:thermalisation_layer_mpo])

    renyi_entropy_average.append(re_mean)
    operator_entanglement_average.append(oe_mean)
    renyi_entropy_error.append(re_error)
    operator_entanglement_error.append(oe_error)

  axes[0].errorbar(nqubits_list, renyi_entropy_average, yerr=renyi_entropy_error, capsize=3, color=cmap(norm(gamma)))
  axes[1].errorbar(nqubits_list, operator_entanglement_average, yerr=operator_entanglement_error, capsize=3, color=cmap(norm(gamma)))
axes[0].set_title("Average Renyi entropy (Pauli)")
axes[1].set_title("Average Operator entanglement (MPO)")

for i in range(2):
  axes[i].set_xlabel("nqubits")
  axes[i].grid(True, linestyle='--', alpha=0.6)

sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array(gamma_list)
cbar = fig.colorbar(sm, ax=axes)
cbar.set_label("gamma", loc='center', rotation=270, labelpad=15)

plt.show()

# Save the figure
fig_name = "average_entropy_vs_nqubits.png"
fig_path = os.path.join(save_dir, fig_name)
fig.savefig(fig_path, dpi=150)
plt.close(fig)