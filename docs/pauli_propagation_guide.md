# Pauli Propagation Functions User Guide

## Purpose

The `pauli_propagation_functions` module provides high-level utilities to simulate quantum circuit evolution in the Heisenberg picture. It expands quantum operators into sums of Pauli strings and propagates them backwards through a circuit using the `PauliPropagation.jl` backend. This approach is highly efficient for analyzing operator growth, entanglement surrogates, and noise channels.

## Dependencies

To use this module, ensure the following packages are installed and available in your environment:

* `PauliPropagation`
* `LinearAlgebra`

---

## Main Workflow

1. **Define the Observable:** Represent your initial target operator using `PauliString` or `PauliSum` structures from `PauliPropagation.jl`.
2. **Execute Propagation:** Pass the circuit gates, parameters, and your observable into `propagate_layerbylayer`. The function handles the reverse-order layer execution required by the Heisenberg picture.
3. **Analyze Diagnostics:** Extract the evolved operator along with tracked data arrays for norm, entropy, and state overlaps across each time step.

---

## Exported Functions Summary

### Simulation Driver

* **`propagate_layerbylayer`**
The core workflow engine. Iterates backwards through circuit layers, applying single-layer propagation, optional truncation filters, noise attenuation, and automatic tracking of metrics.

### Diagnostics & Metrics

* **`overlap`**
Calculates the expectation value $\langle \psi | \hat{O} | \psi \rangle$ for a given dense state vector `ψ`. Optimizes computational performance if `ψ` is the $|0\dots0\rangle$ state.
* **`pauli_norm`**
Computes the squared Frobenius norm ($\sum |c_i|^2$) of the observable coefficients, acting as a diagnostic for tracking truncation loss.
* **`shannon_entropy`**
Measures the distribution complexity of the operator across the Pauli basis using normalized squared coefficients.
* **`renyi_entropy`**
Quantifies the "spread" or fragmentation of the operator in the Pauli basis for an order `k`. Automatically falls back to Shannon entropy if `k=1`.

### Noise & Transformations

* **`applynoiselayer`**
Modifies a `PauliSum` in-place to simulate decoherence. Attenuates coefficients exponentially according to the Pauli string weight (depolarizing) and specific $X$/$Y$ counts (dephasing).
* **`compute_matrix`**
Constructs the explicit, dense $2^n \times 2^n$ matrix representation of a Pauli operator or sum. (Best reserved for small qubit numbers).
* **`decode_pauli`**
A low-level bitwise decoder that converts unsigned integer representations of Pauli strings into human-readable text strings (e.g., `"IXYZ"`).

---

## Important Parameters

* **`max_weight` (Integer):** Imposes a hard cutoff on the number of non-identity elements allowed within a single Pauli string. Helps prevent exponential memory growth during long circuit depths.
* **`min_abs_coeff` (Float64):** The truncation threshold. Coefficients with an absolute value below this threshold are dropped from the `PauliSum` after a layer propagation step.
* **`γ` (Float64):** The noise strength scaling parameter used inside the single-layer propagation routines to regulate the intensity of depolarizing channel effects.