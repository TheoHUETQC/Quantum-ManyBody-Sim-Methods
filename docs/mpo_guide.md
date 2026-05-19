# Matrix Product Operator (MPO) Functions User Guide

## Purpose

The `mpo_functions` module provides a comprehensive toolkit for simulating quantum operator evolution using Tensor Networks in the Heisenberg picture. By representing observables as Matrix Product Operators (MPOs), this module allows you to efficiently propagate operators backwards through a circuit while managing entanglement growth through systematic singular value truncations.

## Dependencies

To use this module, you must have the following Julia packages installed and loaded in your environment:

* `ITensors`
* `ITensorMPS`

---

## Main Workflow

1. **Define the Observable:** Initialize your target operator as an `MPO` using `ITensors.jl`.
2. **Determine Truncation Limits:** (Optional but recommended) Run `find_truncations` to identify the optimal bond dimensions and cutoffs for your specific circuit and observable.
3. **Execute Propagation:** Pass your circuit (a vector of tensor layers) and your MPO into `propagate_layerbylayer`. The function automatically handles the reverse-order execution and tracking of your desired metrics.
4. **Analyze the Results:** Extract the propagated MPO alongside the tracked diagnostic dictionaries (norms, max link dimensions, entropies, and overlaps).

---

## Exported Functions Summary

### Simulation Drivers

* **`propagate_layerbylayer`**
The core engine of the module. It iterates backwards through the circuit layers applying gates to the MPO ($U^\dagger O U$). It handles bond truncation, noise injection, and renormalization at each step, while automatically logging tracking metrics like the maximum link dimension.
* **`find_truncations`**
An automated sensitivity analysis tool. It iteratively sweeps through `maxdim` and `cutoff` values to find the most computationally lightweight parameters that still maintain physical accuracy (based on a user-defined tolerance for overlap convergence).

### Diagnostics & Metrics

* **`overlap`**
Computes the real-valued expectation value $\langle \psi | \hat{O} | \psi \rangle$ of the current MPO observable given a reference Matrix Product State (MPS) $\psi$.
* **`operator_entropy`**
Calculates the entanglement (Shannon) entropy of the MPO at a specific bond index. It performs an SVD to extract the singular values at the bond and computes the entropy of their normalized squared distribution.

### Noise & Transformations

* **`applynoiselayer`**
Injects decoherence into the system by applying a local depolarizing noise channel site-by-site to the MPO. It sums the $X$, $Y$, and $Z$ transformations and immediately truncates to prevent uncontrolled bond dimension growth.
* **`compute_matrix`**
Contracts the entire MPO tensor network into a single, dense $2^n \times 2^n$ matrix. *(Note: This scales exponentially and should only be used for small numbers of qubits for debugging or exact verification).*

---

## Important Parameters to Understand

* **`maxdim` (Integer):** The absolute maximum bond dimension (or "link dimension") allowed between any two adjacent tensors in the MPO. This acts as a hard memory ceiling to prevent exponential resource growth.
* **`cutoff` (Float64):** The singular value threshold for truncation. During SVDs (like gate application or noise addition), any singular values smaller than this threshold are discarded, compressing the network at the cost of a small, controlled loss of accuracy.
* **`γ` (Float64):** The noise strength scaling parameter. Dictates the probability/intensity of the depolarizing channel applied to the MPO during layer propagation.