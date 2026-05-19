# Exact Simulation Functions User Guide

## Purpose

The `exact_functions` module provides a reference implementation for simulating quantum operator evolution using full, dense matrices in the Heisenberg picture. Unlike tensor network approaches that compress the system state, this module tracks the complete, uncompressed operator state. It is primarily used for validating approximate methods (such as MPO simulations) on smaller qubit counts where exact numerical calculations are feasible.

## Dependencies

This module relies entirely on the Julia standard library:

* **`LinearAlgebra`** (for matrix multiplication, adjoints, SVDs, and norm calculations)

---

## Main Workflow

1. **Initialize the Operator:** Create a dense representation of your physical observable. For example, use `get_Zi` to generate a local Pauli-Z matrix for a specific qubit.
2. **Execute Exact Propagation:** Pass your circuit (represented as a vector of dense unitary matrix layers) and your operator matrix into `propagate_layerbylayer`. The function performs exact matrix multiplications backwards through the circuit.
3. **Analyze Metrics:** Retrieve the fully evolved matrix and inspect the tracked dictionary for exact diagnostics, including full-rank operator entropy and exact state expectation values.

---

## Exported Functions Summary

### Simulation Drivers

* **`propagate_layerbylayer`**
The core engine for dense simulation. It processes circuit layers in reverse order according to the Heisenberg picture, updating the observable via $O \leftarrow U^\dagger O U$. It tracks exact matrix norms, exact state overlaps, and full-rank partition entropies at every layer.

### Diagnostics & Metrics

* **`overlap`**
Computes the exact real-valued expectation value $\langle \psi | \hat{O} | \psi \rangle$ by performing direct matrix-vector multiplication using a dense matrix $O$ and a state vector $\psi$.
* **`operator_entropy`**
Calculates the exact entanglement (Shannon) entropy of an operator across a specified qubit partition (bond). It reshapes the full $2^n \times 2^n$ matrix into a bipartite tensor and computes the singular value spectrum without truncation.

### Operator Generation

* **`get_Zi`**
Generates a full $2^n \times 2^n$ dense complex matrix representing a Pauli-Z operator acting on a targeted qubit $i$, embedded in an $n$-qubit space via Kronecker products ($I \otimes \dots \otimes Z_i \otimes \dots \otimes I$).

---

## Important Usage Notes & Limitations

> ⚠️ **Scaling Warning:** Because this module relies on dense matrices, memory and computational requirements scale exponentially as $\mathcal{O}(2^{2n})$ where $n$ is the number of qubits.

* **Qubit Limits:** It is highly recommended to restrict the use of this module to small systems ($n \le 12$ qubits) to prevent memory exhaustion (OOM errors) and extremely long simulation runtimes.
* **Hermiticity:** Functions like `overlap` automatically cast results to real values (`Float64`), assuming the input matrix represents a physical, Hermitian observable.