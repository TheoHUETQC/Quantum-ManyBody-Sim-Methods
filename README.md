# Quantum-ManyBody-Sim-Methods

Numerical methods for the study of quantum many-body dynamics. This repository was developed during a Master 2 Internship in Theoretical Physics under the supervision of Dr. Jacopo De Nardis (College de France / Université de Cergy CYU).

Most simulations are implemented in Julia (for high-performance linear algebra) or Python.

---

## Purpose of this Repository

The primary goal is to explore and implement classical algorithms capable of simulating the evolution of quantum systems, bypassing the exponential growth of the Hilbert space. This project investigates the boundaries of classical simulation through two main perspectives:

1. State-space compression (Schrödinger picture)
2. Operator-space dynamics (Heisenberg picture)

## Repository Organization

The repository is organized by physical paradigms to reflect the different ways of approaching the many-body problem:

### Tensor Networks

Focuses on methods that exploit the low-entanglement area law of physical states.

    - **mps-states/**: Representation of 1D quantum states as Matrix Product States.
       - *tutorials/*: Basic manipulation, Canonical forms, and Time-Evolving Block Decimation (TEBD).
    - **notes/**: Theoretical summaries from conferences (e.g., Quant25) and literature.

### Heisenberg Picture

Instead of the state $`|\psi \rangle`$, we track the evolution of observables $`\hat{O}(t)`$.
    - **operator-entanglement/**: Uses the MPO (Matrix Product Operator) formalism to simulate operator growth. The complexity is controlled by the bond dimension $`\chi`$, which truncates the entanglement in the operator space.
    - **pauli-propagation/**: A method decomposing operators into a basis of Pauli strings to study their weight distribution and diffusion.

### Stochastic Methods
    - **quantum-monte-carlo/**: Stochastic sampling techniques to overcome the dimensionality curse for specific classes of Hamiltonians.

---

## Methods Explained

### 1. Matrix Product States (MPS)

The MPS ansatz decomposes a global state into local tensors:

$$\lvert \psi \rangle = \sum_{s_1 \dots s_N} \text{Tr}(A_1^{s_1} A_2^{s_2} \dots A_N^{s_N}) \lvert s_1 \dots s_N \rangle$$

By limiting the size of these matrices (the bond dimension), we can efficiently simulate 1D systems with limited entanglement.

### 2. Operator Entanglement (OE)

When an operator evolves, its complexity increases (operator growth). By treating the operator as a state in the Liouville space, we apply Tensor Network techniques to compress it. The Bond Dimension $`\chi`$ determines the maximum amount of Operator Space Entanglement Entropy (OSEE) captured.

### 3. Pauli Propagation

This approach tracks the evolution of an operator $`\hat{O}`$ in the Pauli basis:

$$\hat{O}(t) = \sum_{\sigma \in \{I, X, Y, Z\}^N} c_\sigma(t) \hat{\sigma}$$

It is particularly useful for studying quantum chaos and the ballistic or diffusive spread of information.

---

More coming soon