# Quantum-ManyBody-Sim-Methods

Numerical methods for the study of **quantum many-body dynamics**. This repository was developed during a Master 2 Internship in Theoretical Physics under the supervision of Dr. Jacopo De Nardis (*Collège de France / Université de Cergy CYU*).

Most simulations are implemented in **Julia** (for high-performance linear algebra) or **Python**.

---

## Purpose of this Repository

The primary goal is to explore and implement classical algorithms capable of simulating the evolution of quantum systems, bypassing the exponential growth of the Hilbert space. This project investigates the boundaries of classical simulation through two main perspectives:

1. **State-space compression** (Schrödinger picture)
2. **Operator-space dynamics** (Heisenberg picture)

---

## Repository Architecture

The project is structured into modular source code, step-by-step documentation guides, and interactive notebooks that bring the simulations to life.

```
Quantum-ManyBody-Sim-Methods/
│
├── src/                               # Core simulation modules (Julia)
├── docs/                              # Comprehensive user guides for each module
├── notebooks/                         # Practical use cases and tutorials (Julia/Python)
└── notes/                             # Theoretical summaries (e.g., Quant25)

```

### Core Simulation Modules (`src/` & `docs/`)

The repository abstracts its simulation architectures into dedicated Julia files within `src/`. Each module is paired with an independent markdown guide in `docs/` detailing its mathematical background, API syntax, and parameter choices:

* **Exact Simulation**
    * `src/exact_functions.jl` | Guide: `docs/exact_guide.md`
    * *Description:* Computes exact, uncompressed operator evolution via dense matrix operations. Ideal for validating approximate methods on small system sizes (typically less than 12 qubits) before scaling up.


* **Matrix Product Operators (MPO)**
    * `src/matrix_product_operator_functions.jl` | Guide: `docs/mpo_guide.md`
    * *Description:* Tracks operator growth in the Heisenberg picture using Tensor Networks. It includes tools for local depolarizing noise injection and automated sensitivity analysis to discover optimal bond dimension limitations.


* **Pauli Propagation**
    * `src/pauli_propagation_functions.jl` | Guide: `docs/pauli_propagation_guide.md`
    * *Description:* Decomposes quantum operators into a sprawling basis of Pauli strings to monitor information diffusion, scrambling, and weight distributions over time.



### Hands-On Demonstrations (`notebooks/`)

The notebooks import and utilize the core source modules to execute physical experiments. They serve as the practical interface for the codebase:

* **`transverse_ising.ipynb`**
    * *The Entry Point:* This is the foundational use case of the repository. It provides a straightforward, easy-to-follow example implementing the Transverse Field Ising Model (TFIM) to show how the exact and MPO modules track physical observables.


* **`random_bricklayer_circuits.ipynb` & `noisy_random_circuit.ipynb`**
    * Advanced setups exploring entangling unitary circuits, operator entanglement growth, and the destructive effects of local noise channels.


* **`01_mps_basics.ipynb` & `02_mps_evolution.ipynb`**
    * Python-based tutorials dedicated to basic Matrix Product State (MPS) manipulation, canonical forms, and Time-Evolving Block Decimation (TEBD).



---

## Theoretical Methods Explained

### 1. Matrix Product States (MPS)

The MPS ansatz decomposes a global wave function into a chain of local, low-rank tensors:

$$\lvert \psi \rangle = \sum_{s_1 \dots s_N} \text{Tr}(A_1^{s_1} A_2^{s_2} \dots A_N^{s_N}) \lvert s_1 \dots s_N \rangle$$

By limiting the internal bond dimension, we can efficiently simulate 1D quantum systems that obey the entanglement area law.

### 2. Operator Entanglement (OE)

When an operator evolves under a chaotic Hamiltonian or circuit, its complexity grows. By treating the operator as a state in Liouville space, we apply Tensor Network compression directly to the operator space. The bond dimension restricts the maximum amount of Operator Space Entanglement Entropy (OSEE) captured, keeping the simulation computationally tractable.

### 3. Pauli Propagation

This approach tracks the explicit time-evolution of an operator $\hat{O}$ by projecting it onto a complete basis of Pauli strings:

$$\hat{O}(t) = \sum_{P \in \{I, X, Y, Z\}^N} c_P(t) \hat{\sigma}$$

Monitoring the coefficients $c_P(t)$ yields clean data regarding quantum chaos, operator weights, and the ballistic or diffusive spreading of information (quantum scrambling).

---

## References & Resources

### Papers & Literature

* **Random Circuits:** [Hydrodynamics of Operator Spreading](https://arxiv.org/abs/2603.20400)
* **Pauli Propagation:** [Operator Growth and Pauli Dynamics](https://arxiv.org/abs/2505.21606)
* **Haar Random Matrices:** [Stats of Quantum Unitaries](https://arxiv.org/abs/math-ph/0609050)
* **Operator Entanglement:** [Dubail & Jacobsen Lecture Notes](https://www.phys.ens.psl.eu/~jacobsen/AMP21_Dubail.pdf)
* **Tensor Networks:** [Time-Evolving Block Decimation (TEBD) Algorithms](https://tensornetwork.org/mps/algorithms/timeevo/tebd.html)

### Core Libraries & Packages

* **Julia Ecosystem:** [Julia Language Documentation](https://docs.julialang.org/en/v1/) | [ITensors.jl Framework](https://docs.itensor.org/ITensors/stable/index.html) | [PauliPropagation.jl](https://github.com/MSRudolph/PauliPropagation.jl) | [LinearAlgebra Stdlib](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/) | [Plots.jl](https://docs.juliaplots.org/stable/) | [LaTeXStrings.jl](https://juliapackages.com/p/latexstrings)
* **Python Ecosystem:** [NumPy Reference](https://numpy.org/doc/stable/) | [SciPy Reference](https://docs.scipy.org/doc/scipy/) | [Matplotlib Visualization](https://matplotlib.org/stable/index.html)

### Additional Material

* **Coursework:** Jacopo De Nardis's lecture series on *Quasi-exact numerical methods for quantum systems* (Master 2 Theoretical Physics).
* **Conference Notes:** Summary from Miles Stoudenmire's talk at Quant25, available in [notes/TN_talk_note_Quant25.pdf](https://github.com/TheoHUETQC/Quantum-ManyBody-Sim-Methods/blob/main/notes/TN_talk_note_Quant25.pdf).
* **Diagramming Tool:** [Overleaf TikZ Tutorial for Beginners (Flowcharts & Graphs)](https://www.overleaf.com/learn)

---

*Note: Since this internship project is currently active, more methods, documentation updates, and notebooks are added regularly.*